import logging
from copy import deepcopy
import os
import numpy as np
from sklearn.metrics import mean_absolute_error
import torch
from tqdm import tqdm
import pandas as pd
import pickle
import wandb
from utils.firm_sim import firm_sim_visualize


class Trainer(object):

    def __init__(self,
                 args,
                 config,
                 model,
                 optimizer=None,
                 logger=None,
                 accelerator=None,
                 risk_predictor=None,
                 risk_optimizer=None):
        self.need_raw_text = True
        self.args = deepcopy(args)
        self.config = deepcopy(config)
        self.model = model
        self.optimizer = optimizer
        self.tb_logger = logger
        self.accelerator = accelerator
        torch.set_num_threads(8)
        self.train_step = 0
        self.eval_step = 0
        self.redundancy_beta = self.config.train.redundancy_beta
        self.beta = self.config.train.beta
        self.predict_risk = self.config.train.predict_risk
        self.risk_preditor = risk_predictor
        self.risk_optimizer = risk_optimizer

    def train(self, train_dataloader, eval_dataloader, is_continue=False):
        if self.predict_risk:
            self.train_predictor(train_dataloader=train_dataloader,
                                 eval_dataloader=eval_dataloader,
                                 is_continue=is_continue)
            return
        if is_continue:
            logging.info("continue to the previous training...")
            self.load_pretrained_model()
        best_score = float('inf')
        for epoch in range(self.config.train.n_epochs):
            if self.accelerator.is_local_main_process:
                logging.info(f"Epoch {epoch}")
            '''train'''
            self.model.train()
            for i, data in tqdm(enumerate(train_dataloader),
                                total=len(train_dataloader),
                                disable=not self.accelerator.is_local_main_process):
                self.optimizer.zero_grad()
                input_x = [
                    data['padded_trans'], data['trans_len'], data['mean_qa'], data['mask'], data['att_mask']
                ]
                NCEloss, ConciseLoss, Uniformloss, Redundancyloss, res_mask, anchor, utimate_selected_sents = self.model(
                    input_x)
                # backward_loss = NCEloss + self.redundancy_beta * Redundancyloss + self.beta * ConciseLoss
                backward_loss = NCEloss + self.beta * ConciseLoss
                # backward_loss = NCEloss + self.beta * ConciseLoss + risk_loss
                # backward_loss = NCEloss + self.redundancy_beta * Redundancyloss
                self.accelerator.backward(backward_loss)
                self.optimizer.step()
                self.log_output('train_NCELoss', NCEloss, 'train')
                self.log_output('train_ConciseLoss', ConciseLoss, 'train')
                self.log_output('UniformLoss', Uniformloss, 'train')
                self.log_output('train_RedundancyLoss', Redundancyloss, 'train')
                self.log_output('train_loss', backward_loss.item(), 'train')
                self.log_output('uniformity_loss', Uniformloss, 'train')
            '''eval'''
            self.model.eval()
            total_eval_loss = []
            with torch.no_grad():
                for i, data in tqdm(enumerate(eval_dataloader),
                                    total=len(eval_dataloader),
                                    disable=not self.accelerator.is_local_main_process):
                    input_x = [
                        data['padded_trans'], data['trans_len'], data['mean_qa'], data['mask'],
                        data['att_mask']
                    ]
                    NCEloss, ConciseLoss, Uniformloss, Redundancyloss, res_mask, anchor, utimate_selected_sents = self.model(
                        input_x)
                    # eval_loss = NCEloss + self.redundancy_beta * Redundancyloss + self.beta * ConciseLoss
                    eval_loss = NCEloss + self.beta * ConciseLoss
                    # eval_loss = NCEloss + self.beta * ConciseLoss + risk_loss
                    # eval_loss = NCEloss + self.redundancy_beta * Redundancyloss
                    total_eval_loss.append(eval_loss.item())
                    self.log_output('eval_NCELoss', NCEloss, 'eval')
                    self.log_output('eval_ConciseLoss', ConciseLoss, 'eval')
                    self.log_output('eval_RedundancyLoss', Redundancyloss, 'eval')
                    self.log_output('eval_loss', eval_loss.item(), 'eval')
                    self.log_output('Ablation_evalloss', NCEloss + ConciseLoss, 'eval')
                    self.log_output('Ablation_evalloss_v2', NCEloss + self.beta * ConciseLoss, 'eval')
            cur_score = np.mean(total_eval_loss)
            if cur_score < best_score:
                if self.accelerator.is_local_main_process:
                    logging.info(
                        f"best score is: {best_score}, current score is: {cur_score}, save best_checkpoint.pth"
                    )
                best_score = cur_score
                self.save_checkpoint('best_checkpoint.pth')

        self.save_checkpoint('latest_checkpoint.pth')
        if self.args.monitor == 'wandb':
            wandb.finish()

    def train_predictor(self, is_continue, train_dataloader, eval_dataloader):
        mse_loss = torch.nn.MSELoss()
        self.load_pretrained_backbone()
        self.model.eval()
        if is_continue:
            logging.info("continue to the previous predictor training...")
            self.load_predictor_model()
        best_score = float('inf')
        for epoch in range(self.config.train.n_epochs):
            if self.accelerator.is_local_main_process:
                logging.info(f"Epoch {epoch}")
            '''train'''
            self.risk_preditor.train()
            for i, data in tqdm(enumerate(train_dataloader),
                                total=len(train_dataloader),
                                disable=not self.accelerator.is_local_main_process):
                self.risk_optimizer.zero_grad()
                input_x = [
                    data['padded_trans'], data['trans_len'], data['mean_qa'], data['mask'], data['att_mask']
                ]
                label = data['y']
                NCEloss, ConciseLoss, Uniformloss, Redundancyloss, res_mask, mean_selected_sent, utimate_selected_sents = self.model(
                    input_x)
                y_pred = self.risk_preditor(mean_selected_sent)
                backward_loss_1 = NCEloss + self.redundancy_beta * Redundancyloss + self.beta * ConciseLoss
                backward_loss_2 = mse_loss(y_pred, label)
                backward_loss = 0.5 * backward_loss_1 + backward_loss_2
                self.accelerator.backward(backward_loss)
                self.risk_optimizer.step()
                self.log_output('train_Riskloss', backward_loss_2, 'train')
                self.log_output('train_loss', backward_loss, 'train')
            '''eval'''
            self.risk_preditor.eval()
            self.model.eval()
            total_eval_loss = []
            with torch.no_grad():
                for i, data in tqdm(enumerate(eval_dataloader),
                                    total=len(eval_dataloader),
                                    disable=not self.accelerator.is_local_main_process):
                    input_x = [
                        data['padded_trans'], data['trans_len'], data['mean_qa'], data['mask'],
                        data['att_mask']
                    ]
                    label = data['y']
                    NCEloss, ConciseLoss, Uniformloss, Redundancyloss, res_mask, mean_selected_sent, utimate_selected_sents = self.model(
                        input_x)
                    eval_loss_1 = NCEloss + self.redundancy_beta * Redundancyloss + self.beta * ConciseLoss
                    y_pred = self.risk_preditor(mean_selected_sent)
                    eval_loss_2 = mse_loss(y_pred, label)
                    eval_loss = 0.5 * eval_loss_1 + eval_loss_2
                    total_eval_loss.append(eval_loss_2.cpu().numpy())
                    self.log_output('eval_loss', eval_loss, 'train')
                    self.log_output('eval_Riskloss', eval_loss_2, 'eval')

            cur_score = np.mean(total_eval_loss)
            if cur_score < best_score:
                if self.accelerator.is_local_main_process:
                    logging.info(
                        f"best score is: {best_score}, current score is: {cur_score}, save best_predictor.pth"
                    )
                best_score = cur_score
                self.save_predictor('best_predictor.pth')

        self.save_predictor('latest_predictor.pth')
        if self.args.monitor == 'wandb':
            wandb.finish()

    def test(self, test_dataloader, load_pre_train=False):
        # save_container(self.model.model.container.detach().cpu().numpy(), 'end')
        if self.predict_risk:
            self.test_predictor(test_dataloader=test_dataloader)
            return
        if load_pre_train:
            self.load_pretrained_model()
        self.model.eval()
        with torch.no_grad():
            loss_list = []
            anchor_list = []
            paths_list = []
            sent_dist = []
            selecte_sent_rep = []
            for i, data in tqdm(enumerate(test_dataloader),
                                total=len(test_dataloader),
                                disable=not self.accelerator.is_local_main_process):
                input_x = [
                    data['padded_trans'], data['trans_len'], data['mean_qa'], data['mask'], data['att_mask']
                ]
                label = data['y'] if self.predict_risk else None
                NCEloss, ConciseLoss, Uniformloss, containerloss, res_mask, mean_selected_sent, utimate_selected_sents = self.model(
                    input_x, label)
                # test_loss = NCEloss + self.beta * ConciseLoss

                loss_list.append(NCEloss.cpu().numpy())

                # selecte_sent_rep.extend(get_selected_rep(discrete_mask=res_mask, data=data))
                # sent_dist.extend(collect_sent_dist(discrete_mask=res_mask, data=data))

                # visualize the discrete mask
                # if i < 5:
                visualize_mask(discrete_mask=res_mask, data=data)

                # generate firm-simlarity data file
                # anchor_list.append(mean_selected_sent)
                # paths_list.extend(data['path'])

            # # generate firm simlarity file
            # generate_firm_sim_file(anchor_list=anchor_list,
            #                        paths_list=paths_list,
            #                        project_name='intimate_version_firm_sim')
            # with open(f'./sent_dist_ttrep_v2.pkl', 'wb') as fOut:
            #     pickle.dump(sent_dist, fOut, protocol=pickle.HIGHEST_PROTOCOL)
            # with open(f'./selected_rep_EARNIE.pkl', 'wb') as fOut:
            #     pickle.dump(selecte_sent_rep, fOut, protocol=pickle.HIGHEST_PROTOCOL)
            print(f'The NCE loss is {np.mean(loss_list)}')

    def test_predictor(self, test_dataloader):
        self.load_pretrained_backbone()
        self.load_predictor_model()
        self.model.eval()
        self.risk_preditor.eval()
        mse_loss = torch.nn.MSELoss()
        with torch.no_grad():
            prediction_loss = []
            mae_list = []
            for i, data in tqdm(enumerate(test_dataloader),
                                total=len(test_dataloader),
                                disable=not self.accelerator.is_local_main_process):
                input_x = [
                    data['padded_trans'], data['trans_len'], data['mean_qa'], data['mask'], data['att_mask']
                ]
                label = data['y']
                NCEloss, ConciseLoss, Uniformloss, containerloss, res_mask, selected_mean_sent, utimate_selected_sents = self.model(
                    input_x, label)
                y_pred = self.risk_preditor(selected_mean_sent)
                eval_loss = mse_loss(y_pred, label)
                prediction_loss.append(eval_loss.cpu().numpy())
                mae_list.append(mean_absolute_error(y_pred.cpu().numpy(), label.cpu().numpy()))

            print(f'The risk prediction loss is {np.mean(prediction_loss)}')
            print(f'The risk mae is {np.mean(mae_list)}')

    def load_pretrained_model(self):
        if self.args.load_mode == 'best':
            pretrained_data = torch.load(os.path.join(self.args.checkpoint, self.args.pretrained))
            logging.info(f'The pre-trained model【{self.args.pretrained}】 will be loaded.')
            # pretrained_data = torch.load(os.path.join(self.args.checkpoint, 'best_checkpoint.pth'))
        elif self.args.load_mode == 'latest':
            pretrained_data = torch.load(os.path.join(self.args.checkpoint, 'latest_checkpoint.pth'))
        self.model = self.accelerator.unwrap_model(self.model)
        self.model.load_state_dict(pretrained_data)

    def load_pretrained_backbone(self):
        if self.args.load_mode == 'best':
            pretrained_data = torch.load(os.path.join(self.args.checkpoint, 'best_backbone.pth'))
            # pretrained_data = torch.load(os.path.join(self.args.checkpoint, 'best_checkpoint.pth'))
        elif self.args.load_mode == 'latest':
            pretrained_data = torch.load(os.path.join(self.args.checkpoint, 'latest_backbone.pth'))
        self.model = self.accelerator.unwrap_model(self.model)
        self.model.load_state_dict(pretrained_data)

    def load_predictor_model(self):
        if self.args.load_mode == 'best':
            pretrained_data = torch.load(os.path.join(self.args.checkpoint, 'best_predictor.pth'))
        elif self.args.load_mode == 'latest':
            pretrained_data = torch.load(os.path.join(self.args.checkpoint, 'latest_predictor.pth'))
        self.risk_preditor = self.accelerator.unwrap_model(self.risk_preditor)
        self.risk_preditor.load_state_dict(pretrained_data)

    def save_checkpoint(self, model_name):
        # save checkpoint
        self.accelerator.wait_for_everyone()
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        self.accelerator.save(unwrapped_model.state_dict(), os.path.join(self.args.checkpoint, model_name))

    def save_predictor(self, model_name):
        # save checkpoint
        self.accelerator.wait_for_everyone()
        unwrapped_model = self.accelerator.unwrap_model(self.risk_preditor)
        self.accelerator.save(unwrapped_model.state_dict(), os.path.join(self.args.checkpoint, model_name))

    def log_output(self, metric_name, content, train_eval=None):
        if self.accelerator.is_local_main_process:
            if self.args.monitor == 'wandb':
                wandb.log({metric_name: content})
            elif self.args.monitor == 'tensorboard':
                if train_eval == 'train':
                    self.train_step += 1
                    self.tb_logger.add_scalar(metric_name, content, global_step=self.train_step)
                elif train_eval == 'eval':
                    self.eval_step += 1
                    self.tb_logger.add_scalar(metric_name, content, global_step=self.eval_step)
                else:
                    AssertionError('Args [train_eval] must be given when monitor is tensorboard.')


def generate_firm_sim_file(anchor_list, paths_list, project_name):
    anchor_emb = torch.cat(anchor_list)
    with open(f'result/firm_sim/{project_name}.pkl', 'wb') as fOut:
        pickle.dump({'anchor': anchor_emb, 'path': paths_list}, fOut, protocol=pickle.HIGHEST_PROTOCOL)

    firm_sim_visualize(project_name=project_name)


def save_container(index, epoch, container):
    with open(f'result/container/epoch_{epoch}_time_{index}_container.pkl', 'wb') as fOut:
        pickle.dump(container, fOut, protocol=pickle.HIGHEST_PROTOCOL)


def visualize_mask(discrete_mask, data):
    # visualization
    natural_text = data['trans_raw']
    paths = data['path']
    # visual_num = 0
    for raw, selected, length, path in zip(natural_text, discrete_mask, data['trans_len'], paths):
        # each transcript
        score_list = []
        for score in selected[:length]:
            score_list.append(int(score.item()))
        dirs = path.split("/")
        pd.DataFrame({
            'text': raw,
            'basic_score': score_list,
        }).to_csv(f'result/visual_mask/{dirs[2]}-{dirs[3]}', index=False)
        # visual_num += 1
        # if visual_num == 10:
        #     exit()


def collect_sent_dist(discrete_mask, data):
    dist = []
    for selected_pos, length in zip(discrete_mask, data['trans_len']):
        indices = torch.nonzero(selected_pos != 0.).squeeze().cpu().numpy()
        position = indices / length
        dist.append(position)

    return dist


def get_selected_rep(discrete_mask, data):
    reps = data['padded_trans']
    length_list = data['trans_len']
    discrete_mask = torch.where(discrete_mask > 0, 1., 0.)
    selected_sent_rep = []
    for rep, mask, length in zip(reps, discrete_mask, length_list):
        selected_pos = mask[:length]
        sent_rep = rep[:length]
        indices = torch.nonzero(selected_pos == 1.).squeeze()
        print(f'true:{length}, select:{len(indices)}')
        selected_sent_rep.append(sent_rep[indices])
    return selected_sent_rep
