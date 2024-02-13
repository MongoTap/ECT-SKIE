import argparse
import logging
from dataset.loader import load_test_dataset, load_train_dataset
from dataset.loader_toy import load_toy_dataset
from model import *
import yaml
from model.model_loss import *
from model.module import RiskPredictor
from trainer import *
from utils.utils import dict2namespace, dict_merge, get_optimizer, init_monitor, init_workspace, set_logger
from accelerate import Accelerator


def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()['__doc__'])
    parser.add_argument('--doc', type=str, default='ECT-SKIE', help='A string for project name')
    parser.add_argument('--model', type=str, default='Model', help='The model name')
    parser.add_argument('--trainer', type=str, default='Trainer', help='The trainer to execute')
    parser.add_argument('--try_toy', type=str, default=True, help='Whether to try toy samples.')
    parser.add_argument('--pretrained',
                        type=str,
                        default='checkpoint.pth',
                        help='The pretrained model checkpoint name')
    parser.add_argument('--test', type=str, default=False, help='Whether load model for test')
    parser.add_argument('--config', type=str, default='conf.yml', help='Path to the config file')
    parser.add_argument('--comment', type=str, default='', help='A string for experiment comment')
    parser.add_argument('--result', type=str, default='result', help='Path for saving running related data.')
    parser.add_argument('--seed', type=int, default=426, help='Random seed')
    parser.add_argument('--verbose',
                        type=str,
                        default='info',
                        help='Verbose level: info | debug | warning | critical')
    parser.add_argument('--train_continue',
                        type=str,
                        default=False,
                        help='Whether to continue the previous training')
    parser.add_argument('--load_mode',
                        type=str,
                        default='best',
                        help='When args.test = True, load the [best, latest] checkpoint')
    parser.add_argument(
        '--monitor',
        type=str,
        default='tensorboard',
        help='the visualization tool for monitoring the train process. optinal [wandb, tensorboard]')

    args = parser.parse_args()

    # init args
    init_workspace(args)

    # parse config file
    with open(os.path.join('conf', args.config), 'r') as f:
        config_dict = yaml.safe_load(f)
    new_config = dict2namespace(config_dict)

    # setup logger
    set_logger(args)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    return args, new_config, dict_merge(config_dict)


def main():
    # init accelerator
    accelerator = Accelerator()
    # init config
    args, config, config_dict = parse_args_and_config()
    # init model
    model = eval(args.model)()
    model.init_model(config.model)
    logging.info(f"Init the model named 【{args.model}】...")
    num_parameters = sum([l.nelement() for l in model.parameters()])

    if accelerator.is_local_main_process:
        logging.info("Writing log file to {}".format(args.log))
        logging.info("Args = {}".format(args))
        logging.info("Config = {}".format(config))
        logging.info("number of parameters: %d", num_parameters)

    # init optimizer
    optimizer = get_optimizer(config=config, model=model)
    predictor, risk_optimizer = None, None

    # We release a toy samples dataset which is operational.
    if args.try_toy:
        tb_logger = None
        toy_dataloader = load_toy_dataset(config=config)
        model, toy_dataloader, optimizer = accelerator.prepare(model, toy_dataloader, optimizer)
        trainer = eval(args.trainer)(args, config, model, optimizer, tb_logger, accelerator, predictor,
                                     risk_optimizer)
        trainer.test(toy_dataloader, load_pre_train=True)
        return

    if config.train.predict_risk:
        predictor = RiskPredictor(384, [384, 150, 50],
                                  3,
                                  dropout=0.3,
                                  in_LN=True,
                                  hid_LN=True,
                                  out_LN=True,
                                  out=True)
        risk_optimizer = get_optimizer(config=config, model=predictor)
        predictor, risk_optimizer = accelerator.prepare(predictor, risk_optimizer)

    if args.test:
        tb_logger = None
        test_dataloader = load_test_dataset(config=config)
        model, test_dataloader, optimizer = accelerator.prepare(model, test_dataloader, optimizer)
        trainer = eval(args.trainer)(args, config, model, optimizer, tb_logger, accelerator, predictor,
                                     risk_optimizer)
        trainer.test(test_dataloader, load_pre_train=True)
    else:
        # init monitor only if mode = train
        tb_logger = init_monitor(args, wandb_config=config_dict)

        train_dataloader, eval_dataloader = load_train_dataset(config=config)
        model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
            model, optimizer, train_dataloader, eval_dataloader)
        trainer = eval(args.trainer)(args, config, model, optimizer, tb_logger, accelerator, predictor,
                                     risk_optimizer)
        trainer.train(train_dataloader, eval_dataloader, args.train_continue)

        test_dataloader = load_test_dataset(config=config)
        test_dataloader = accelerator.prepare(test_dataloader)
        trainer.test(test_dataloader, load_pre_train=False)


if __name__ == '__main__':
    main()
