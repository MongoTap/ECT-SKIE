import torch
from torch.utils.data import DataLoader
from dataset.dataset import CSDataset_wiRaw, CSDataset_woRaw
from torch.nn.utils.rnn import pad_sequence
import numpy as np


def generate_mask(len_list):
    '''
    need input like this: tensor([124, 59, 15, 177,...])
    :param len_list:
    :return:
    '''
    sequence_length = torch.LongTensor(len_list)
    batch_size = len_list.size(0)
    max_len = len_list.max()
    seq_range = torch.arange(0, max_len)
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_length_expand = sequence_length.unsqueeze(1).expand_as(seq_range_expand)
    ''' or '''
    mask = torch.zeros(batch_size, max_len, dtype=torch.float)
    for e_id, src_len in enumerate(len_list):
        mask[e_id, :src_len] = 1

    return seq_range_expand >= seq_length_expand
    # return torch.where(seq_range_expand >= seq_length_expand, 1., 0.)


def generate_attention_mask(x_len_tensor):
    max_len = x_len_tensor.max()
    att_mask = []
    for length in x_len_tensor:
        mask = torch.ones(max_len, max_len)
        mask[:length, :length] = 0.
        att_mask.append(mask)
    return torch.stack(att_mask).detach()


def collate_fn_woRaw(batch):
    # x_pre = []
    # x_qa = []
    x_qa_mean = []
    # pre_len = []
    # qa_len = []
    y = []
    trans_len = []
    trans = []

    for b in batch:
        # x_pre.append(b['x_pre'])
        # pre_len.append(len(b['x_pre']))
        # x_qa.append(b['x_qa'])
        x_qa_mean.append(torch.mean(b['x_qa'], dim=0))
        # qa_len.append(len(b['x_qa']))
        trans_len.append(len(b['x_pre']) + len(b['x_qa']))
        y.append(b['y'])
        trans.append(torch.cat([b['x_pre'], b['x_qa']]))

    # padded_pre = pad_sequence(x_pre, batch_first=True, padding_value=0.)
    # padded_qa = pad_sequence(x_qa, batch_first=True, padding_value=0.)
    padded_trans = pad_sequence(trans, batch_first=True, padding_value=0.)
    x_len_tensor = torch.as_tensor(trans_len)
    mask = generate_mask(x_len_tensor)
    return {
        'y': torch.from_numpy(np.array(y)).float().view(-1, 1),
        # 'pre_len': pre_len,  # list
        # 'qa_len': qa_len,  # list
        'trans_len': trans_len,
        # 'padded_pre': padded_pre,
        # 'padded_qa': padded_qa,
        'mean_qa': torch.stack(x_qa_mean, dim=0),
        'padded_trans': padded_trans,
        'mask': mask,
        'att_mask': None
    }


def collate_fn_wiRaw(batch):
    x_pre = []
    x_qa = []
    x_qa_mean = []
    trans = []
    # pre_raw = []
    qa_raw = []
    pre_len = []
    qa_len = []
    trans_len = []
    trans_raw = []
    y = []
    path_list = []

    for b in batch:
        # x_pre.append(b['x_pre'])
        pre_len.append(len(b['x_pre']))
        # x_qa.append(b['x_qa'])
        x_qa_mean.append(torch.mean(b['x_qa'], dim=0))
        qa_len.append(len(b['x_qa']))
        # trans_len.append(len(b['x_pre']) + len(b['x_qa']))
        trans_len.append(len(b['x_pre']))
        # pre_raw.append(b['pre_raw'])
        # qa_raw.append(b['qa_raw'])
        y.append(b['y'])
        path_list.append(b['path'])
        trans.append(torch.cat([b['x_pre'], b['x_qa']]))
        trans_raw.append(b['pre_raw'] + b['qa_raw'])

    padded_trans = pad_sequence(trans, batch_first=True, padding_value=0.)
    x_len_tensor = torch.as_tensor(trans_len)
    mask = generate_mask(x_len_tensor)
    return {
        'y': torch.from_numpy(np.array(y)).float().view(-1, 1),
        # 'pre_len': pre_len,  # list
        # 'qa_len': qa_len,  # list
        # 'padded_pre': padded_pre,
        # 'padded_qa': padded_qa,
        # 'pre_raw': pre_raw,
        # 'qa_raw': qa_raw,
        'path': path_list,
        'trans_len': trans_len,
        'mean_qa': torch.stack(x_qa_mean, dim=0),
        'trans_raw': trans_raw,
        'padded_trans': padded_trans,
        'mask': mask,
        'att_mask': None
    }


def collate_fn_wiRaw_attMask_need(batch):
    packed_batch = collate_fn_wiRaw(batch)
    x_len = torch.as_tensor(packed_batch['trans_len'])
    att_mask = generate_attention_mask(x_len)
    packed_batch['att_mask'] = att_mask
    return packed_batch


def collate_fn_woRaw_attMask_need(batch):
    packed_batch = collate_fn_woRaw(batch)
    x_len = torch.as_tensor(packed_batch['trans_len'])
    att_mask = generate_attention_mask(x_len)
    packed_batch['att_mask'] = att_mask
    return packed_batch


def load_test_dataset(config, sample_years=[2018], load_ratio=1.):
    need_raw_text = config.train.need_raw_text
    need_att_mask = config.train.need_att_mask
    if need_raw_text:
        test_dataset = CSDataset_wiRaw(sample_years, label_name=config.data.label, load_ratio=load_ratio)
    else:
        test_dataset = CSDataset_woRaw(sample_years, label_name=config.data.label, load_ratio=load_ratio)

    if need_raw_text and need_att_mask:
        collate_fn = collate_fn_wiRaw_attMask_need
    if need_raw_text and not need_att_mask:
        collate_fn = collate_fn_wiRaw
    if not need_raw_text and need_att_mask:
        collate_fn = collate_fn_woRaw_attMask_need
    if not need_raw_text and not need_att_mask:
        collate_fn = collate_fn_woRaw

    test_dataloader = DataLoader(test_dataset,
                                 batch_size=config.train.batch_size,
                                 shuffle=False,
                                 num_workers=0,
                                 pin_memory=True,
                                 collate_fn=collate_fn,
                                 drop_last=False)
    return test_dataloader


def load_train_dataset(config,
                       train_years=[2015, 2016],
                       train_load_ratio=1.,
                       test_years=[2017],
                       test_load_ratio=1.):
    train_dataset = CSDataset_woRaw(train_years, label_name=config.data.label, load_ratio=train_load_ratio)
    eval_dataset = CSDataset_woRaw(test_years, label_name=config.data.label, load_ratio=test_load_ratio)

    if config.train.need_att_mask:
        collate_fn = collate_fn_woRaw_attMask_need
    else:
        collate_fn = collate_fn_woRaw

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=config.train.batch_size,
                                  shuffle=True,
                                  num_workers=0,
                                  pin_memory=True,
                                  collate_fn=collate_fn,
                                  drop_last=True)

    eval_dataloader = DataLoader(eval_dataset,
                                 batch_size=config.train.batch_size,
                                 shuffle=True,
                                 num_workers=0,
                                 pin_memory=True,
                                 collate_fn=collate_fn,
                                 drop_last=True)

    return train_dataloader, eval_dataloader
