from dataset.dataset_toy import DatasetToy
from torch.utils.data import DataLoader
import torch
from torch.nn.utils.rnn import pad_sequence
import numpy as np

from dataset.loader import generate_mask


def collate_fn(batch):
    x = []
    x_qa_mean = []
    # qa_raw = []
    x_len = []
    qa_len = []
    trans_raw = []
    y = []
    path_list = []

    for b in batch:
        x_qa_mean.append(torch.mean(b['x_qa'], dim=0))
        qa_len.append(len(b['x_qa']))
        x_len.append(len(b['x']))
        # qa_raw.append(b['qa'])
        y.append(b['y'])
        path_list.append(b['path'])
        x.append(b['x'])
        trans_raw.append(b['transcript'])

    padded_x = pad_sequence(x, batch_first=True, padding_value=0.)
    x_len_tensor = torch.as_tensor(x_len)
    mask = generate_mask(x_len_tensor)
    return {
        'y': torch.from_numpy(np.array(y)).float().view(-1, 1),
        'path': path_list,
        'trans_len': x_len,
        'mean_qa': torch.stack(x_qa_mean, dim=0),
        'trans_raw': trans_raw,
        'mask': mask,
        'padded_trans': padded_x,
        'att_mask': None
    }


def load_toy_dataset(config):
    test_dataset = DatasetToy(label_name=config.data.label)

    toy_dataloader = DataLoader(test_dataset,
                                batch_size=config.train.batch_size,
                                shuffle=False,
                                num_workers=0,
                                pin_memory=True,
                                collate_fn=collate_fn,
                                drop_last=False)
    return toy_dataloader