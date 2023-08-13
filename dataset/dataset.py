import logging
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


def loading_dataset(years: list, label_name: str, loading_raw: bool = False, load_ratio=1.) -> dict:
    '''
    loading_dataset
    :param load_ratio: The proportion of the dataset loaded.
    Positive values represent the previous percentage, and negative values represent the last percentage.
    :return: data
    '''
    load_num = 0
    # if label_name not in ['firm_std_10_post', 'firm_std_20_post', 'firm_std_60_post']:
    #     raise AssertionError("not in ['firm_std_10_post', 'firm_std_20_post', 'firm_std_60_post']")
    logging.info('loading the Dataset:')
    x_pre = []
    x_qa = []
    y = []
    x_pre_raw = []
    x_qa_raw = []
    path = []
    for year in years:
        with open(f'dataset/embeddings/embeddings_{year}_cpu.pkl', 'rb') as fIn:
            stored_datas = pickle.load(fIn)
            for doc in tqdm(stored_datas):
                if doc[label_name] == 0.:
                    continue
                x_pre.append(doc['pre'])  # tensor: N * d
                qa = []
                for qa_round in doc['qa']:
                    qa.append(qa_round['q'])
                    qa.append(qa_round['a'])
                x_qa.append(torch.cat(qa))
                # x_qa.append(doc['qa']) # [{'q': tensor, 'a': tensor}, {...}]
                y.append(doc[label_name])
        if loading_raw:
            with open(f'dataset/embeddings/doc_info_{year}.pkl', 'rb') as fIn:
                stored_datas = pickle.load(fIn)
                for doc_info in tqdm(stored_datas):
                    if doc[label_name] == 0.:
                        continue
                    x_pre_raw.append(doc_info['pre'])
                    qa = []
                    for qa_round in doc_info['qa']:
                        qa.extend(qa_round['q'])
                        qa.extend(qa_round['a'])
                    x_qa_raw.append(qa)
                    path.append(doc_info['path'])

    load_num = int(len(y) * load_ratio)
    if load_num == 0:
        raise AssertionError("the number of dataset loaded can not be zero.")
    elif load_num > 0:
        return {
            'x_pre': x_pre[:load_num],
            'x_qa': x_qa[:load_num],
            'y': y[:load_num],
            'x_pre_raw': x_pre_raw[:load_num],
            'x_qa_raw': x_qa_raw[:load_num],
            'path': path[:load_num]
        }
    else:
        load_num = -load_num
        return {
            'x_pre': x_pre[load_num:],
            'x_qa': x_qa[load_num:],
            'y': y[load_num:],
            'x_pre_raw': x_pre_raw[load_num:],
            'x_qa_raw': x_qa_raw[load_num:],
            'path': path[load_num:]
        }


class CSDataset_woRaw(Dataset):

    def __init__(self, years: list, label_name='firm_std_10_post', load_ratio=1.):
        super(CSDataset_woRaw, self).__init__()
        data = loading_dataset(years, label_name, loading_raw=False, load_ratio=load_ratio)
        self.x_pre = data['x_pre']
        self.x_qa = data['x_qa']
        self.y = np.log(data['y'])

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return {
            'x_pre': self.x_pre[index],
            'x_qa': self.x_qa[index],
            'y': self.y[index],
        }


class CSDataset_wiRaw(Dataset):

    def __init__(self, years: list, label_name='firm_std_10_post', load_ratio=1.):
        super(CSDataset_wiRaw, self).__init__()
        self.label_name = label_name
        data = loading_dataset(years, label_name, loading_raw=True, load_ratio=load_ratio)
        self.x_pre = data['x_pre']
        self.x_qa = data['x_qa']
        self.y = np.log(data['y'])
        self.x_pre_raw = data['x_pre_raw']
        self.x_qa_raw = data['x_qa_raw']
        self.path = data['path']

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return {
            'x_pre': self.x_pre[index],
            'x_qa': self.x_qa[index],
            'pre_raw': self.x_pre_raw[index],
            'qa_raw': self.x_qa_raw[index],
            'y': self.y[index],
            'path': self.path[index]
        }