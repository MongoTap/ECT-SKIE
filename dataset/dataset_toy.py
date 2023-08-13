import numpy as np
from torch.utils.data import Dataset
import pickle
from tqdm import tqdm
import torch


def loading_toy_samples(label_name: str):
    x = []
    x_qa = []
    y = []
    transcript = []
    qa = []
    path = []
    with open(f'dataset/data/toy_samples.pkl', 'rb') as fIn:
        stored_datas = pickle.load(fIn)
        for doc in tqdm(stored_datas):
            if doc[label_name] == 0.:
                continue
            x.append(doc['transcript_rep'])  # tensor: N * d
            qa = []
            for qa_round in doc['QA_rep']:
                qa.append(qa_round['q'])
                qa.append(qa_round['a'])
            x_qa.append(torch.cat(qa))
            y.append(doc[label_name])

            transcript.append(doc['transcript'])
            qa.append(doc['QA'])
            path.append(doc['path'])

        return {'x': x, 'x_qa': x_qa, 'y': y, 'transcript': transcript, 'qa': qa, 'path': path}


class DatasetToy(Dataset):

    def __init__(self, label_name='firm_std_10_post'):
        super(DatasetToy, self).__init__()
        data = loading_toy_samples(label_name)
        self.x = data['x']
        self.x_qa = data['x_qa']
        self.y = np.log(data['y'])
        self.transcript = data['transcript']
        self.qa = data['qa']
        self.paths = data['path']

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return {
            'x': self.x[index],
            'x_qa': self.x_qa[index],
            'y': self.y[index],
            'transcript': self.transcript[index],
            'qa': self.qa[index],
            'path': self.paths[index]
        }
