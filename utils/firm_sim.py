import torch
import pickle
import pandas as pd
from tqdm import tqdm


def firm_sim_visualize(project_name, topK=15):
    df = pd.read_csv('dataset/updated_index_with_DVs.csv')
    df = df[df['year'].isin([2015, 2016, 2017, 2018])]
    firm = df['conm'].to_list()
    path = df['path'].to_list()

    with open(f'result/firm_sim/{project_name}.pkl', 'rb') as fIn:
        data = pickle.load(fIn)
        paths = data['path']
        rep = data['anchor']
        rep = torch.nn.functional.normalize(rep, dim=-1)
        score = torch.matmul(rep, rep.permute(1, 0))
        print(score.shape)
        diag = torch.diag(score)
        diag = torch.diag_embed(diag)
        score = score - diag
        ranked_score, rank_indices = torch.sort(score, dim=-1, descending=True)
        rk_indices = rank_indices.tolist()

        comn_list = []
        sim_comn = []
        for pth, rk in tqdm(zip(paths, rk_indices), total=len(paths)):
            comn = firm[path.index(pth[21:-4])]
            topK_firm = []
            num = 0
            for indice in rk:
                tg = firm[path.index(paths[indice][21:-4])]
                if tg != comn:
                    topK_firm.append(tg)
                    num += 1
                if num == topK:
                    break
            comn_list.append(comn)
            sim_comn.append(topK_firm)

        pd.DataFrame({
            'firm': comn_list,
            'sim': sim_comn
        }).to_csv(f'result/firm_sim/{project_name}.csv', index=False)
