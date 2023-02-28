import argparse
import numpy as np
import torch
import time
from torch.utils.data import Dataset, DataLoader
from data_utils import BipartiteGraphDataset
from model import *
from eval import *


MODEL = {'VanillaMF': VanillaMF, 'LightGCN': LightGCN}


def parse_args():
    config_args = {
        'lr': 0.001,
        'dropout': 0.3,
        'cuda': 4,
        'epochs': 1000,
        'weight_decay': 1e-4,
        'seed': 42,
        'model': 'LightGCN',
        'dim': 64,
        'layers': 3,
        'dataset': 'Avideo',
        'topk': [5, 10, 20],
        'patience': 5,
        'eval_freq': 50,
        'lr_reduce_freq': 500,
        'save_freq': 200,
        'neg_num': 10,
        'batch_size': 1024,
        'gamma': 0.5,
        'transfer': True,
        'source_dataset': 'Avideo',
    }

    parser = argparse.ArgumentParser()
    for param, val in config_args.items():
        parser.add_argument(f"--{param}", default=val)
    args = parser.parse_args()
    return args


args = parse_args()
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
args.device = 'cuda:' + str(args.cuda) if int(args.cuda) >= 0 else 'cpu'

dataset = BipartiteGraphDataset(args, args.dataset)
model = MODEL[args.model](args, dataset)
pretrained_model = torch.load('datasets/' + args.dataset + '/' + args.model + '.pth')
model.load_state_dict(pretrained_model['model'])
print(str(model))

tot_params = sum([np.prod(p.size()) for p in model.parameters()])
print(f'Total number of parameters: {tot_params}')
if args.cuda is not None and int(args.cuda) >= 0:
    model = model.to(args.device)

 inters, inters_layers = [], [[], [], [], []]
 batch_num = dataset.trainDataSize // args.batch_size + 1
 for i in range(batch_num):
     batch = dataset.train_data[i*args.batch_size: (i+1)*args.batch_size] \
         if (i+1)*args.batch_size <= dataset.trainDataSize else dataset.train_data[i*args.batch_size:]
     batch_users = torch.LongTensor(batch['userid'].tolist()).to(args.device)
     batch_items = torch.LongTensor(batch['itemid'].tolist()).to(args.device)
     ratings, inter, inter_layers = model(batch_users, batch_items)
     inters.append(inter)
     for j in range(len(inters_layers)):
         inters_layers[j].append(inter_layers[j])
 inters = torch.cat(inters, dim=0).detach().cpu()
 inters_layers = [torch.cat(inters_layers[i], dim=0).detach().cpu() for i in range(len(inters_layers))]
 state = {'model': model.state_dict(), 'optimizer': pretrained_model['optimizer'], 'epoch': pretrained_model['epoch'],
          'inter': inters, 'inter_layers': inters_layers}
 torch.save(state, 'datasets/' + args.dataset + '/' + args.model + '_' + str(args.epochs) + '.pth')

def test():
    model.eval()
    model.mode = 'test'
    testDict = dataset.testDict
    results = {'Precision': np.zeros(len(args.topk)),
               'Recall': np.zeros(len(args.topk)),
               'AUC': 0.,
               'MRR': np.zeros(len(args.topk)),
               'MAP': np.zeros(len(args.topk)),
               'NDCG': np.zeros(len(args.topk))}
    with torch.no_grad():
        users = list(testDict.keys())
        batch_num = len(users) // args.batch_size + 1
        for i in range(batch_num):
            batch_users = users[i*args.batch_size: (i+1)*args.batch_size] \
                if (i+1)*args.batch_size <= len(users) else users[i*args.batch_size:]
            all_pos = dataset.get_user_pos_items(batch_users)
            groundTruth = [testDict[u] for u in batch_users]
            batch_users = torch.LongTensor(batch_users).to(args.device)

            rating = model.get_rating(batch_users)
            exclude_index = []
            exclude_items = []
            for range_i, items in enumerate(all_pos):
                exclude_index.extend([range_i] * len(items))
                exclude_items.extend(items)
            rating[exclude_index, exclude_items] = -(1 << 10)
            _, rating_K = torch.topk(rating, k=args.topk[-1])
            rating_K = rating_K.cpu().numpy()

            rating = rating.cpu().numpy()
            aucs = [AUC(rating[i], dataset, test_data) for i, test_data in enumerate(groundTruth)]
            results['AUC'] += np.sum(aucs)
            r = getLabel(groundTruth, rating_K)
            for j, k in enumerate(args.topk):
                pre, rec = RecallPrecision_atK(groundTruth, r, k)
                mrr = MRR_atK(groundTruth, r, k)
                map = MAP_atK(groundTruth, r, k)
                ndcg = NDCG_atK(groundTruth, r, k)
                results['Precision'][j] += pre
                results['Recall'][j] += rec
                results['MRR'][j] += mrr
                results['MAP'][j] += map
                results['NDCG'][j] += ndcg

        for key in results.keys():
            results[key] /= float(len(users))
        for j, k in enumerate(args.topk):
            print(f'Precision@{k}: {results["Precision"][j]} \n '
                  f'Recall@{k}: {results["Recall"][j]} \n '
                  f'MRR@{k}: {results["MRR"][j]} \n '
                  f'MAP@{k}: {results["MAP"][j]} \n '
                  f'NDCG@{k}: {results["NDCG"][j]} \n')
        print(f'AUC: {results["AUC"]} \n')


print('Start Test')
test()
