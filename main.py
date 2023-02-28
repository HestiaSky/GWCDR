import argparse
import numpy as np
import torch
import time
from torch.utils.data import Dataset, DataLoader
from data_utils import BipartiteGraphDataset
from model import *
from eval import *
from utils import *


MODEL = {'VanillaMF': VanillaMF, 'LightGCN': LightGCN}


def set_requires_grad(mp, requires_grad=True):
    for param in mp.parameters():
        param.requires_grad = requires_grad


def parse_args():
    config_args = {
        'lr': 0.001,
        'dropout': 0.3,
        'cuda': 0,
        'epochs': 1000,
        'weight_decay': 1e-4,
        'seed': 42,
        'model': 'VanillaMF',
        'dim': 64,
        'layers': 3,
        'dataset': 'Avideo',
        'topk': [5, 10],
        'patience': 5,
        'eval_freq': 50,
        'lr_reduce_freq': 500,
        'save_freq': 200,
        'neg_num': 10,
        'batch_size': 1024,
        'gamma': 0.5,
        'transfer': False,
        'source_dataset': 'Avideo',
        'pretrain_epoch': 100,
        'distance': 'wdsk',
        'save': 0,
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
print(str(model))

optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_reduce_freq, gamma=float(args.gamma))
tot_params = sum([np.prod(p.size()) for p in model.parameters()])
print(f'Total number of parameters: {tot_params}')
if args.cuda is not None and int(args.cuda) >= 0:
    model = model.to(args.device)

if args.transfer:
    pretrained_model = torch.load('datasets/' + args.source_dataset + '/' + args.model + '_' + str(args.epochs) + '.pth')
    source_inter = pretrained_model['inter'].to(args.device)
    source_inter_layers = pretrained_model['inter_layers']
    source_inter_layers = [inter.to(args.device) for inter in source_inter_layers]
    torch.cuda.empty_cache()


def train(epoch):
    model.train()
    t = time.time()
    dataset.uniform_sampling()
    avg_loss = 0.
    batch_num = len(dataset.S) // args.batch_size + 1
    for i in range(batch_num):
        batch = dataset.S[i * args.batch_size: (i + 1) * args.batch_size] \
            if (i + 1) * args.batch_size <= len(dataset.S) else dataset.S[i * args.batch_size:]
        users, pos_items, neg_items = torch.LongTensor(batch[:, 0]).to(args.device), \
                                      torch.LongTensor(batch[:, 1]).to(args.device), \
                                      torch.LongTensor(batch[:, 2]).to(args.device)
        loss, reg_loss, pos_inter, pos_inter_layers = model.bpr_loss(users, pos_items, neg_items)
        loss = loss + reg_loss * args.weight_decay
        if args.transfer and epoch > args.pretrain_epoch:
            ot_loss = model.ot_loss(pos_inter, source_inter)
            loss += ot_loss * 0.1
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        avg_loss += loss.cpu().item()

    lr_scheduler.step()
    avg_loss /= (len(dataset.S) // args.batch_size) + 1
    print(f'Average loss:{avg_loss} \n Epoch time: {time.time()-t} \n')


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


# Train Model
t_total = time.time()
for epoch in range(args.epochs):
    print(f'Epoch {epoch}')
    if args.transfer and epoch == args.pretrain_epoch:
        dataset.uniform_sampling()
        batch = dataset.S[:args.batch_size]
        users, pos_items, neg_items = torch.LongTensor(batch[:, 0]).to(args.device), \
                                      torch.LongTensor(batch[:, 1]).to(args.device), \
                                      torch.LongTensor(batch[:, 2]).to(args.device)
        loss, reg_loss, pos_inter, _ = model.bpr_loss(users, pos_items, neg_items)
        model.updata_feature_a(pos_inter, source_inter)
        torch.cuda.empty_cache()
    train(epoch)
    torch.cuda.empty_cache()
    if (epoch + 1) % args.eval_freq == 0:
        test()
        torch.cuda.empty_cache()
    if (epoch + 1) % args.save_freq == 0 and args.save == 1:
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
        state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch,
                 'inter': inters, 'inter_layers': inters_layers}
        torch.save(state, 'datasets/' + args.dataset + '/' + args.model + '_' + str(epoch + 1) + '.pth')
        torch.cuda.empty_cache()

print(f'Model training finished! Total time is {time.time()-t_total}')





