from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

import json
import numpy as np
from numpy.linalg import eig
import pandas as pd
from matplotlib import pyplot as plt

import torch
from torch import nn
from torch.nn import Parameter

from .for_eval import evaluate, accuracy



data_plot = None


def train_AutoEncoder(model, 
                      optimizer, 
                      tr_data, 
                      batch_size, 
                      device, 
                      n_epoch, 
                      save_dir):
    
    print('begin train AutoEncoder ...')
    
    model.train()
    mse = nn.MSELoss()
    n_sample = tr_data.shape[0]
    n_batch = n_sample // batch_size if n_sample % batch_size == 0 else n_sample // batch_size + 1
    tr_loss_h = History('min')
    tr_loss_batch = []

    for epoch in range(1, n_epoch + 1):
        print(f'\nEpoch {epoch}:')
        print('-' * 10)
        
        loss, tr_data = 0., tr_data[torch.randperm(n_sample)]
        for i in range(n_batch):
            x = tr_data[i * batch_size : (i + 1) * batch_size]
            x = x.float().to(device)
            optimizer.zero_grad()
            _, gen = model(x)
            batch_loss = mse(x, gen)
            batch_loss.backward()
            optimizer.step()

            if x.shape[0]==batch_size:
                tr_loss_batch.append(batch_loss.item())
            loss += batch_loss * x.shape[0]
            print(f'{i + 1}/{n_batch}', end='\r')


        loss /= n_sample
        tr_loss_h.add(loss)
        if tr_loss_h.better:
            torch.save(model, f'{save_dir}/pretrain_AE.pt')
        if epoch%10==0:
            torch.save(model, f'{save_dir}/pretrain_AE_{epoch}.pt')
        print(f'tr loss : {loss.item():.4f}  min tr loss : {tr_loss_h.best.item():.4f}')        
        print(f'lr: {optimizer.param_groups[0]["lr"]}')

    print('*' * 50)
    print('End AutoEncoder training !')
    print('saving loss to {save_dir}/train_loss_AE.json')
    with open(f'{save_dir}/train_loss_AE.json', 'w') as f:
        json.dump(tr_loss_h.history, f)
    with open(f'{save_dir}/train_loss_batch_AE.json', 'w') as f:
        json.dump(tr_loss_batch, f)
    print('*' * 50)


def train_DEKM(model, 
               optimizer, 
               tr_ds, 
               batch_size, 
               n_cluster, 
               device, 
               n_epoch, 
               save_dir):
    
    print('begin train DEKM ...')

    mse = nn.MSELoss()
    n_sample = tr_ds.shape[0]
    n_batch = n_sample // batch_size if n_sample % batch_size == 0 else n_sample // batch_size + 1
    # tr_acc_h = History('max')

    tr_loss_h = History('min')
    tr_loss_batch = []
    
    X = tr_ds[:,:45].float().to(device)
    for epoch in range(1, n_epoch + 1):
        print(f'\nEpoch {epoch}:')
        print('-' * 10)

        # get learning target
        H, C, U = train_kmeans(model, X, n_cluster)
        V = get_V(H, C, U, n_cluster)
        H_new = H @ V
        U_new = U @ V
        H_new[:, -1] = U_new[:, -1][C]
        # acc = accuracy(label.numpy(), C)
                
        # save model
        # tr_acc_h.add(acc)
        if tr_loss_h.better:
            model.center = Parameter(torch.tensor(U, device=device), False)
            torch.save(model, f'{save_dir}/DEKM.pt')
        if epoch%10==0:
            model.center = Parameter(torch.tensor(U, device=device), False)
            torch.save(model, f'{save_dir}/DEKM_{epoch}.pt')

        # shuffle
        idx = torch.randperm(n_sample)
        X, H_new = X[idx], H_new[idx]
        
        # update weight
        model.train()
        loss_all = 0
        for i in range(n_batch):
            optimizer.zero_grad()
            Y_new = torch.from_numpy(H_new[i * batch_size : (i + 1) * batch_size]).to(device)
            Y = model.encode(X[i * batch_size : (i + 1) * batch_size])
            loss = mse(Y, Y_new)
            loss.backward()
            optimizer.step()

            if Y_new.shape[0]==batch_size:
                tr_loss_batch.append(loss.item())
            print(f'{i + 1}/{n_batch}', end='\r')
            loss_all += loss * Y_new.shape[0]
        # print(f'Acc: {acc:.4f}')
        print(f'lr: {optimizer.param_groups[0]["lr"]}')
        loss_all /= n_sample
        tr_loss_h.add(loss_all)

    # pd.DataFrame({'Accuracy': tr_acc_h.history}).to_excel(f'{save_dir}/train.xlsx', index=False)
    print('saving loss to {save_dir}/train_loss_DEKM.npy')
    with open(f'{save_dir}/train_loss_DEKM.json', 'w') as f:
        json.dump(tr_loss_h.history, f)
    with open(f'{save_dir}/train_loss_batch_DEKM.json', 'w') as f:
        json.dump(tr_loss_batch, f)


    # evaluate
    print('\nload the best DEKM ...')
    print('*' * 50)    
    model = torch.load(f'{save_dir}/DEKM.pt', device).eval()
    print('Evaluate the test data ...')
    x = tr_ds[:,:45].float().to(device)
    with torch.no_grad():
        H = model.encode(x)
        C = model.get_distance(H).min(1)[1].cpu().numpy()
    # x = x.to("cpu")
    idx = torch.randperm(tr_ds[:,0].numel())
        # acc = accuracy(C, y)
        
    H, C = H[idx][:10000].cpu().numpy(), C[idx][:10000]
    H_2D = TSNE(2).fit_transform(H)
    plt.scatter(H_2D[:, 0], H_2D[:, 1], 16, C, cmap='Paired')
    # plt.title(f'Test data\nAccuracy: {acc:.4f}')
    plt.savefig(f'{save_dir}/test.png')
    
    # print(f'test acc {acc:4f}')
    print('End DEKM training !')
    print('*' * 50)
    

def train_kmeans(model, X, n_cluster):
    model.eval()
    with torch.no_grad():
        H = model.encode(X).cpu().numpy()

    kmeans = KMeans(n_cluster).fit(H)
        
    return H, kmeans.labels_, kmeans.cluster_centers_


def get_V(H, C, U, n_cluster):
    H_sub_U = H - U[C]
    Sw = np.sum([H_sub_U[C == i].T @ H_sub_U[C == i] for i in range(n_cluster)], 0)
    eig_value, V = eig(Sw)
    sort_idx = np.argsort(eig_value)
    eig_value = eig_value[sort_idx]
    #V = V[:, sort_idx].T  # it seems that V.T will get bad result
    V = V[:, sort_idx]
    return V
          
    
class History:
    def __init__(self, target='min'):
        self.value = None
        self.best = float('inf') if target == 'min' else 0.
        self.n_no_better = 0
        self.better = False
        self.target = target
        self.history = [] 
        self._check(target)
        
    def add(self, value):
        if self.target == 'min' and value < self.best:
            self.best = value
            self.n_no_better = 0
            self.better = True
        elif self.target == 'max' and value > self.best:
            self.best = value
            self.n_no_better = 0
            self.better = True
        else:
            self.n_no_better += 1
            self.better = False
            
        self.value = value
        self.history.append(value.item())
        
    def _check(self, target):
        if target not in {'min', 'max'}:
            raise ValueError('target only allow "max" or "min" !')

