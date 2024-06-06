import argparse
import os
from time import time
from pathlib import Path as p
import numpy as np

import torch
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR

from utils.data_no_label import load_data_no_label, get_data, Load_Data, get_data_multi
from utils.for_train_no_label import train_AutoEncoder, train_DEKM
from utils.nn import AutoEncoder, DEKM



def get_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('-bs', default=4096, type=int, help='batch size')
    parser.add_argument('-pre_epoch', default=100, type=int, help='epochs for train Autoencoder')
    parser.add_argument('-epoch', default=100, type=int, help='epochs for train DEKM')
    parser.add_argument('-k', default=5, type=int, help='num of clusters')
    parser.add_argument('-take', type=float, default=1., help='the size of data will be used in training')
    parser.add_argument('-save_dir', default='weight', help='location where model will be saved')
    parser.add_argument('-seed', type=int, default=None, help='torch random seed')

    parser.add_argument('--data_directory', type=str,
                        default='/home/singh_a_WMGDS.WMG.WARWICK.AC.UK/wmg/cytecom/Data-Reporting-Tool/data/Second_Example_Antibiotic_Treated_Sets/S-aureus/Antibiotic-Susceptible/SA-S-2hr-Exposure/1/', help='directory containing tiff files')
    parser.add_argument('--main_directory', default='./../../CyteCountData/220623-S.aureus/', type=str)
    parser.add_argument('--base_files', default=['cytecount_0001.tiff','cytecount_0002.tiff','cytecount_0003.tiff','cytecount_0004.tiff'], help='tiff files names to use as base for normalisaton')
    # parser.add_argument('--base_files', nargs='*', help='tiff files names to use as base for normalisaton')
    parser.add_argument('--source_file', default="cytecount_0001.tiff", help='source tiff files to compare residual against')
    parser.add_argument('--one_data', default="no", help='yes when using only one sample for training')

    parser.add_argument('--colors', type=list,
                        default=[[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [0, 255, 255], [255, 0, 255], [0, 127, 127], [127, 127, 0], [127, 0, 127]],
                        help='color for the different clusters format [(r,g,b), (r1,g1,b1),..]')
    parser.add_argument('--lower_thresh', default=200, type=int, help='save a histogram image')
    parser.add_argument('--higher_thresh', default=500, type=int, help='save a histogram image')

    parser.add_argument('--rm_bag', default=True, type=bool, help='save a histogram image')

    parser.add_argument('--normalise', type=str, default="yes", help='Normalise data: divide by base')
    parser.add_argument('--log', type=str, default="no", help='Take log of normaised data')


    args = parser.parse_args()
    return args
    

def main():
    arg = get_arg()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if not os.path.exists(arg.save_dir):
        os.makedirs(arg.save_dir, exist_ok=True) 
    else:
        for path in p(arg.save_dir).glob('*.png'):
            path.unlink()
        
    if arg.seed is not None:
        torch.manual_seed(arg.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # tr_ds = get_data(arg)
    if arg.one_data == "yes":
        tr_ds = get_data(arg)
        # if arg.auc:
        #     data = create_AUC_data(data, arg)
    else:
        dir_list = next(os.walk(arg.main_directory))[1]
        tr_ds = None
        for idx, dir in enumerate(dir_list):
            E_num = arg.main_directory + dir
            antibiotics = [1,2,3]
            for antibiotic in antibiotics:
                arg.data_directory = E_num + "/" + str(antibiotic)
                data_one = get_data_multi(arg, idx, antibiotic)
                # if arg.auc:
                #     data_one = create_AUC_data(data_one, arg)
                if tr_ds is None:
                    tr_ds = data_one
                else:
                    tr_ds = np.vstack((tr_ds, data_one))
    # train_dataloader = torch.utils.data.DataLoader(tr_ds, batch_size=arg.bs, shuffle=True, drop_last=True)
    print('\ndata shape:', tr_ds.shape)
    tr_ds = torch.tensor(tr_ds)
    print('\ntrain num:', tr_ds.shape[0])
    # print('test num:', test_ds.shape[0])
    
    # train AutoEncoder
    ae = AutoEncoder().to(device)    
    print(f'\nAE param: {ae.num_param():.2f} M')
    opt = Adam(ae.parameters())
    t0 = time()
    train_AutoEncoder(ae, opt, tr_ds[:,:45], arg.bs, device, arg.pre_epoch, arg.save_dir)
    t1 = time()
    
    # train DEKM
    print('\nload the best encoder and build DEKM ...')
    dekm = DEKM(torch.load(f'{arg.save_dir}/pretrain_AE.pt').encoder).to(device)  
    print(f'DEC param: {dekm.num_param():.2f} M') 
    opt = Adam(dekm.parameters())
    t2 = time()
    train_DEKM(dekm, opt, tr_ds, arg.bs, arg.k, device, arg.epoch, arg.save_dir)
    t3 = time()
    
    print(f'\ntrain AE time: {t1 - t0:.2f} s')
    print(f'train DEKM time: {t3 - t2:.2f} s')



if __name__ == '__main__':
    main()
    