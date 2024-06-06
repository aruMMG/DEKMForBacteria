import argparse
import os
from time import time
from pathlib import Path as p
from PIL import Image

import torch
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR

from utils.data_no_label import load_data_no_label, get_data, Load_Data
from utils.for_train_no_label import train_AutoEncoder, train_DEKM
from utils.nn import AutoEncoder, DEKM
import matplotlib.pyplot as plt
from utils.nn import AutoEncoder, DEKM
from sklearn.manifold import TSNE
import numpy as np


def get_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('-bs', default=512, type=int, help='batch size')
    parser.add_argument('-pre_epoch', default=10, type=int, help='epochs for train Autoencoder')
    parser.add_argument('-epoch', default=10, type=int, help='epochs for train DEKM')
    parser.add_argument('-k', default=5, type=int, help='num of clusters')
    parser.add_argument('-take', type=float, default=1., help='the size of data will be used in training')
    parser.add_argument('-weight_dir', default='weight/SA_folds/fold3/SA_k3_fold3', help='location where model weight are saved')
    parser.add_argument('-save_dir', default='test', help='location where output will be saved')
    parser.add_argument('-seed', type=int, default=None, help='torch random seed')

    parser.add_argument('--data_directory', type=str, default='./../../CyteCountData/E01070/1/', help='directory containing tiff files')
    parser.add_argument('--base_files', default=['cytecount_0001.tiff','cytecount_0002.tiff','cytecount_0003.tiff','cytecount_0004.tiff'], help='tiff files names to use as base for normalisaton')
    # parser.add_argument('--base_files', nargs='*', help='tiff files names to use as base for normalisaton')
    parser.add_argument('--source_file', default="cytecount_0001.tiff", help='source tiff files to compare residual against')

    parser.add_argument('--colors', type=list,
                        default=[[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [0, 255, 255], [255, 0, 255], [0, 127, 127], [127, 127, 0], [127, 0, 127]],
                        help='color for the different clusters format [(r,g,b), (r1,g1,b1),..]')
    parser.add_argument('--lower_thresh', default=200, type=int, help='save a histogram image')
    parser.add_argument('--higher_thresh', default=500, type=int, help='save a histogram image')

    parser.add_argument('--rm_bag', default=True, type=bool, help='save a histogram image')

    parser.add_argument('--normalise', type=str, default="yes", help='Normalise data: divide by base')
    parser.add_argument('--log', type=str, default="no", help='Take log of normaised data')
    parser.add_argument('--test', type=str, default="no", help='Take log of normaised data')


    args = parser.parse_args()
    return args
    

def test():
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
    
    tr_ds = get_data(arg)
    # tr_ds = Load_Data(data_stack)
    # train_dataloader = torch.utils.data.DataLoader(tr_ds, batch_size=arg.bs, shuffle=True, drop_last=True)
    print('\ntrain num:', tr_ds.shape[0])
    # print('test num:', test_ds.shape[0])
    
    print('\nload the best DEKM ...')
    print('*' * 50)    
    model = torch.load(f'{arg.weight_dir}/DEKM.pt', device).eval()
    print('Evaluate the test data ...')
    x = tr_ds[:,:45].float().to(device)
    with torch.no_grad():
        H = model.encode(x)
        Centers = model.get_distance(H).min(1)[1].cpu().numpy()
    # x = x.to("cpu")
    idx = torch.randperm(tr_ds[:,0].numel())
        # acc = accuracy(C, y)
        
    H, C = H[idx][:1000].cpu().numpy(), Centers[idx][:1000]
    H_2D = TSNE(2).fit_transform(H)
    plt.scatter(H_2D[:, 0], H_2D[:, 1], 16, C, cmap='Paired')
    # plt.title(f'Test data\nAccuracy: {acc:.4f}')
    plt.savefig(f'{arg.save_dir}/test.png')


    cluster_image_colored = np.zeros((1518, 2012, 3), dtype='uint8')
    indepen_cluster_image_colored_0 = np.zeros((1518, 2012, 3), dtype='uint8')
    indepen_cluster_image_colored_1 = np.zeros((1518, 2012, 3), dtype='uint8')
    indepen_cluster_image_colored_2 = np.zeros((1518, 2012, 3), dtype='uint8')
    indepen_cluster_image_colored_3 = np.zeros((1518, 2012, 3), dtype='uint8')
    indepen_cluster_image_colored_4 = np.zeros((1518, 2012, 3), dtype='uint8')
    indepen_cluster_image_colored_5 = np.zeros((1518, 2012, 3), dtype='uint8')
    indepen_cluster_image_colored_6 = np.zeros((1518, 2012, 3), dtype='uint8')
    indepen_cluster_image_colored_7 = np.zeros((1518, 2012, 3), dtype='uint8')
    indepen_cluster_image_colored_8 = np.zeros((1518, 2012, 3), dtype='uint8')
    indepen_cluster_image_colored_9 = np.zeros((1518, 2012, 3), dtype='uint8')
    binary_indepen_cluster_image_0 = np.zeros((1518, 2012), dtype='uint8')
    binary_indepen_cluster_image_1 = np.zeros((1518, 2012), dtype='uint8')
    binary_indepen_cluster_image_2 = np.zeros((1518, 2012), dtype='uint8')
    binary_indepen_cluster_image_3 = np.zeros((1518, 2012), dtype='uint8')
    binary_indepen_cluster_image_4 = np.zeros((1518, 2012), dtype='uint8')
    binary_indepen_cluster_image_5 = np.zeros((1518, 2012), dtype='uint8')
    binary_indepen_cluster_image_6 = np.zeros((1518, 2012), dtype='uint8')
    binary_indepen_cluster_image_7 = np.zeros((1518, 2012), dtype='uint8')
    binary_indepen_cluster_image_8 = np.zeros((1518, 2012), dtype='uint8')
    binary_indepen_cluster_image_9 = np.zeros((1518, 2012), dtype='uint8')
    for x in range(tr_ds.shape[0]):
            i = int(tr_ds[x,tr_ds.shape[1]-2])
            j = int(tr_ds[x,tr_ds.shape[1]-1])
            if Centers[x]==0:
                cluster_image_colored[i,j,:] = (255, 0, 0)
                indepen_cluster_image_colored_0[i,j,:] = (255, 0, 0)
                binary_indepen_cluster_image_0[i,j] = 255
            elif Centers[x]==1:
                cluster_image_colored[i,j,:] = (0, 255, 0)
                indepen_cluster_image_colored_1[i,j,:] = (0, 255, 0)
                binary_indepen_cluster_image_1[i,j] = 255
            elif Centers[x]==2:
                cluster_image_colored[i,j,:] = (0, 0, 255)
                indepen_cluster_image_colored_2[i,j,:] = (0, 0, 255)
                binary_indepen_cluster_image_2[i,j] = 255
            elif Centers[x]==3:
                cluster_image_colored[i,j,:] = (255, 255, 0)
                indepen_cluster_image_colored_3[i,j,:] = (255, 255, 0)
                binary_indepen_cluster_image_3[i,j] = 255
            elif Centers[x]==4:
                cluster_image_colored[i,j,:] = (255, 0, 255)
                indepen_cluster_image_colored_4[i,j,:] = (255, 0, 255)
                binary_indepen_cluster_image_4[i,j] = 255
            elif Centers[x]==5:
                cluster_image_colored[i,j,:] = (0, 255, 255)
                indepen_cluster_image_colored_5[i,j,:] = (0, 255, 255)
                binary_indepen_cluster_image_5[i,j] = 255
            elif Centers[x]==6:
                cluster_image_colored[i,j,:] = (127, 127, 127)
                indepen_cluster_image_colored_6[i,j,:] = (127, 127, 127)
                binary_indepen_cluster_image_6[i,j] = 255
            elif Centers[x]==7:
                cluster_image_colored[i,j,:] = (127, 0, 127)
                indepen_cluster_image_colored_7[i,j,:] = (127, 0, 127)
                binary_indepen_cluster_image_7[i,j] = 255
            elif Centers[x]==8:
                cluster_image_colored[i,j,:] = (127, 0, 0)
                indepen_cluster_image_colored_8[i,j,:] = (127, 0, 0)
                binary_indepen_cluster_image_8[i,j] = 255
            elif Centers[x]==9:
                cluster_image_colored[i,j,:] = (0, 0, 127)                
                indepen_cluster_image_colored_9[i,j,:] = (0, 0, 127)
                binary_indepen_cluster_image_9[i,j] = 255
    
    im = Image.fromarray(cluster_image_colored)
    im1 = Image.fromarray(indepen_cluster_image_colored_0)
    im2 = Image.fromarray(indepen_cluster_image_colored_1)
    im3 = Image.fromarray(indepen_cluster_image_colored_2)
    im4 = Image.fromarray(indepen_cluster_image_colored_3)
    im5 = Image.fromarray(indepen_cluster_image_colored_4)
    im6 = Image.fromarray(indepen_cluster_image_colored_5)
    im7 = Image.fromarray(indepen_cluster_image_colored_6)
    im8 = Image.fromarray(indepen_cluster_image_colored_7)
    im9 = Image.fromarray(indepen_cluster_image_colored_8)
    im10 = Image.fromarray(indepen_cluster_image_colored_9)

    im1_binary = Image.fromarray(binary_indepen_cluster_image_0)
    im2_binary = Image.fromarray(binary_indepen_cluster_image_1)
    im3_binary = Image.fromarray(binary_indepen_cluster_image_2)
    im4_binary = Image.fromarray(binary_indepen_cluster_image_3)
    im5_binary = Image.fromarray(binary_indepen_cluster_image_4)
    im6_binary = Image.fromarray(binary_indepen_cluster_image_5)
    im7_binary = Image.fromarray(binary_indepen_cluster_image_6)
    im8_binary = Image.fromarray(binary_indepen_cluster_image_7)
    im9_binary = Image.fromarray(binary_indepen_cluster_image_8)
    im10_binary = Image.fromarray(binary_indepen_cluster_image_9)


    im.save(f'{arg.save_dir}/output.png')
    im1.save(f'{arg.save_dir}/output_0.png')
    im1_binary.save(f'{arg.save_dir}/binary_output_0.png')
    im2_binary.save(f'{arg.save_dir}/binary_output_1.png')
    im3_binary.save(f'{arg.save_dir}/binary_output_2.png')
    im2.save(f'{arg.save_dir}/output_1.png')
    im3.save(f'{arg.save_dir}/output_2.png')
    if arg.k>=4:
        im4.save(f'{arg.save_dir}/output_3.png')
        im4_binary.save(f'{arg.save_dir}/binary_output_3.png')
    if arg.k>=5:
        im5.save(f'{arg.save_dir}/output_4.png')
        im5_binary.save(f'{arg.save_dir}/binary_output_4.png')
    if arg.k>=8:
        im6.save(f'{arg.save_dir}/output_5.png')
        im6_binary.save(f'{arg.save_dir}/binary_output_5.png')
        im7.save(f'{arg.save_dir}/output_6.png')
        im7_binary.save(f'{arg.save_dir}/binary_output_6.png')
        im8.save(f'{arg.save_dir}/output_7.png')
        im8_binary.save(f'{arg.save_dir}/binary_output_7.png')
    if arg.k>=10:
        im9.save(f'{arg.save_dir}/output_8.png')
        im9_binary.save(f'{arg.save_dir}/binary_output_8.png')
        im10.save(f'{arg.save_dir}/output_9.png')
        im10_binary.save(f'{arg.save_dir}/binary_output_9.png')
if __name__ == '__main__':
    test()
    