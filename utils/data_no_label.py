import torch
from torchvision.datasets import MNIST


import os
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt

import cv2
from skimage import feature
from scipy import ndimage

import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt

def load_data_no_label(ratio=1):
    tr_ds = MNIST('mnist', True, None, download=True)
    n_sample = len(tr_ds)
    idx = torch.randperm(n_sample)[:int(n_sample * ratio)]
    tr_ds = ((tr_ds.data[:, None, ...] / 255.)[idx])
    test_ds = MNIST('mnist', False, None, download=True)
    test_ds = (test_ds.data[:, None, ...] / 255.)
    
    return tr_ds, test_ds

def load_data_bacteria(ratio=1):
    tr_ds = MNIST('mnist', True, None, download=True)
    n_sample = len(tr_ds)
    idx = torch.randperm(n_sample)[:int(n_sample * ratio)]
    tr_ds = ((tr_ds.data[:, None, ...] / 255.)[idx])
    test_ds = MNIST('mnist', False, None, download=True)
    test_ds = (test_ds.data[:, None, ...] / 255.)
    
    return tr_ds, test_ds


def loadDataFromFiles(data_directory, background):
    files = [f for f in os.listdir(data_directory) if f.startswith('cyte') and f.endswith('.tiff')]
    files = sorted(files)
    im0 = np.array(Image.open(os.path.join(data_directory, files[0])))
    h, w = im0.shape
    data = np.zeros((len(files), h, w), dtype=np.float64)

    for i, filename in enumerate(files):
        # print(data_directory + filename)
        im = np.array(Image.open(os.path.join(data_directory, filename))).astype(np.float64)
        clean_im = remove_background(im, background)
        data[i, :] = np.array(clean_im)
    data[data == 0] = np.nan
    return data


def detect_edges_canny(im, args):
    edges = feature.canny(im, low_threshold=args.lower_thresh, high_threshold=args.higher_thresh)
    # edge-detection based on the final image <- add args for low and high threshold for initial user calibration
    return edges

def background_mask(edges):
    dilate1 = ndimage.binary_dilation(edges)  # binary dilation
    dilated = ndimage.binary_fill_holes(dilate1)
    dilated = np.uint8(dilated)
    return dilated

def remove_background(im, background):
    foreground = cv2.bitwise_and(im, im, mask=background)
    foreground[foreground == 0] = np.nan
    return foreground

def remove_background_connor(im, edges):
    # fillholes = ndimage.binary_fill_holes(edges)  # binary fill holes
    dilate1 = ndimage.binary_dilation(edges)  # binary dilation
    dilated = ndimage.binary_fill_holes(dilate1)
    dilated = np.uint8(dilated)
    foreground = cv2.bitwise_and(im, im, mask=dilated)
    foreground[foreground == 0] = np.nan
    return foreground

def compute_gradient_values(arr):
    gradient_arr = np.empty(arr.shape, dtype=np.float64)
    channels = arr.shape[0]
    for i in range(channels):
        if i == 0:
            gradient_arr[i] = arr[i] / arr[i]
        else:
            gradient_arr[i] = arr[i] / arr[i-1]            
    return gradient_arr


def normalise(data_directory, base_files):
    im_base = np.array(Image.open(os.path.join(data_directory, base_files[0])))
    im_base = im_base.astype(np.float64)
    for idx, file in enumerate(base_files):
        if idx == 0:
            continue
        im = np.array(Image.open(os.path.join(data_directory, file)))
        im_base = im_base + im
    return im_base / len(base_files)


def get_background(args):
    im_first = np.array(Image.open(os.path.join(args.data_directory, args.source_file))).astype(np.float64)
    edges = detect_edges_canny(im_first, args)
    return edges


def load_arguments_from_file(file_path):
    """Reads arguments from a file, ignoring lines that start with #, and returns them as a list."""
    print("reading from args.txt")
    args = []
    with open(file_path, 'r') as file:
        for line in file:
            # Strip whitespace from the beginning and end of the line
            stripped_line = line.strip()
            # Ignore lines that start with # or are empty
            if not stripped_line.startswith('#') and stripped_line:
                args.extend(stripped_line.split())
    print(args)
    return args


def get_data(args):
    edges = get_background(args)

    background = background_mask(edges)
    if args.test=="yes":
        background = np.ones(background.shape, dtype=np.uint8)

    data_stack = loadDataFromFiles(args.data_directory, background)
    im_base = normalise(args.data_directory, args.base_files)
    clean_im_base = remove_background(im_base, background)

    if args.normalise == "yes":
        data_stack = data_stack/clean_im_base
        if args.log == "yes":
            data_stack = np.log10(data_stack)
    
    ch, width, height = data_stack.shape
    width_indices = np.repeat(np.arange(width)[:, np.newaxis], height, axis=1)
    height_indices = np.repeat(np.arange(height)[np.newaxis, :], width, axis=0)
    data_stack = np.vstack((data_stack, width_indices[np.newaxis, :, :], height_indices[np.newaxis, :, :]))
    data_stack = data_stack.reshape((data_stack.shape[0], data_stack.shape[1]*data_stack.shape[2]))

    if args.rm_bag:
        has_nan = np.any(np.isnan(data_stack), axis=0)
        filtered_data = data_stack[:, ~has_nan]
        filtered_data = filtered_data.T
        n_sample = len(filtered_data)
        idx = torch.randperm(n_sample)[:int(n_sample)]
        filtered_data = (filtered_data[idx])
        return torch.tensor(filtered_data)
    else:
        return torch.tensor(data_stack.T)
def get_data_multi(args, idx, antibiotic):
    edges = get_background(args)

    background = background_mask(edges)

    data_stack = loadDataFromFiles(args.data_directory, background)
    im_base = normalise(args.data_directory, args.base_files)
    clean_im_base = remove_background(im_base, background)

    if args.normalise == "yes":
        data_stack = data_stack/clean_im_base
        if args.log == "yes":
            data_stack = np.log10(data_stack)
    elif args.norm_data:
        data_stack /= 2**16
    
    ch, width, height = data_stack.shape
    width_indices = np.repeat(np.arange(width)[:, np.newaxis], height, axis=1)
    height_indices = np.repeat(np.arange(height)[np.newaxis, :], width, axis=0)
    idx_indices = np.ones(height_indices.shape)*idx
    antibiotic_indices = np.ones(height_indices.shape)*antibiotic
    data_stack = np.vstack((data_stack, width_indices[np.newaxis, :, :], height_indices[np.newaxis, :, :], idx_indices[np.newaxis, :, :], antibiotic_indices[np.newaxis, :, :]))
    data_stack = data_stack.reshape((data_stack.shape[0], data_stack.shape[1]*data_stack.shape[2]))
    if args.rm_bag:
        has_nan = np.any(np.isnan(data_stack), axis=0)
        filtered_data = data_stack[:, ~has_nan]
        filtered_data = filtered_data.T
        return filtered_data
        # n_sample = len(filtered_data)
        # idx = torch.randperm(n_sample)[:int(n_sample)]
        # filtered_data = (filtered_data[idx])
        # return torch.tensor(filtered_data)
    else:
        return data_stack.T
        # return torch.tensor(data_stack.T)

from torch.utils.data import Dataset, DataLoader
class Load_Data(Dataset):
    def __init__(self, data_stack):
        super().__init__()
        self.data_stack = data_stack

    def __len__(self):
        return len(self.data_stack)
    
    def __getitem__(self, index):
        X = torch.tensor(self.data_stack[index, :45])
        p_loc = torch.tensor(self.data_stack[index, 45:])
        return X, p_loc
