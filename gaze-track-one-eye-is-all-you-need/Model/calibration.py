import json
import os, time
import shutil
import numpy as np 
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers
import lit_model, lit_tiny_resnet, lit_tiny_inception, lit_tiny_inception_resnet
from comet_ml import Experiment
from pytorch_lightning.loggers import CometLogger
from gazetrack_data import gazetrack_dataset
from glob import glob

import argparse

import torch
import os
import cv2
import random
import numpy as np

from PIL import Image
from glob import glob
from tqdm import tqdm, trange
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import json
import pytorch_lightning as pl
import sys

from lit_model import lit_gazetrack_model

from model import gazetrack_model
from gazetrack_data import gazetrack_dataset

from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
import re

parser = argparse.ArgumentParser(description='Test GazeTracker')
parser.add_argument('--dataset_dir', default='../dataset/', help='Path to converted dataset')
parser.add_argument('--save_dir', default='../metrics/', help='Model')
parser.add_argument('--checkpoints', default='../model/', help='Model checkpoints')

if __name__ == '__main__':
    args = parser.parse_args()

    models_dict = {
    "google_cnn": lit_model.lit_gazetrack_model,
    "google_tiny_resnet_2": lit_tiny_resnet.lit_gazetrack_model,
    "google_tiny_inception": lit_tiny_inception.lit_gazetrack_model,
    "google_tiny_inception_resnet": lit_tiny_inception_resnet.lit_gazetrack_model
    }

    def get_val_loss(fname):
        return float(re.findall("val_loss=[-+]?(?:\d*\.\d+|\d+)", fname)[0].replace("val_loss=", ""))

    def get_best_checkpoint(modelname, chkptsdir=args.checkpoints):
        ckpts = os.listdir(chkptsdir+modelname+"/")
        return min(ckpts, key=lambda x: get_val_loss(x))

    def euc(a, b):
        return np.sqrt(np.sum(np.square(a - b), axis=1))

    import colorsys
    def get_colors(num_colors):
        colors=[]
        for i in np.arange(0., 360., 360. / num_colors):
            hue = i/360.
            lightness = (50 + np.random.rand() * 10)/100.
            saturation = (90 + np.random.rand() * 10)/100.
            colors.append(colorsys.hls_to_rgb(hue, lightness, saturation))
        return colors

    logger = CometLogger(
        api_key="zQb5zTn15DuSBvVEBjWvs2HzK",
        project_name="TEST_MODELS",
    )

    models = {modelname: models_dict[modelname](None, None, 256, logger) for modelname in models_dict.keys()}
    for key in models.keys():
        w = torch.load(args.checkpoints+key+"/"+get_best_checkpoint(key))['state_dict']
        models[key].cuda()
        models[key].load_state_dict(w)
        models[key].eval()

    f = args.dataset_dir+"/test/images/"

    all_files = glob(f+"*.jpg")
    all_files = [i[:-10] for i in all_files]
    files = np.unique(all_files)
    print('Found ', len(all_files), ' images from ', len(files), ' subjects.')

    fnames = []
    nums = []
    for i in tqdm(files):
        fnames.append(i)
        nums.append(len(glob(i+"*.jpg")))
    fnames = np.array(fnames)
    nums = np.array(nums)
    ids = np.argsort(nums)
    ids = ids[::-1]
    fnames_sorted = fnames[ids]
    nums_sorted = nums[ids]
    files = fnames_sorted.copy()
    nums_sorted[0], nums_sorted[-1], sum(nums_sorted)

    activations = {key:{} for key in models.keys()}
    def get_activation(name, key):
        def hook(model, input, output):
            activations[key][name] = output.detach()
        return hook
    for key in models.keys(): models[key].combined_model[6].register_forward_hook(get_activation('out', key))

    def train_svr(file):
        file = file.replace("test", "train")
        dataset = gazetrack_dataset(file, phase="test", v=False)
        loader = DataLoader(
            dataset,
            batch_size=256,
            num_workers=10,
            pin_memory=False,
            shuffle=False,
        )

        preds, gt, dot_nums = {key:[] for key in models.keys()}, [], []
        calib_preds, calib_gt = [], []
        for j in loader:
            leye, reye, kps, target = j[1].cuda(), j[2].cuda(), j[3].cuda(), j[4].cuda()
            for key in models.keys():
                with torch.no_grad():
                    pred = list(models[key](leye, reye, kps).cpu().detach().numpy())
                pred = list(activations[key]["out"].detach().cpu().numpy())
                preds[key] = preds[key] + pred
            gt.extend(target.cpu().detach().numpy())

        gt = np.array(gt)
        preds = {key:np.array(preds[key]) for key in models.keys()}
        reg = MultiOutputRegressor(SVR(kernel="rbf", C=20, gamma=0.06))
        
        reg.fit(np.hstack([preds[key] for key in models.keys()]), gt)
        
        return reg

    def comp_pred_test_svr(fname, ct=False):
        reg = train_svr(fname)
        
        f = fname.replace("train", "test")
        if ct:
            f = fname
            
        test_dataset = gazetrack_dataset(f, phase="test", v=False)
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=256,
            num_workers=10,
            pin_memory=False,
            shuffle=False,
        )

        preds_pre, preds_final, gt, dot_nums = {key:[] for key in models.keys()}, [], [], []
        cent_fin = []

        calib_preds, calib_gt = [], []

        for j in test_dataloader:
            leye, reye, kps, target = j[1].cuda(), j[2].cuda(), j[3].cuda(), j[4].cuda()
            acts = {key:[] for key in models.keys()}
            for key in models.keys():
                with torch.no_grad():
                    pred = models[key](leye, reye, kps)
                pred = pred.cpu().detach().numpy()
                act = list(activations[key]["out"].cpu().detach().numpy())
                acts[key].extend(act)
                preds_pre[key].extend(pred)

            pred_fin = reg.predict(np.hstack([np.array(acts[key]) for key in models.keys()]))
            
            preds_final.extend(pred_fin)
            gt.extend(target.cpu().detach().numpy())

        preds_pre = {key:np.array(preds_pre[key]) for key in models.keys()}
        preds_final = np.array(preds_final)
        
        pts = np.unique(gt, axis=0)

        c = get_colors(len(pts))
        random.shuffle(c)

        gt = np.array(gt)
        dist_pre = {key:euc(preds_pre[key], gt) for key in models.keys()}
        dist_final = euc(preds_final, gt)
        
        out = [dist_pre, dist_final, gt, preds_pre, preds_final, pts, c]

        return out

    svr_out = {}
    for i in tqdm(files[:]):
        svr_out[i] = comp_pred_test_svr(i)

    means_pre = {key:[] for key in models.keys()}
    means_post = []
    for idx, i in enumerate(svr_out):
        for key in models.keys():
            means_pre[key].extend(svr_out[i][0][key])
        means_post.extend(svr_out[i][1])
    print("Mean without SVR: ", {key:np.mean(means_pre[key]) for key in models.keys()}, " Mean after SVR: ", np.mean(means_post))

    mean_errs_pre = []
    mean_errs_final = []
    for i in svr_out:
        mean_errs_pre.append({key:np.mean(svr_out[i][0][key]) for key in models.keys()})
        mean_errs_final.append(np.mean(svr_out[i][1]))

    np.save(args.save_dir+"mean_errs_pre.npy", mean_errs_pre)
    np.save(args.save_dir+"mean_errs_final.npy", mean_errs_final)

    np.save(args.save_dir+"means_pre.npy", means_pre)
    np.save(args.save_dir+"means_post.npy", means_post)

    # plt.figure(figsize=(15,10))
    # plt.title('SVR Mean Comparison (13 Point Calibration)')
    # ctr = 0
    # plt.hlines(y=np.mean(mean_errs_pre), xmin=0, xmax=len(mean_errs_pre), color='b', linestyles='dashed', label="Overall Base Model Mean Error: "+str(np.round(np.mean(mean_errs_pre), 3))+" cm")
    # plt.hlines(y=np.median(mean_errs_final), xmin=0, xmax=len(mean_errs_pre), color='k', linestyles='dashed', label="Overall Post SVR Mean Error: "+str(np.round(np.mean(mean_errs_final), 3))+" cm")
    # for i in range(len(means_pre)):
    #     if(means_post[i]<= means_pre[i]):
    #         plt.vlines(x=ctr, ymin=means_post[i], ymax=means_pre[i], colors='green')
    #     else:
    #         plt.vlines(x=ctr, ymin=means_pre[i], ymax=means_post[i], colors='red')
    #     ctr+=1
    # plt.scatter([i for i in range(len(mean_errs_pre))], mean_errs_pre, s=15, label="Base Model Mean Error", color='b')
    # plt.scatter([i for i in range(len(mean_errs_pre))], mean_errs_final, s=15, label="Post SVR Mean Error", color='black')



    # plt.xlabel('Subject id')
    # plt.ylabel('Mean Euclidean Distance (cm)')
    # plt.ylim(0)
    # plt.legend()