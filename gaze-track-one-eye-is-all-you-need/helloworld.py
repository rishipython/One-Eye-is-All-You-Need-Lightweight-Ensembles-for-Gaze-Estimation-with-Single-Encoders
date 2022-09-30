# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import pytorch_lightning as pl

# from Model.gazetrack_data import gazetrack_dataset
# from torch.utils.data import DataLoader
# from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau

# train_dataset = gazetrack_dataset("/data/dataset/train/", phase='train')
# f, leye, reye, kps, out, screen_w, screen_h = train_dataset[0]
# print(leye.shape)
# print(reye.shape)

import dlib

p = './shape_predictor_68_face_landmarks.dat'
detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor(p)