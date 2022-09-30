import argparse
import os
import lit_model_one_eye, lit_tiny_resnet_one_eye, lit_tiny_inception_one_eye, lit_tiny_inception_resnet_one_eye
from pytorch_lightning.loggers import CometLogger
import re
from gazetrack_data import gazetrack_dataset
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

parser = argparse.ArgumentParser(description='Test GazeTracker')
parser.add_argument('--dataset_dir', default='../dataset/', help='Path to converted dataset')
parser.add_argument('--model', default='google_cnn_one_eye', help='Model')
parser.add_argument('--checkpoints', default='../model/', help='Model checkpoints')

if __name__ == '__main__':
    args = parser.parse_args()

    models = {
    "google_cnn_one_eye": lit_model_one_eye.lit_gazetrack_model,
    "google_tiny_resnet_one_eye": lit_tiny_resnet_one_eye.lit_gazetrack_model,
    "google_tiny_inception_one_eye": lit_tiny_inception_one_eye.lit_gazetrack_model,
    "google_tiny_inception_resnet_one_eye": lit_tiny_inception_resnet_one_eye.lit_gazetrack_model
    }

    def get_val_loss(fname):
        return float(re.findall("val_loss=[-+]?(?:\d*\.\d+|\d+)", fname)[0].replace("val_loss=", ""))

    def get_best_checkpoint(chkptsdir=args.checkpoints+args.model+"/"):
        ckpts = os.listdir(chkptsdir)
        return min(ckpts, key=lambda x: get_val_loss(x))

    def euc(a, b):
        return np.sqrt(np.sum(np.square(a - b), axis=1))

    logger = CometLogger(
        api_key="zQb5zTn15DuSBvVEBjWvs2HzK",
        project_name="TEST_MODELS",
    )

    model = models[args.model](None, None, 256, logger)
    if(torch.cuda.is_available()):
        dev = torch.device('cuda:0')
    else:
        dev = torch.device('cpu')

    print("Best checkpoint: ", args.checkpoints+args.model+"/"+get_best_checkpoint())
    weights = torch.load(args.checkpoints+args.model+"/"+get_best_checkpoint(), map_location=dev)['state_dict']
    model.load_state_dict(weights)
    model.to(dev)
    model.eval()

    for ds in ['val', 'test']:
        file_root = args.dataset_dir+ds+"/images/"
        dataset = gazetrack_dataset(file_root, phase='test')
        dataloader = DataLoader(dataset, batch_size=256, num_workers=8, pin_memory=False, shuffle=False)
        
        preds_left, preds_right, preds_avg, gt = [], [], [], []
        for j in tqdm(dataloader):
            leye, reye, kps, target = j[1].to(dev), j[2].to(dev), j[3].to(dev), j[4].to(dev)
            
            with torch.no_grad():
                pred1 = model(leye, kps[:,:4])
                pred2 = model(reye, kps[:,4:])
            pred1 = pred1.cpu().detach().numpy()
            pred2 = pred2.cpu().detach().numpy()
            preds_left.extend(pred1)
            preds_right.extend(pred2)
            preds_avg.extend(0.1*pred1+0.9*pred2)
            
            gt.extend(target.cpu().detach().numpy())
        
        preds_left = np.array(preds_left)
        preds_right = np.array(preds_right)
        preds_avg = np.array(preds_avg)
        pts = np.unique(gt, axis=0)

        gt = np.array(gt)
        dist_left = euc(preds_left, gt)
        dist_right = euc(preds_right, gt)
        dist_avg = euc(preds_avg, gt)
        print(f"Mean Euclidean Distance ({ds} set):")
        print("\tRight eye: ", dist_right.mean())
        print("\tLeft eye: ", dist_left.mean())
        print("\tRight + Left eye: ", dist_avg.mean())