import argparse
import os
import lit_model, lit_tiny_resnet, lit_tiny_inception, lit_mobilenet, lit_tiny_inception_resnet
from pytorch_lightning.loggers import CometLogger
import re
from gazetrack_data import gazetrack_dataset
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

parser = argparse.ArgumentParser(description='Test GazeTracker')
parser.add_argument('--dataset_dir', default='../dataset/', help='Path to converted dataset')
parser.add_argument('--model', default='google_cnn', help='Model')
parser.add_argument('--checkpoints', default='../model/', help='Model checkpoints')

if __name__ == '__main__':
    args = parser.parse_args()

    models = {
    "google_cnn": lit_model.lit_gazetrack_model,
    "google_tiny_resnet_2": lit_tiny_resnet.lit_gazetrack_model,
    "google_tiny_inception": lit_tiny_inception.lit_gazetrack_model,
    "google_tiny_inception_resnet": lit_tiny_inception_resnet.lit_gazetrack_model,
    "google_mobilenet": lit_mobilenet.lit_gazetrack_model
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
    weights = torch.load(args.checkpoints+args.model+"/"+get_best_checkpoint(), map_location=dev)['state_dict']
    model.load_state_dict(weights)
    model.to(dev)
    model.eval()

    for ds in ['val', 'test']:
        file_root = args.dataset_dir+ds+"/images/"
        dataset = gazetrack_dataset(file_root, phase='test')
        dataloader = DataLoader(dataset, batch_size=256, num_workers=8, pin_memory=False, shuffle=False)

        preds, gt = [], []
        for j in tqdm(dataloader):
            leye, reye, kps, target = j[1].to(dev), j[2].to(dev), j[3].to(dev), j[4].to(dev)
            
            with torch.no_grad():
                pred = model(leye, reye, kps)
            pred = pred.cpu().detach().numpy()
            preds.extend(pred)  
            
            gt.extend(target.cpu().detach().numpy())
            
        preds = np.array(preds)
        pts = np.unique(gt, axis=0)

        gt = np.array(gt)
        dist = euc(preds, gt)
        print(f"Mean Euclidean Distance ({ds} set): ", dist.mean())