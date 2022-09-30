import argparse
import os
import lit_model, lit_tiny_resnet, lit_tiny_inception, lit_mobilenet
from pytorch_lightning.loggers import CometLogger
import re
from gazetrack_data import gazetrack_dataset
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

parser = argparse.ArgumentParser(description='Test GazeTracker')
parser.add_argument('--dataset_dir', default='../dataset/', help='Path to converted dataset')
parser.add_argument('--model_1', default='google_tiny_resnet_2', help='First model')
parser.add_argument('--model_2', default='google_tiny_inception', help="Second model")
parser.add_argument('--checkpoints', default='../model/', help='Model checkpoints')

if __name__ == '__main__':
    args = parser.parse_args()

    models = {
    "google_cnn": lit_model.lit_gazetrack_model,
    "google_tiny_resnet_2": lit_tiny_resnet.lit_gazetrack_model,
    "google_tiny_inception": lit_tiny_inception.lit_gazetrack_model,
    "google_mobilenet": lit_mobilenet.lit_gazetrack_model
    }

    def get_val_loss(fname):
        return float(re.findall("val_loss=[-+]?(?:\d*\.\d+|\d+)", fname)[0].replace("val_loss=", ""))

    def get_best_checkpoint(chkptsdir=args.checkpoints, model=args.model_1):
        ckpts = os.listdir(chkptsdir+model+"/")
        return min(ckpts, key=lambda x: get_val_loss(x))

    def euc(a, b):
        return np.sqrt(np.sum(np.square(a - b), axis=1))

    logger = CometLogger(
        api_key="zQb5zTn15DuSBvVEBjWvs2HzK",
        project_name="TEST_MODELS",
    )

    model1 = models[args.model_1](None, None, 256, logger)
    model2 = models[args.model_2](None, None, 256, logger)
    
    if(torch.cuda.is_available()):
        dev = torch.device('cuda:0')
    else:
        dev = torch.device('cpu')

    weights1 = torch.load(args.checkpoints+args.model_1+"/"+get_best_checkpoint(model=args.model_1), map_location=dev)['state_dict']
    model1.load_state_dict(weights1)
    model1.to(dev)
    model1.eval()

    weights2 = torch.load(args.checkpoints+args.model_2+"/"+get_best_checkpoint(model=args.model_2), map_location=dev)['state_dict']
    model2.load_state_dict(weights2)
    model2.to(dev)
    model2.eval()

    for ds in ['val', 'test']:
        file_root = args.dataset_dir+ds+"/images/"
        dataset = gazetrack_dataset(file_root, phase='test')
        dataloader = DataLoader(dataset, batch_size=256, num_workers=8, pin_memory=False, shuffle=False)

        preds, gt = [], []
        for j in tqdm(dataloader):
            leye, reye, kps, target = j[1].to(dev), j[2].to(dev), j[3].to(dev), j[4].to(dev)
            
            with torch.no_grad():
                pred1 = model1(leye, reye, kps)
                pred2 = model2(leye, reye, kps)
            pred = (pred1.cpu().detach().numpy()/2.0) + (pred2.cpu().detach().numpy()/2.0)
            preds.extend(pred)  
            
            gt.extend(target.cpu().detach().numpy())
            
        preds = np.array(preds)
        pts = np.unique(gt, axis=0)

        gt = np.array(gt)
        dist = euc(preds, gt)
        print(f"Mean Euclidean Distance ({ds} set): ", dist.mean())