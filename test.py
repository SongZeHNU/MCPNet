# coding: utf-8
import os
import argparse
import datetime
import time
import pandas as pd
import importlib
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.backends.cudnn as cudnn
from sklearn.metrics import confusion_matrix
from utils.utils import init_experiment, seed_torch, str2bool
from utils.schedulers import get_scheduler
from data.open_set_datasets import get_class_splits, get_datasets
from config import exp_root
import logging
from models.modeling_my_proto import VisionTransformer, CONFIGS
from tqdm import tqdm
import numpy as np
from methods.ARPL.core import evaluation
import sklearn
import sklearn.metrics
from sklearn.metrics import average_precision_score
import os.path as osp
import timm
import timm.optim
import timm.scheduler
import math
import random
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


logger = logging.getLogger(__name__)
parser = argparse.ArgumentParser("Training")

# Dataset
parser.add_argument('--dataset', type=str, default='cod_test', help="")  #fgco_test
parser.add_argument('--out-num', type=int, default=10, help='For cifar-10-100')
parser.add_argument('--image_size', type=int, default=384)

# optimization
parser.add_argument('--optim', type=str, default=None, help="Which optimizer to use {adam, sgd}")
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--lr', type=float, default=2e-5, help="learning rate for model")
parser.add_argument('--weight_decay', type=float, default=1e-4, help="LR regularisation on weights")
parser.add_argument('--gan_lr', type=float, default=0.0002, help="learning rate for gan")
parser.add_argument('--max-epoch', type=int, default=200) #600
parser.add_argument('--scheduler', type=str, default='cosine_warm_restarts_warmup')

# model
parser.add_argument('--weight-pl', type=float, default=0.1, help="weight for center loss")
parser.add_argument('--label_smoothing', type=float, default=0.3, help="Smoothing constant for label smoothing."
                                                                        "No smoothing if None or 0")
parser.add_argument('--beta', type=float, default=0.1, help="weight for entropy loss")
parser.add_argument('--model', type=str, default='timm_resnet50_pretrained')
parser.add_argument('--resnet50_pretrain', type=str, default='places_moco',
                        help='Which pretraining to use if --model=timm_resnet50_pretrained.'
                             'Options are: {iamgenet_moco, places_moco, places}', metavar='BOOL')
parser.add_argument('--feat_dim', type=int, default=128, help="Feature vector dim, only for classifier32 at the moment")

# aug
parser.add_argument('--transform', type=str, default='rand-augment')
parser.add_argument('--rand_aug_m', type=int, default=30)
parser.add_argument('--rand_aug_n', type=int, default=2)

# misc
parser.add_argument('--num_workers', default=2, type=int)
parser.add_argument('--split_train_val', default=False, type=str2bool,
                        help='Subsample training set to create validation set', metavar='BOOL')
parser.add_argument('--device', default='cuda:1', type=str, help='Which GPU to use')
parser.add_argument('--nz', type=int, default=100)
parser.add_argument('--ns', type=int, default=1)
parser.add_argument('--eval-freq', type=int, default=1)
parser.add_argument('--print-freq', type=int, default=100)
parser.add_argument('--checkpt_freq', type=int, default=20)
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--use-cpu', action='store_true')
parser.add_argument('--eval', action='store_true', help="Eval", default=False)
parser.add_argument('--cs', action='store_true', help="Confusing Sample", default=False)
parser.add_argument('--train_feat_extractor', default=True, type=str2bool,
                        help='Train feature extractor (only implemented for renset_50_faces)', metavar='BOOL')
parser.add_argument('--split_idx', default=0, type=int, help='0-4 OSR splits for each dataset')
parser.add_argument('--use_softmax_in_eval', default=False, type=str2bool,
                        help='Do we use softmax or logits for evaluation', metavar='BOOL')


# ###################################### self-defined model ######################################


# ################################################################################################
# ############################################ main ##############################################
# ################################################################################################
args = parser.parse_args()
args.exp_root = exp_root
args.epochs = args.max_epoch
img_size = args.image_size
results = dict()
global best_auroc
best_auroc = 0
global best_auroc_epoch
global best_auroc_acc1
global best_auroc_acc2
global best_auroc_auroc1
global best_auroc_auroc2
best_auroc_epoch = 0
best_auroc_acc1 = 0
best_auroc_acc2 = 0
best_auroc_auroc1 = 0
best_auroc_auroc2 = 0




for i in range(1):

    # ------------------------
    # INIT
    # ------------------------
    if args.feat_dim is None:
        args.feat_dim = 128 if args.model == 'classifier32' else 2048

    args.train_classes, args.open_set_classes = get_class_splits(args.dataset, args.split_idx,
                                                                 cifar_plus_n=args.out_num)

    img_size = args.image_size

    args.save_name = '{}_{}_{}'.format(args.model, args.seed, args.dataset)
    runner_name = os.path.dirname(__file__).split("/")[-2:]

    # ------------------------
    # SEED
    # ------------------------
    seed_torch(args.seed)

    # ------------------------
    # DATASETS
    # ------------------------
    datasets = get_datasets(args.dataset, transform=args.transform, train_classes=args.train_classes,
                            open_set_classes=args.open_set_classes, balance_open_set_eval=False,    #balance
                            split_train_val=args.split_train_val, image_size=args.image_size, seed=args.seed,
                            args=args)

    # ------------------------
    # RANDAUG HYPERPARAM SWEEP
    # ------------------------


    # ------------------------
    # DATALOADER
    # ------------------------
    dataloaders = {}
    for k, v, in datasets.items():
        shuffle = True if k == 'train' else False
        dataloaders[k] = DataLoader(v, batch_size=args.batch_size, shuffle=shuffle, sampler=None, num_workers=args.num_workers)

    # ------------------------
    # SAVE PARAMS
    # ------------------------
    options = vars(args)
    options.update(
        {
            'item': i,
            'known': args.train_classes,
            'unknown': args.open_set_classes,
            'img_size': img_size,
            'dataloaders': dataloaders,
            'num_classes': len(args.train_classes)
        }
    )



    if options['dataset'] == 'cifar-10-100':
        file_name = '{}_{}.csv'.format(options['dataset'], options['out_num'])
        if options['cs']:
            file_name = '{}_{}_cs.csv'.format(options['dataset'], options['out_num'])
    else:
        file_name = options['dataset'] + '.csv'
        if options['cs']:
            file_name = options['dataset'] + 'cs' + '.csv'



torch.manual_seed(options['seed'])
os.environ['CUDA_VISIBLE_DEVICES'] = options['gpu']
use_gpu = torch.cuda.is_available()
if options['use_cpu']: use_gpu = False

if use_gpu:
    print("Currently using GPU: {}".format(options['gpu']))
    cudnn.benchmark = False
    torch.cuda.manual_seed_all(options['seed'])
else:
    print("Currently using CPU")

# -----------------------------
# DATALOADERS
# -----------------------------
# trainloader = dataloaders['train']
# testloader = dataloaders['val']
testloader = dataloaders['test_known']
outloader = dataloaders['test_unknown']

# -----------------------------
# MODEL
# -----------------------------
print("Creating model: {}".format(options['model']))


config = CONFIGS['ViT-B_16']
#config.split = 'non-overlap'
config.split = 'overlap'



net = VisionTransformer(config, args.img_size, zero_head=True, num_classes=len(args.train_classes), smoothing_value=0.0)
net.load_state_dict(torch.load('./open_set_recognition/log/your_model_path/checkpoints/xxx.pth'))




feat_dim = args.feat_dim

options.update(
    {
        'feat_dim': feat_dim,
        'use_gpu': use_gpu
    }
)

# -----------------------------
# PREPARE EXPERIMENT
# -----------------------------
if use_gpu:
    # net_pre = net_pre.cuda()
    net = net.cuda()


start_time = time.time()

import numpy as np

if options['max_epoch'] == 200:
        net.eval()
        correct, total = 0, 0
        correct2, total2 = 0, 0

        torch.cuda.empty_cache()

        _pred_k_acc, _pred_k, _pred_u, _labels = [], [], [], []
        _pred_k_acc2, _pred_k2, _pred_u2 = [], [], []



        for data, labels, idx in tqdm(testloader):
            if options['use_gpu']:
                data, labels = data.cuda(), labels.cuda()

            with torch.no_grad():

    
                outputs_ori, outputs_part, _, whole_w, part_w, whole_f, part_f = net(data)



                predictions = outputs_ori.data.max(1)[1]
                total += labels.size(0)
                correct += (predictions == labels.data).sum()

                predictions2 = outputs_part.data.max(1)[1]
                total2 += labels.size(0)
                correct2 += (predictions2 == labels.data).sum()

                _pred_k.append(outputs_ori.data.cpu().numpy())
                _pred_k2.append(outputs_part.data.cpu().numpy())
                _labels.append(labels.data.cpu().numpy())



        for batch_idx, (data, labels, idx) in enumerate(tqdm(outloader)):
            if options['use_gpu']:
                data, labels = data.cuda(), labels.cuda()

            with torch.no_grad():
           
                outputs_ori, outputs_part, _, whole_w, part_w, whole_f, part_f = net(data)

            
                _pred_u.append(outputs_ori.data.cpu().numpy())
                _pred_u2.append(outputs_part.data.cpu().numpy())

        # Accuracy
        acc1 = float(correct) * 100. / float(total)
        print('Acc_net1: {:.5f}'.format(acc1))
        acc2 = float(correct2) * 100. / float(total2)
        print('Acc_net2: {:.5f}'.format(acc2))

        _pred_k = np.concatenate(_pred_k, 0)
        _pred_u = np.concatenate(_pred_u, 0)
        _pred_k2 = np.concatenate(_pred_k2, 0)
        _pred_u2 = np.concatenate(_pred_u2, 0)
        _labels = np.concatenate(_labels, 0)

        # Out-of-Distribution detction evaluation
        x1, x2 = np.max(_pred_k, axis=1), np.max(_pred_u, axis=1)
        # x1, x2 = np.mean(_pred_k, axis=1), np.mean(_pred_u, axis=1)
        results = evaluation.metric_ood(x1, x2)['Bas']

        # OSCR
        _oscr_socre = evaluation.compute_oscr(_pred_k, _pred_u, _labels)

        results['ACC'] = acc1
        results['OSCR'] = _oscr_socre * 100.
        # results['AUPR'] = ap_score * 100
        auroc1 = results['AUROC']

        print("net1 Acc (%): {:.3f}\t AUROC (%): {:.3f}\t OSCR (%): {:.3f}\t".format(results['ACC'], results['AUROC'], results['OSCR']))

        # Out-of-Distribution detction evaluation
        x1, x2 = np.max(_pred_k2, axis=1), np.max(_pred_u2, axis=1)
        # # x1, x2 = np.mean(_pred_k, axis=1), np.mean(_pred_u, axis=1)
        results = evaluation.metric_ood(x1, x2)['Bas']

        # # OSCR
        _oscr_socre = evaluation.compute_oscr(_pred_k2, _pred_u2, _labels)

        results['ACC'] = acc2
        results['OSCR'] = _oscr_socre * 100.
        # results['AUPR'] = ap_score * 100
        auroc2 = results['AUROC']

        print("net2 Acc (%): {:.3f}\t AUROC (%): {:.3f}\t OSCR (%): {:.3f}\t".format(results['ACC'], results['AUROC'], results['OSCR']))


elapsed = round(time.time() - start_time)
elapsed = str(datetime.timedelta(seconds=elapsed))
print("Finished. Total elapsed time (h:m:s): {}".format(elapsed))

