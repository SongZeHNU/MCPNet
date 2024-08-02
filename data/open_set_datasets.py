from data.cub import get_cub_datasets
from data.cor import get_ncoc_datasets, get_ncoctest_datasets, get_acoc_datasets, get_acoctest_datasets
from data.tinyimagenet import get_tiny_image_net_datasets
from data.cifar import get_cifar_10_100_datasets, get_cifar_10_10_datasets
from data.open_set_splits.osr_splits import osr_splits
from data.augmentations import get_transform
from config import osr_split_dir

import os
import sys
import pickle
import torch

"""
For each dataset, define function which returns:
    training set
    validation set
    open_set_known_images
    open_set_unknown_images
"""

get_dataset_funcs = {
    'cub': get_cub_datasets,
    'ncoc': get_ncoc_datasets,
    'ncoc_test': get_ncoctest_datasets,
    'acoc': get_acoc_datasets,
    'acoc_test': get_acoctest_datasets,
    'tinyimagenet': get_tiny_image_net_datasets,
    'cifar-10-100': get_cifar_10_100_datasets,
    'cifar-10-10': get_cifar_10_10_datasets
}

def get_datasets(name, transform='default', image_size=224, train_classes=(0, 1, 8, 9),
                 open_set_classes=range(10), balance_open_set_eval=False, split_train_val=True, seed=0, args=None):

    """
    :param name: Dataset name
    :param transform: Either tuple of train/test transforms or string of transform type
    :return:
    """

    print('Loading datasets...')

    if isinstance(transform, tuple):
        train_transform, test_transform = transform
    else:
        train_transform, test_transform = get_transform(transform_type=transform, image_size=image_size, args=args)

    if name in get_dataset_funcs.keys():
        datasets = get_dataset_funcs[name](train_transform, test_transform,
                                  train_classes=train_classes,
                                  open_set_classes=open_set_classes,
                                  balance_open_set_eval=balance_open_set_eval,
                                  split_train_val=split_train_val,
                                  seed=seed)
    else:
        raise NotImplementedError

    return datasets

def get_class_splits(dataset, split_idx=0, cifar_plus_n=10):
   
    if dataset in ('cifar-10-10', 'mnist', 'svhn'):
        train_classes = osr_splits[dataset][split_idx]
        open_set_classes = [x for x in range(10) if x not in train_classes]

    elif dataset == 'cifar-10-100':
        train_classes = osr_splits[dataset][split_idx]
        open_set_classes = osr_splits['cifar-10-100-{}'.format(cifar_plus_n)][split_idx]

    elif dataset == 'tinyimagenet':
        train_classes = osr_splits[dataset][split_idx]
        open_set_classes = [x for x in range(200) if x not in train_classes]

    elif dataset == 'cub':

        osr_path = os.path.join(osr_split_dir, 'cub_osr_splits.pkl')
        with open(osr_path, 'rb') as f:
            class_info = pickle.load(f)

        train_classes = class_info['known_classes']

        open_set_classes = class_info['unknown_classes']
        open_set_classes = open_set_classes['Hard'] + open_set_classes['Medium'] + open_set_classes['Easy']

    elif dataset == 'ncoc':
       
        train_classes = list(range(69))
        ##3 unknown
        del train_classes[7]   #fish
        del train_classes[24]  #bird
        del train_classes[37]  #other


        #30 unknown
        # index =   [2, 3,6,8,10,13,15,16,18,20, 22,24,25,27,28,31,35,38,45,43,42,49,48,53,56,55,58,64,66,68] 
        # new_train_class = [x for i, x in enumerate(train_classes) if i not in index]
        # train_classes = new_train_class

        #50 unknown
        #index =  [2,3,5,6,7,8,10,11,13,14,15,16,18,19,20,21, 22,24,25,27,28,30,31,33,35,36,38,39,41,45,46,43,42,49,47,48,50,51,52,53,57,56,55,58,61,62,63,64,66,68] 
        # new_train_class = [x for i, x in enumerate(train_classes) if i not in index]
        # train_classes = new_train_class

        open_set_classes = [x for x in range(69) if x not in train_classes]
    
    elif dataset == 'ncoc_test':
       
        train_classes = list(range(69))
        ##del open set
        del train_classes[7]   #fish
        del train_classes[24]  #bird
        del train_classes[37]  #other
        open_set_classes = [x for x in range(69) if x not in train_classes]
    elif dataset == 'acoc':
       
        train_classes = list(range(20))   #class num

        
        #3 unknown
        del train_classes[3]  
        del train_classes[11]  
        del train_classes[16]  

        #8 unknown
        # del train_classes[1] 
        # del train_classes[2]
        # del train_classes[5]
        # del train_classes[6]
        # del train_classes[8]
        # del train_classes[8]
        # del train_classes[10]
        # del train_classes[11]


        #15 unknown
        # index =   [0, 1, 2,3,4,7,8,9,10, 13,12,15,16,18,19] 
        # new_train_class = [x for i, x in enumerate(train_classes) if i not in index]
        # train_classes = new_train_class

        open_set_classes = [x for x in range(20) if x not in train_classes]
    elif dataset == 'acoc_test':
       
        train_classes = list(range(20))
        ##del open set
        del train_classes[3]   
        del train_classes[11]  
        del train_classes[16]  
        open_set_classes = [x for x in range(20) if x not in train_classes]


    else:

        raise NotImplementedError

    return train_classes, open_set_classes

# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__