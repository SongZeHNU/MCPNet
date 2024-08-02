import os
import pandas as pd
import numpy as np
from copy import deepcopy
from PIL import Image
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url
from torch.utils.data import Dataset
from config import NCOC_root, ACOC_root
import torchvision.transforms as transforms
class COD():
    def __init__(self, root, is_train=True, data_len=None, transform=None):
        self.root = root
        self.is_train = is_train
        self.transform = transform
        img_txt_file = open(os.path.join(self.root, 'images.txt'))
        label_txt_file = open(os.path.join(self.root, 'image_class_labels.txt'))
        train_val_file = open(os.path.join(self.root, 'train_test_split.txt'))
        img_name_list = []
        for line in img_txt_file:
            img_name_list.append(line[:-1].split(' ')[-1])
        label_list = []
        for line in label_txt_file:
            label_list.append(int(line[:-1].split(' ')[-1]) - 1)
        train_test_list = []
        for line in train_val_file:
            train_test_list.append(int(line[:-1].split(' ')[-1]))
        train_file_list = [x for i, x in zip(train_test_list, img_name_list) if i]
        test_file_list = [x for i, x in zip(train_test_list, img_name_list) if not i]
        if self.is_train:
            # self.train_img = [cv2.imread(os.path.join(self.root, 'images', train_file)) for train_file in  #scipy.misc
            #                   train_file_list[:data_len]]
            self.train_img = [(os.path.join(self.root, 'images', train_file)) for train_file in  #scipy.misc
                              train_file_list[:data_len]]
            self.train_label = [x for i, x in zip(train_test_list, label_list) if i][:data_len]
            self.train_imgname = [x for x in train_file_list[:data_len]]
        if not self.is_train:
            # self.test_img = [cv2.imread(os.path.join(self.root, 'images', test_file)) for test_file in
            #                  test_file_list[:data_len]]
            self.test_img = [(os.path.join(self.root, 'images', test_file)) for test_file in
                              test_file_list[:data_len]]
            
            self.test_label = [x for i, x in zip(train_test_list, label_list) if not i][:data_len]
            self.test_imgname = [x for x in test_file_list[:data_len]]
    def __getitem__(self, index):
        if self.is_train:
            img, target, imgname = self.train_img[index], self.train_label[index], self.train_imgname[index]
            # if len(img.shape) == 2:
            #     img = np.stack([img] * 3, 2)

            img = Image.open(img)
            img.convert('RGB')
            #img = Image.fromarray(img, mode='RGB')

            if self.transform is not None:
                img = self.transform(img)
        else:
            img, target, imgname = self.test_img[index], self.test_label[index], self.test_imgname[index]
            # if len(img.shape) == 2:
            #     img = np.stack([img] * 3, 2)
            #img = Image.fromarray(img, mode='RGB')
            img = Image.open(img)
            img.convert('RGB')
            if self.transform is not None:
                img = self.transform(img)

        return img, target

    def __len__(self):
        if self.is_train:
            return len(self.train_label)
        else:
            return len(self.test_label)

class CustomCod(Dataset):
    base_folder = 'images/'
    url = 'http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz'
    filename = 'CUB_200_2011.tgz'
    tgz_md5 = '97eceeb196236b17998738112f37df78'

    def __init__(self, root, train=True, transform=None, target_transform=None, loader=default_loader, download=False):

        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform

        self.loader = loader
        self.train = train


        if download:
            self._download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        self.uq_idxs = np.array(range(len(self)))

    def _load_metadata(self):
        images = pd.read_csv(os.path.join(self.root, 'images.txt'), sep=' ',
                             names=['img_id', 'filepath'])
        image_class_labels = pd.read_csv(os.path.join(self.root, 'image_class_labels.txt'),
                                         sep=' ', names=['img_id', 'target'])
        train_test_split = pd.read_csv(os.path.join(self.root, 'train_test_split.txt'),
                                       sep=' ', names=['img_id', 'is_training_img'])

        data = images.merge(image_class_labels, on='img_id')
        self.data = data.merge(train_test_split, on='img_id')

        if self.train:
            self.data = self.data[self.data.is_training_img == 1]
        else:
            self.data = self.data[self.data.is_training_img == 0]

    def _check_integrity(self):
        try:
            self._load_metadata()
        except Exception:
            return False

        for index, row in self.data.iterrows():
            filepath = os.path.join(self.root, self.base_folder, row.filepath)
            if not os.path.isfile(filepath):
                print(filepath)
                return False
        return True

    def _download(self):
        import tarfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        download_url(self.url, self.root, self.filename, self.tgz_md5)

        with tarfile.open(os.path.join(self.root, self.filename), "r:gz") as tar:
            tar.extractall(path=self.root)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        path = os.path.join(self.root, self.base_folder, sample.filepath)
        #print(path)
        target = sample.target - 1  # Targets start at 1 by default, so shift to 0
        img = self.loader(path)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, self.uq_idxs[idx]



def subsample_dataset(dataset, idxs):

    mask = np.zeros(len(dataset)).astype('bool')
    mask[idxs] = True

    dataset.data = dataset.data[mask]
    dataset.uq_idxs = dataset.uq_idxs[idxs]

    return dataset


def subsample_classes(dataset, include_classes=range(160)):

    include_classes_cub = np.array(include_classes) + 1     # CUB classes are indexed 1 --> 200 instead of 0 --> 199

    for x, (_, r) in enumerate(dataset.data.iterrows()):
        if int(r['target']) in include_classes_cub:
            cls_idxs= x


    cls_idxs = [x for x, (_, r) in enumerate(dataset.data.iterrows()) if int(r['target']) in include_classes_cub]

    target_xform_dict = {}
    for i, k in enumerate(include_classes):
        target_xform_dict[k] = i

    dataset = subsample_dataset(dataset, cls_idxs)

    dataset.target_transform = lambda x: target_xform_dict[x]

    return dataset


def get_train_val_split(train_dataset, val_split=0.2):

    val_dataset = deepcopy(train_dataset)
    train_dataset = deepcopy(train_dataset)

    train_classes = np.unique(train_dataset.data['target'])

    # Get train/test indices
    train_idxs = []
    val_idxs = []
    for cls in train_classes:

        cls_idxs = np.where(train_dataset.data['target'] == cls)[0]

        v_ = np.random.choice(cls_idxs, replace=False, size=((int(val_split * len(cls_idxs))),))
        t_ = [x for x in cls_idxs if x not in v_]

        train_idxs.extend(t_)
        val_idxs.extend(v_)

    # Get training/validation datasets based on selected idxs
    train_dataset = subsample_dataset(train_dataset, train_idxs)
    val_dataset = subsample_dataset(val_dataset, val_idxs)

    return train_dataset, val_dataset


def get_equal_len_datasets(dataset1, dataset2):

    """
    Make two datasets the same length
    """

    if len(dataset1) > len(dataset2):

        rand_idxs = np.random.choice(range(len(dataset1)), size=(len(dataset2,)), replace=False)
        subsample_dataset(dataset1, rand_idxs)

    elif len(dataset2) > len(dataset1):

        rand_idxs = np.random.choice(range(len(dataset2)), size=(len(dataset1,)), replace=False)
        subsample_dataset(dataset2, rand_idxs)

    return dataset1, dataset2


def get_ncoc_datasets(train_transform, test_transform, train_classes=range(160),
                       open_set_classes=range(160, 200), balance_open_set_eval=False, split_train_val=True, seed=0):

    np.random.seed(seed)

    #train_dataset_whole = COD(root=cod_root, is_train=True, transform=train_transform)


    # Init train dataset and subsample training classes
   
    train_dataset_whole = CustomCod(root=NCOC_root, transform=train_transform, train=True)
    train_dataset_whole = subsample_classes(train_dataset_whole, include_classes=train_classes)

    # Split into training and validation sets
    train_dataset_split, val_dataset_split = get_train_val_split(train_dataset_whole)
    val_dataset_split.transform = test_transform

    # Get test set for known classes
    #test_dataset_known = COD(root=cod_root, is_train=False, transform=train_transform)
    test_dataset_known = CustomCod(root=NCOC_root, transform=test_transform, train=False)
    test_dataset_known = subsample_classes(test_dataset_known, include_classes=train_classes)

    # Get testset for unknown classes
    test_dataset_unknown = CustomCod(root=NCOC_root, transform=test_transform, train=False)
    test_dataset_unknown = subsample_classes(test_dataset_unknown, include_classes=open_set_classes)

    if balance_open_set_eval:
        test_dataset_known, test_dataset_unknown = get_equal_len_datasets(test_dataset_known, test_dataset_unknown)

    # Either split train into train and val or use test set as val
    train_dataset = train_dataset_split if split_train_val else train_dataset_whole
    val_dataset = val_dataset_split if split_train_val else test_dataset_known

    all_datasets = {
        'train': train_dataset,
        'val': val_dataset,
        'test_known': test_dataset_known,
        'test_unknown': test_dataset_unknown,
    }

    return all_datasets



def get_ncoctest_datasets(train_transform, test_transform, train_classes=range(160),
                       open_set_classes=range(160, 200), balance_open_set_eval=False, split_train_val=True, seed=0):

    np.random.seed(seed)

    #train_dataset_whole = COD(root=cod_root, is_train=True, transform=train_transform)


    # Init train dataset and subsample training classes

    # train_dataset_whole = CustomCod(root=cod_root, transform=train_transform, train=True)
    # train_dataset_whole = subsample_classes(train_dataset_whole, include_classes=train_classes)

    # # Split into training and validation sets
    # train_dataset_split, val_dataset_split = get_train_val_split(train_dataset_whole)
    # val_dataset_split.transform = test_transform

    # Get test set for known classes
    #test_dataset_known = COD(root=cod_root, is_train=False, transform=train_transform)
    test_dataset_known = CustomCod(root=NCOC_root, transform=test_transform, train=False)
    test_dataset_known = subsample_classes(test_dataset_known, include_classes=train_classes)

    # Get testset for unknown classes
    test_dataset_unknown = CustomCod(root=NCOC_root, transform=test_transform, train=False)
    test_dataset_unknown = subsample_classes(test_dataset_unknown, include_classes=open_set_classes)

    if balance_open_set_eval:
        test_dataset_known, test_dataset_unknown = get_equal_len_datasets(test_dataset_known, test_dataset_unknown)

    # Either split train into train and val or use test set as val
    # train_dataset = train_dataset_split if split_train_val else train_dataset_whole
    # val_dataset = val_dataset_split if split_train_val else test_dataset_known

    all_datasets = {
        'test_known': test_dataset_known,
        'test_unknown': test_dataset_unknown,
    }

    return all_datasets



def get_acoc_datasets(train_transform, test_transform, train_classes=range(160),
                       open_set_classes=range(160, 200), balance_open_set_eval=False, split_train_val=True, seed=0):

    np.random.seed(seed)

    #train_dataset_whole = COD(root=cod_root, is_train=True, transform=train_transform)


    # Init train dataset and subsample training classes
    
    train_dataset_whole = CustomCod(root=ACOC_root, transform=train_transform, train=True)
    train_dataset_whole = subsample_classes(train_dataset_whole, include_classes=train_classes)

    # Split into training and validation sets
    train_dataset_split, val_dataset_split = get_train_val_split(train_dataset_whole)
    val_dataset_split.transform = test_transform

    # Get test set for known classes
    #test_dataset_known = COD(root=cod_root, is_train=False, transform=train_transform)
    test_dataset_known = CustomCod(root=ACOC_root, transform=test_transform, train=False)
    test_dataset_known = subsample_classes(test_dataset_known, include_classes=train_classes)

    # Get testset for unknown classes
    test_dataset_unknown = CustomCod(root=ACOC_root, transform=test_transform, train=False)
    test_dataset_unknown = subsample_classes(test_dataset_unknown, include_classes=open_set_classes)

    if balance_open_set_eval:
        test_dataset_known, test_dataset_unknown = get_equal_len_datasets(test_dataset_known, test_dataset_unknown)

    # Either split train into train and val or use test set as val
    train_dataset = train_dataset_split if split_train_val else train_dataset_whole
    val_dataset = val_dataset_split if split_train_val else test_dataset_known

    all_datasets = {
        'train': train_dataset,
        'val': val_dataset,
        'test_known': test_dataset_known,
        'test_unknown': test_dataset_unknown,
    }

    return all_datasets

def get_acoctest_datasets(train_transform, test_transform, train_classes=range(160),
                       open_set_classes=range(160, 200), balance_open_set_eval=False, split_train_val=True, seed=0):

    np.random.seed(seed)

    #train_dataset_whole = COD(root=cod_root, is_train=True, transform=train_transform)


    # Init train dataset and subsample training classes

    # train_dataset_whole = CustomCod(root=cod_root, transform=train_transform, train=True)
    # train_dataset_whole = subsample_classes(train_dataset_whole, include_classes=train_classes)

    # # Split into training and validation sets
    # train_dataset_split, val_dataset_split = get_train_val_split(train_dataset_whole)
    # val_dataset_split.transform = test_transform

    # Get test set for known classes
  
    #test_dataset_known = COD(root=cod_root, is_train=False, transform=train_transform)
    test_dataset_known = CustomCod(root=ACOC_root, transform=test_transform, train=False)
    test_dataset_known = subsample_classes(test_dataset_known, include_classes=train_classes)

    # Get testset for unknown classes
    test_dataset_unknown = CustomCod(root=ACOC_root, transform=test_transform, train=False)
    test_dataset_unknown = subsample_classes(test_dataset_unknown, include_classes=open_set_classes)

    if balance_open_set_eval:
        test_dataset_known, test_dataset_unknown = get_equal_len_datasets(test_dataset_known, test_dataset_unknown)

    # Either split train into train and val or use test set as val
    # train_dataset = train_dataset_split if split_train_val else train_dataset_whole
    # val_dataset = val_dataset_split if split_train_val else test_dataset_known

    all_datasets = {
        'test_known': test_dataset_known,
        'test_unknown': test_dataset_unknown,
    }

    return all_datasets


if __name__ == '__main__':

    x = get_cub_datasets(None, None, split_train_val=False, train_classes=np.random.choice(range(200), size=100, replace=False))
    print([len(v) for k, v in x.items()])
    # z = x['train'][0]
    # debug = 0