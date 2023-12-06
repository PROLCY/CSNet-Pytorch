import os

import json
from torch.utils.data import Dataset

from config import Config

# scored crops dataset
class SCDataset(Dataset):
    def __init__(self, mode, cfg) :
        self.cfg = cfg

        self.dataset_path = self.cfg.scored_crops_data
        
        if mode == 'train':
            self.annotation_path = os.path.join(self.dataset_path, 'crops_training_set.json')
            self.random_crops_count = self.cfg.scored_crops_N       
            
        if mode == 'test':
            self.annotation_path = os.path.join(self.dataset_path, 'crops_testing_set.json')
            self.random_crops_count = self.cfg.test_crops_N

        self.image_list, self.crops_list = self.build_data_list()

    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, index):
        image = self.image_list[index]
        crops_list = self.crops_list[index]
        return image, crops_list

    def build_data_list(self):
        data_list = []
        with open(self.annotation_path, 'r') as f:
            data_list = json.load(f)
            
        image_list = []
        crops_list = []

        for data in data_list:
            image_list.append(data['name'])
            crops_list.append(data['crops'])
        
        return image_list, crops_list

# best crop dataset
class BCDataset(Dataset):
    def __init__(self, mode, cfg) :
        self.cfg = cfg

        self.dataset_path = self.cfg.best_crop_data
        
        if mode == 'train':
            self.annotation_path = os.path.join(self.dataset_path, 'best_training_set.json')

        if mode == 'test':
            self.annotation_path = os.path.join(self.dataset_path, 'best_testing_set.json')

        self.image_list, self.best_crop_list = self.build_data_list()

    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, index):
        image_name = self.image_list[index]
        best_crop_bounding_box = self.best_crop_list[index]
        return image_name, best_crop_bounding_box

    def build_data_list(self):
        data_list = []
        with open(self.annotation_path, 'r') as f:
            data_list = json.load(f)
        
        image_list = []
        best_crop_list = []

        for data in data_list:
            image_list.append(data['name'])
            best_crop_list.append(data['crop'])
        
        return image_list, best_crop_list
    
# unlabeled dataset
class UNDataset(Dataset):
    def __init__(self, mode, cfg) :
        self.cfg = cfg

        self.dataset_path = self.cfg.unlabeled_data
        
        if mode == 'train':
            self.annotation_path = os.path.join(self.dataset_path, 'unlabeled_training_set.json')

        self.image_list = self.build_data_list()

    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, index):
        image_name = self.image_list[index]
        return image_name

    def build_data_list(self):
        data_list = []
        with open(self.annotation_path, 'r') as f:
            data_list = json.load(f)
        
        image_list = []

        for data in data_list:
            image_list.append(data['name'])
        
        return image_list

if __name__ == '__main__':
    cfg = Config()
    sc_dataset = SCDataset('train', cfg)
    bc_dataset = BCDataset('train', cfg)
    un_dataset = UNDataset('train', cfg)
    print(sc_dataset.__getitem__(0))
    print(bc_dataset.__getitem__(0))
    print(un_dataset.__getitem__(0))