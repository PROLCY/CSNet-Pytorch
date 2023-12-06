import os
import random

import cv2
import numpy as np
from PIL import Image
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from config import Config
from csnet import CSNet
from dataset import SCDataset, BCDataset, UNDataset
from image_utils.image_augmentation import get_augmented_image
from image_utils.image_perturbation import get_perturbed_image
from test import test_while_training

Image.MAX_IMAGE_PIXELS = None

def not_convert_to_tesnor(batch):
        return batch

def build_dataloader(cfg):
    sc_dataset = SCDataset('train', cfg)
    sc_loader = DataLoader(dataset=sc_dataset,
                              batch_size=cfg.scored_crops_batch_size,
                              collate_fn=not_convert_to_tesnor,
                              shuffle=True,
                              num_workers=cfg.num_workers)
    bc_dataset = BCDataset('train', cfg)
    bc_loader = DataLoader(dataset=bc_dataset,
                              batch_size=cfg.best_crop_K,
                              collate_fn=not_convert_to_tesnor,
                              shuffle=True,
                              num_workers=cfg.num_workers)
    un_dataset = UNDataset('train', cfg)
    un_loader = DataLoader(dataset=un_dataset,
                              batch_size=cfg.unlabeled_P,
                              shuffle=True,
                              collate_fn=not_convert_to_tesnor,
                              num_workers=cfg.num_workers)

    return sc_loader, bc_loader, un_loader

class Trainer(object):
    def __init__(self, model, cfg):
        self.cfg = cfg
        self.model = model

        self.image_dir = self.cfg.image_dir

        self.device = torch.device('cuda:{}'.format(self.cfg.gpu_id))

        self.sc_loader, self.bc_loader, self.un_loader = build_dataloader(cfg)

        self.sc_random_crops_count = self.cfg.scored_crops_N

        self.sc_batch_size = self.cfg.scored_crops_batch_size
        self.bc_batch_size = self.cfg.best_crop_K
        self.un_batch_size = self.cfg.unlabeled_P

        self.perturbation_type_list = self.cfg.perturbation_type_list
        self.augmentation_type_list = self.cfg.augmentation_type_list
        
        self.loss_fn = torch.nn.MarginRankingLoss(margin=self.cfg.pairwise_margin, reduction='mean')
        self.optimizer = optim.Adam(params=model.parameters(),
                                    lr=self.cfg.learning_rate,
                                    weight_decay=self.cfg.weight_decay)

        self.epoch = 0
        self.max_epoch = self.cfg.max_epoch
        
        self.train_iter = 0
        self.sc_iter = 0
        self.bc_iter = 0
        self.un_iter = 0
        self.sc_loss_sum = 0
        self.bc_loss_sum = 0
        self.un_loss_sum = 0

        self.transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=self.cfg.mean, std=self.cfg.std)
        ])


    def training(self):
        print('\n======train start======\n')
        bc_iterator = iter(self.bc_loader)
        un_iterator = iter(self.un_loader)
        for index, data in enumerate(self.sc_loader):
            self.model.train().to(self.device)

            sc_data_list = data

            try:
                bc_data_list = next(bc_iterator)
            except:
                bc_iterator = iter(self.bc_loader)
            
            try:
                un_data_list = next(un_iterator)
            except:
                un_iterator = iter(self.un_loader)
            
            sc_pos_images, sc_neg_images = self.make_pairs_scored_crops(sc_data_list[0])
            if len(sc_pos_images) == 0:
                sc_loss = None
            else:
                sc_loss = self.calculate_pairwise_ranking_loss(sc_pos_images, sc_neg_images)
            
            bc_pos_images, bc_neg_images = self.make_pairs_perturb(bc_data_list, labeled=True)
            if len(bc_pos_images) == 0:
                bc_loss = None
            else:
                bc_loss = self.calculate_pairwise_ranking_loss(bc_pos_images, bc_neg_images)
            
            un_pos_images, un_neg_images = self.make_pairs_perturb(un_data_list, labeled=False)
            un_loss = self.calculate_pairwise_ranking_loss(un_pos_images, un_neg_images)
            
            total_loss = 0
            if sc_loss != None:
                total_loss += sc_loss
                self.sc_iter += 1
            if bc_loss != None:
                total_loss += bc_loss
                self.bc_iter += 1
            
            total_loss += un_loss
            self.un_iter += 1

            loss_log = f'L_SC: {sc_loss.item() if sc_loss != None else 0.0:.5f}, L_BC: {bc_loss.item() if bc_loss != None else 0.0:.5f}, L_UN: {un_loss.item():.5f}'
            print(loss_log)
            
            self.sc_loss_sum += sc_loss.item() if sc_loss != None else 0
            self.bc_loss_sum += bc_loss.item() if bc_loss != None else 0
            self.un_loss_sum += un_loss.item()
            
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            self.train_iter += 1

        print('\n======train end======\n')

    def convert_image_list_to_tensor(self, image_list):
        tensor = []
        for image in image_list:
            # Grayscale to RGB
            if len(image.getbands()) == 1:
                rgb_image = Image.new("RGB", image.size)
                rgb_image.paste(image, (0, 0, image.width, image.height))
                image = rgb_image
            np_image = np.array(image)
            np_image = cv2.resize(np_image, self.cfg.image_size)
            tensor.append(self.transformer(np_image))
        tensor = torch.stack(tensor, dim=0)
        
        return tensor

    def calculate_pairwise_ranking_loss(self, pos_images, neg_images):
        pos_tensor = self.convert_image_list_to_tensor(pos_images)
        neg_tensor = self.convert_image_list_to_tensor(neg_images)
        
        tensor_concat = torch.cat((pos_tensor, neg_tensor), dim=0).to(self.device)
            
        score_concat = self.model(tensor_concat)
        pos_score, neg_score = torch.split(score_concat, [score_concat.shape[0] // 2, score_concat.shape[0] // 2])
        
        target = torch.ones((pos_score.shape[0], 1)).to(self.device)
        loss = self.loss_fn(pos_score, neg_score, target=target)

        return loss

    def run(self):
        for epoch in range(self.epoch, self.max_epoch):
            self.epoch = epoch + 1
            self.training()

            # save checkpoint
            checkpoint_path = os.path.join(self.cfg.weight_dir, 'checkpoint-weight.pth')
            torch.save(self.model.state_dict(), checkpoint_path)
            print('Checkpoint Saved...\n')

            epoch_log = 'epoch: %d / %d, lr: %8f' % (self.epoch, self.max_epoch, self.optimizer.param_groups[0]['lr'])
            print(epoch_log)

            average_sc_loss = self.sc_loss_sum / self.sc_iter
            average_bc_loss = self.bc_loss_sum / self.bc_iter
            average_un_loss = self.un_loss_sum / self.un_iter

            train_log = f'average_sc_loss:{average_sc_loss}\naverage_bc_loss:{average_bc_loss}\naverage_un_loss:{average_un_loss}'
            print(train_log)

            test_while_training()

            self.train_iter = 0
            self.sc_loss_sum = 0
            self.bc_loss_sum = 0
            self.un_loss_sum = 0
            self.sc_iter = 0
            self.bc_iter = 0
            self.un_iter = 0

    def shuffle_two_lists_in_same_order(self, list1, list2):
        combined_lists = list(zip(list1, list2))
        random.shuffle(combined_lists)
        shuffled_list1, shuffled_list2 = zip(*combined_lists)
        return list(shuffled_list1), list(shuffled_list2)

    def make_pairs_scored_crops(self, data):
        image_name = data[0]
        image = Image.open(os.path.join(self.image_dir, image_name))
        crops_list = data[1]
        crops_list = random.sample(crops_list, self.sc_random_crops_count)

        # sort in descending order by score
        sorted_crops_list = sorted(crops_list, key = lambda x: -x['score'])
        
        boudning_box_pairs = []
        for i in range(len(sorted_crops_list)):
            for j in range(i + 1, len(sorted_crops_list)):
                if sorted_crops_list[i]['score'] == sorted_crops_list[j]['score']:
                    continue
                boudning_box_pairs.append((sorted_crops_list[i]['crop'], sorted_crops_list[j]['crop']))

        pos_images = []
        neg_images = []
        for pos_box, neg_box in boudning_box_pairs:
            pos_image = image.crop(pos_box)
            neg_image = image.crop(neg_box)
            pos_images.append(pos_image)
            neg_images.append(neg_image)

            # augmentation by filling zero pixels
            augmented_pos_image, augmented_neg_image = self.augment_pair((pos_image, neg_image), labeled=True)
            pos_images.append(augmented_pos_image)
            neg_images.append(augmented_neg_image)

        if len(pos_images) != 0:
            pos_images, neg_images = self.shuffle_two_lists_in_same_order(pos_images, neg_images)

        return pos_images, neg_images

    def make_pair_perturb(self, data, labeled=True):
        if labeled == True:
            image_name = data[0]
            image = Image.open(os.path.join(self.image_dir, image_name))
            best_crop_bounding_box = data[1]
            best_crop = image.crop(best_crop_bounding_box)
        else:
            image_name = data
            image = Image.open(os.path.join(os.path.join(self.image_dir, 'unlabeled'), image_name))
            best_crop_bounding_box = [0, 0, image.size[0], image.size[1]]
            best_crop = image

        type_list = self.perturbation_type_list
        selected_type = random.choice(type_list)

        allow_zero_pixel = not labeled

        perturbed_image = get_perturbed_image(image, best_crop_bounding_box, allow_zero_pixel, type=selected_type)
        if perturbed_image == None:
            return None

        return best_crop, perturbed_image

    def make_pairs_perturb(self, data_list, labeled):
        pos_images = []
        neg_images = []
        for data in data_list:
            image_pair = self.make_pair_perturb(data, labeled)
            if image_pair == None:
                continue

            pos_image = image_pair[0]
            neg_image = image_pair[1]
            pos_images.append(pos_image)
            neg_images.append(neg_image)

            # augmentation by filling zero pixels
            augmented_pos_image, augmented_neg_image = self.augment_pair((pos_image, neg_image), labeled)
            pos_images.append(augmented_pos_image)
            neg_images.append(augmented_neg_image)

        if len(pos_images) != 0:
            pos_images, neg_images = self.shuffle_two_lists_in_same_order(pos_images, neg_images)

        return pos_images, neg_images

    def augment_pair(self, image_pair, labeled=True):
        pos_image = image_pair[0]
        neg_image = image_pair[1]
        type_list = self.augmentation_type_list
        selected_type = random.choice(type_list)
        if labeled:
            augment_pos_image = get_augmented_image(pos_image, selected_type)
            augment_neg_image = get_augmented_image(neg_image, selected_type)
        else:
            augment_pos_image = get_augmented_image(pos_image, selected_type)
            augment_neg_image = neg_image

        return augment_pos_image, augment_neg_image

if __name__ == '__main__':
    cfg = Config()
    
    model = CSNet(cfg)
    # weight_file = os.path.join(cfg.weight_dir, 'checkpoint-weight.pth')
    # model.load_state_dict(torch.load(weight_file))

    trainer = Trainer(model, cfg)
    trainer.run()