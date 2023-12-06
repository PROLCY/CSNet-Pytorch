import os

class Config:
    def __init__(self):
        
        self.image_dir = './data/image'

        self.data_dir = './data/annotation'
        self.weight_dir = './weight'

        self.scored_crops_data = os.path.join(self.data_dir, 'scored_crops')
        self.best_crop_data = os.path.join(self.data_dir, 'best_crop')
        self.unlabeled_data = os.path.join(self.data_dir, 'unlabeled')

        self.gpu_id = 0
        self.num_workers = 4

        self.perturbation_type_list = ['shift', 'zoom-out', 'crop', 'rotate']
        self.augmentation_type_list = ['shift', 'zoom-out', 'rotate']

        self.pairwise_margin = 0.3
        self.learning_rate = 2e-5
        self.weight_decay = 5e-4

        self.max_epoch = 100

        self.scored_crops_batch_size = 1
        self.scored_crops_N = 11
        self.test_crops_N = 12
        self.best_crop_K = 8
        self.unlabeled_P = 8
        self.test_score_gap = 0.5

        self.image_size = (224, 224)
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]