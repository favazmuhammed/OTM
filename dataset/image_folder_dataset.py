import os
import torchvision.datasets as datasets
from PIL import Image
import numpy as np
import cv2
from torch.utils.data import Dataset

from .augmenter import Augmenter

class CustomImageFolderDataset(datasets.ImageFolder):

    def __init__(self,
                 root,
                 transform=None,
                 target_transform=None,
                 loader=datasets.folder.default_loader,
                 is_valid_file=None,
                 low_res_augmentation_prob=0.0,
                 crop_augmentation_prob=0.0,
                 photometric_augmentation_prob=0.0,
                 swap_color_channel=False,
                 output_dir='./',
                 ):

        super(CustomImageFolderDataset, self).__init__(root,
                                                       transform=transform,
                                                       target_transform=target_transform,
                                                       loader=loader,
                                                       is_valid_file=is_valid_file)
        self.root = root
        self.augmenter = Augmenter(crop_augmentation_prob, photometric_augmentation_prob, low_res_augmentation_prob)
        self.swap_color_channel = swap_color_channel
        self.output_dir = output_dir  # for checking the sanity of input images

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        sample = Image.fromarray(np.asarray(sample)[:,:,::-1])

        if self.swap_color_channel:
            # swap RGB to BGR if sample is in RGB
            # we need sample in BGR
            sample = Image.fromarray(np.asarray(sample)[:,:,::-1])

        sample = self.augmenter.augment(sample)

        sample_save_path = os.path.join(self.output_dir, 'training_samples', 'sample.jpg')
        if not os.path.isfile(sample_save_path):
            os.makedirs(os.path.dirname(sample_save_path), exist_ok=True)
            cv2.imwrite(sample_save_path, np.array(sample))  # the result has to look okay (Not color swapped)

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target


class PoseDataset(Dataset):
    def __init__(self, train_df, root, transform, head, output_dir = None,
                 low_res_augmentation_prob=0.0,
                 crop_augmentation_prob=0.0,
                 photometric_augmentation_prob=0.0,
                 swap_color_channel=None,
                 save_samples = False):

        self.imgids = train_df['ImageId'].values
        self.z_poses = train_df['cose2theta'].values
        self.thetas = train_df['theta'].values

        self.loader = datasets.folder.default_loader
        self.root = root
        self.transform = transform
        self.swap_color_channel = swap_color_channel
        self.augmenter = Augmenter(crop_augmentation_prob, photometric_augmentation_prob, low_res_augmentation_prob)
        self.save_samples = save_samples
        self.output_dir = output_dir

        assert len(self.imgids) == len(self.z_poses)

    def __getitem__(self, index):
        imgid = self.imgids[index]
        z_pose = self.z_poses[index]
        theta = self.thetas[index]

        
        x1 = self.loader(os.path.join(self.root, imgid))
        x1 = Image.fromarray(np.asarray(x1)[:,:,::-1])
        # resize image
        x1 = x1.resize((112,112))

        # print(np.asarray(x1))


        if self.swap_color_channel:
            # swap RGB to BGR if sample is in RGB
            # we need sample in BGR
            x1 = Image.fromarray(np.asarray(x1)[:,:,::-1])
        
        x1 = self.augmenter.augment(x1)

        if self.save_samples:
          augment_save_path = os.path.join(self.output_dir, 'training_samples', imgid.split('/')[1])
          pose_save_path = os.path.join(self.output_dir, 'z_pose_error', imgid.split('/')[1])

          if not os.path.isfile(augment_save_path):
              os.makedirs(os.path.dirname(augment_save_path), exist_ok=True)
        
          # cv2.imwrite(augment_save_path, np.array(x1))

          if not os.path.isfile(pose_save_path):
              os.makedirs(os.path.dirname(pose_save_path), exist_ok=True)
          
          if abs(theta) > 90:
            cv2.imwrite(pose_save_path, np.array(x1))

        
        # x1 = torch.tensor(x1)
        # x2 = torch.tensor(x1)
        y = int(imgid.split('/')[0])

        if self.transform is not None:
            x1 = self.transform(x1)

        return x1, z_pose, y

    def __len__(self):
        return len(self.imgids)

