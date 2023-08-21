import os
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import pandas as pd
import evaluate_utils
from dataset.image_folder_dataset import CustomImageFolderDataset, PoseFaceDataset
from dataset.classroom_dataset import ClassroomValidationDataset
from dataset.five_validation_dataset import FiveValidationDataset
from dataset.record_dataset import AugmentRecordDataset


data_root = "./"
train_data_path = "../adaface_org/AdaFace/data/faces_emore/imgs_subset"
head = "poseadaface"
train_df = pd.read_csv(os.path.join(data_root, './data/webface4m_subset_posedata.csv'))
train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

if __name__ == "__main__":
    train_dir = os.path.join(data_root, train_data_path)
    train_dataset = PoseFaceDataset(train_df = train_df,
                                    root = train_dir,
                                    transform=train_transform,
                                    head = head,
                                    low_res_augmentation_prob=0.2,
                                    crop_augmentation_prob=0.2,
                                    photometric_augmentation_prob=0.2,
                                    swap_color_channel = True)

    dataloader = DataLoader(train_dataset, batch_size=8, num_workers=16, shuffle=True)

    for images, z_pose, labels in dataloader:
        print(images.size, z_pose.size, labels.size)

