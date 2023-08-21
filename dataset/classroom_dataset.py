import os
import torchvision.datasets as datasets
from PIL import Image
import numpy as np
import cv2
from torch.utils.data import Dataset

from .augmenter import Augmenter



class ClassroomValidationDataset(Dataset):
    def __init__(self, valid_data_df, root, transform, swap_color_channel=None):

        self.img1_ids = valid_data_df['ImageId1'].values
        self.img2_ids = valid_data_df['ImageId2'].values
        self.labels = valid_data_df['issame']

        self.loader = datasets.folder.default_loader
        self.root = root
        self.transform = transform
        self.swap_color_channel = swap_color_channel

        assert len(self.img1_ids) == len(self.img2_ids)
        assert len(self.img2_ids) == len(self.labels)

    def __getitem__(self, index):
        x1_id = self.img1_ids[index]
        x2_id = self.img2_ids[index]

        
        x1 = self.loader(os.path.join(self.root, x1_id))
        x1 = Image.fromarray(np.asarray(x1)[:,:,::-1])
        # resize image
        x1 = x1.resize((112,112))

        x2 = self.loader(os.path.join(self.root, x2_id))
        x2 = Image.fromarray(np.asarray(x2)[:,:,::-1])
        # resize image
        x2 = x2.resize((112,112))


        if self.swap_color_channel:
            # swap RGB to BGR if sample is in RGB
            # we need sample in BGR
            x1 = Image.fromarray(np.asarray(x1)[:,:,::-1])
            x2 = Image.fromarray(np.asarray(x2)[:,:,::-1])


        # x1 = torch.tensor(x1)
        # x2 = torch.tensor(x1)
        y = self.labels[index]

        if self.transform is not None:
            x1 = self.transform(x1)
            x2 = self.transform(x2)
        

        return x1, x2, y

    def __len__(self):
        return len(self.img1_ids)

