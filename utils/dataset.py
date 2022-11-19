import os
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import pandas as pd
from torchvision.io import read_image

# labels_map = {0: "Normal",
#              1: "Pneumo"}

class CustomImageDataset(Dataset):
    def __init__(self, annotation_df, transform=None):
        self.annotation_df = annotation_df
        self.transform = transform

    def __len__(self):
        return len(self.annotation_df)

    def __getitem__(self, idx):
        img_path = self.annotation_df.iloc[idx, 0]
        image = read_image(img_path)
        
        # Convert 1-channel to 3-channel
        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)
        
        label = self.annotation_df.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        return image, label