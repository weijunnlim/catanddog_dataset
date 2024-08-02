import os
import pandas as pd
import torch
from skimage import io
from torch.utils.data import Dataset
from skimage.transform import resize

class CustomImageDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self,index):
        img_path=os.path.join(self.root_dir,self.data.iloc[index,0])
        image=io.imread(img_path)
        image=resize(image,(250,250),anti_aliasing=True)
        y_label=torch.tensor(int(self.data.iloc[index,1]))
        
        if self.transform:
            image=self.transform(image)
        return (image,y_label)