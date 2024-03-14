import os
import glob
from pathlib import Path
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torch



class MnistDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.data = glob.glob(os.path.join(data_dir + '/*/*.jpg'))
        # print(self.data)
        self.transform = transform

        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, index):
            image = self.data[index]
            #print(image)
            #image = self.data[index].split('\\')[-2]
            label = Path(image).parts[-2]
            #print(label)
            image = Image.open(image)
            #print(image, label)
            image = np.array(image)

            image = torch.FloatTensor(image)
            label = torch.LongTensor([int(label)])
            return image, label
        

        if __name__ == "__main__":
            dataset = MnistDataset(data_dir = './data/MNIST - JPG - training', transform=None)
            print(len(dataset))
            print(dataset[0])            