import os
import cv2
import glob
import torch
from tqdm import tqdm
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader

class ASLDataset(Dataset):
    def __init__(self, split, im_path, im_ext='jpg'):
        self.split = split
        self.im_ext = im_ext
        self.images, self.labels = self.load_images(im_path)
        
    def load_images(self, im_path):
        assert os.path.exists(im_path), "Images path {} does not exist".format(im_path)
        ims = []
        labels = []
        for d_name in tqdm(os.listdir(im_path)):
            dir_path = os.path.join(im_path, d_name)
            if os.path.isdir(dir_path):
                for fname in glob.glob(os.path.join(dir_path, '*.{}'.format(self.im_ext))):
                    ims.append(fname)
                    labels.append(d_name)  # Use directory name as label
        print('Found {} images for split {}'.format(len(ims), self.split))
        return ims, labels

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        im = cv2.imread(self.images[index], cv2.IMREAD_GRAYSCALE)
        label = self.labels[index]
        # Convert to 0 to 255 into -1 to 1
        im = 2 * (im / 255) - 1
        # Convert H, W into 1, H, W
        im_tensor = torch.from_numpy(im).unsqueeze(0)
        return im_tensor, torch.tensor(ord(label) - ord('A'))  # Convert label to a numerical value


def get_data_loaders(train_path, test_path, batch_size=16, im_ext='jpg'):
    train_dataset = ASLDataset('train', train_path, im_ext)
    test_dataset = ASLDataset('test', test_path, im_ext)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, test_loader
