import glob
import os
import matplotlib.image as mpimg
import pandas as pd
import numpy as np
import cv2

from torch.utils.data import Dataset
from torchvision import tv_tensors
import torch




class FacialKeypointsDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform=None, heatmap=False):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.key_pts_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.heatmap = heatmap

    def __len__(self):
        return len(self.key_pts_frame)

    def __getitem__(self, idx):
        image_name = os.path.join(self.root_dir, self.key_pts_frame.iloc[idx, 0])

        # image = mpimg.imread(image_name)
        image = cv2.imread(image_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image / 255.0

        # if image has an alpha color channel, get rid of it
        # if image.shape[2] == 4:
        #     image = image[:, :, 0:3]

        key_pts = self.key_pts_frame.iloc[idx, 1:].to_numpy()

        key_pts = key_pts.astype("float").reshape(-1, 2)


        key_pts = np.concatenate([key_pts, np.zeros(key_pts.shape)], axis=1)
        key_pts = tv_tensors.BoundingBoxes(key_pts, format='XYWH', canvas_size=image.shape[0:2])

        if self.transform:
            image, key_pts = self.transform(image, key_pts)

        key_pts = key_pts[:, 0:2]

        if not self.heatmap:

            # # normalize key points
            key_pts[:, 0] = key_pts[:, 0] / image.shape[1]
            key_pts[:, 1] = key_pts[:, 1] / image.shape[2]
            
            # normalize key points -1 to 1
            # key_pts[:, 0] = (key_pts[:, 0] - image.shape[1]/2) / (image.shape[1]/2)
            # key_pts[:, 1] = (key_pts[:, 1] - image.shape[2]/2) / (image.shape[2]/2)

            key_pts = key_pts.type(torch.float)
            
            #flatten key points
            key_pts = key_pts.reshape(-1)

            return image, key_pts
        
        else:

            keypoints_heatmap = np.zeros((224, 224, 68))
            sigma = 3
            window_size = 21

            for i in range(key_pts.shape[0]):
                keypoints_y = int(key_pts[i, 0])
                keypoints_x = int(key_pts[i, 1])

                for x in range(keypoints_x - window_size, keypoints_x + window_size):
                    for y in range(keypoints_y - window_size, keypoints_y + window_size):
                        if x < 0 or y < 0 or x >= 224 or y >= 224:
                            continue
                        # keypoints_heatmap[x, y, i] = np.exp(-((x - keypoints_x) ** 2 + (y - keypoints_y) ** 2) / (2 * sigma ** 2)) / (2 * np.pi * sigma ** 2)
                        keypoints_heatmap[x, y, i] = np.exp(-((x - keypoints_x) ** 2 + (y - keypoints_y) ** 2) / (2 * sigma ** 2)) 

            keypoints_heatmap = torch.tensor(keypoints_heatmap, dtype=torch.float)
            keypoints_heatmap = keypoints_heatmap.permute(2, 0, 1)

            return image, keypoints_heatmap