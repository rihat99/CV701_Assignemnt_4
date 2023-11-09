from dataset import FacialKeypointsDataset
# from models.unet import UNet
from engine import trainer
from utils import plot_results
from model import get_model

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2
import torchvision
torchvision.disable_beta_transforms_warning()


import matplotlib.pyplot as plt
import numpy as np
import random

import yaml
import json
import time
import os


config = yaml.load(open('config.yaml', 'r'), Loader=yaml.FullLoader)

LEARNING_RATE = float(config["LEARNING_RATE"])
BATCH_SIZE = int(config["BATCH_SIZE"])
NUM_EPOCHS = int(config["NUM_EPOCHS"])

LOSS = config["LOSS"]

IMAGE_SIZE = int(config["IMAGE_SIZE"])
MODEL = config["MODEL"]
PRETRAINED = config["PRETRAINED"]

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using {DEVICE} device")


def START_seed():
    seed = 9
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed) 


def main():
    START_seed()

    #run id is date and time of the run
    run_id = time.strftime("%Y-%m-%d_%H-%M-%S")

    #create folder for this run in runs folder
    os.mkdir("./runs/" + run_id)

    save_dir = "./runs/" + run_id
    

    #load data
    transforms_train = v2.Compose([
        v2.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225]),
        v2.ToImage(),
        v2.ToDtype(torch.float),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomRotation(degrees=(-25, 25)),
        v2.RandomAffine(degrees=(-15, 15), translate=(0.2, 0.2), scale=(0.8, 1.2), shear=(-10, 10, -10, 10)),
        v2.RandomResizedCrop((IMAGE_SIZE, IMAGE_SIZE), scale=(0.8, 1.0), antialias=True),
        v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        v2.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
        v2.RandomAutocontrast(p=0.2),
        v2.RandomEqualize(p=0.2),
    ])

    transforms_test = v2.Compose([
        v2.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225]),
        v2.ToImage(),
        v2.ToDtype(torch.float),
        v2.Resize((IMAGE_SIZE, IMAGE_SIZE), antialias=True),
    ])

    train_dataset = FacialKeypointsDataset(csv_file='data/training_frames_keypoints.csv', root_dir='data/training/',
                                        transform=transforms_train)

    test_dataset = FacialKeypointsDataset(csv_file='data/test_frames_keypoints.csv', root_dir='data/test/', 
                                          transform=transforms_test)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)

    #load model
    model = get_model(MODEL, PRETRAINED)
    model.to(DEVICE)
    
    #load optimizer
    if LOSS == "MSE":
        loss = torch.nn.MSELoss()
    else:
        raise Exception("Loss not implemented")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    #train model
    results = trainer(
        model=model,
        train_loader=train_loader,
        val_loader=test_loader,
        loss_fn=loss,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        device=DEVICE,
        epochs=NUM_EPOCHS,
        save_dir=save_dir,
    )

    train_summary = {
        "config": config,
        "results": results,
    }

    with open(save_dir + "/train_summary.json", "w") as f:
        json.dump(train_summary, f, indent=4)

    plot_results(results["train_loss"], results["val_loss"], "Loss", save_dir)

if __name__ == "__main__":
    main()

