import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2
import torchvision
torchvision.disable_beta_transforms_warning()

from models.model import get_model
from utils import load_model
from dataset import FacialKeypointsDataset

import json


RUN_NAME = "2023-11-25_20-04-55"

run_path = f"runs/{RUN_NAME}/"

train_summary = json.load(open(run_path + "train_summary.json"))

MODEL = train_summary["config"]["MODEL"]
IMAGE_SIZE = train_summary["config"]["IMAGE_SIZE"]
BATCH_SIZE = train_summary["config"]["BATCH_SIZE"]
PRETRAINED = train_summary["config"]["PRETRAINED"]

print(f"MODEL: {MODEL}")
print(f"Pretrained: {PRETRAINED}")

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = get_model(MODEL)
model = load_model(model, run_path + "best_model.pth")
model.to(DEVICE)


transforms_test = v2.Compose([
        v2.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225]),
        v2.ToImage(),
        v2.ToDtype(torch.float),
        v2.Resize((IMAGE_SIZE, IMAGE_SIZE), antialias=True),
    ])

test_dataset = FacialKeypointsDataset(csv_file='data/test_frames_keypoints.csv', root_dir='data/test/', 
                                          transform=transforms_test)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)

loss_mae = torch.nn.L1Loss()
loss_mse = torch.nn.MSELoss()

model.eval()

loss_mae_sum = 0
loss_mse_sum = 0

with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(DEVICE), target.to(DEVICE)

        output = model(data)

        loss_mae_sum += loss_mae(output, target).item()
        loss_mse_sum += loss_mse(output, target).item()

    
loss_mae_sum /= len(test_loader)
loss_mse_sum /= len(test_loader)

print(f"MAE: {loss_mae_sum}")
print(f"MSE: {loss_mse_sum}")

# compute number of parameters
num_params = 0
for param in model.parameters():
    num_params += param.numel()

print(f"Number of parameters: {num_params}")