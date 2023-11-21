import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import json
import argparse 
import time

from models.model import get_model
from utils import load_model, calculate_MAR, mouth_corner_angle, classify_emotion, compute_curvature_for_lip


parser = argparse.ArgumentParser(description='Test facial keypoint detection model')
parser.add_argument('--video', type=str, default=None, help='Path to video file')
parser.add_argument('--image', type=str, default=None, help='Path to image file')
parser.add_argument('--camera', action='store_true', help='Use camera')
parser.add_argument('--run', type=str, default="2023-11-22_01-14-02", help='Run name')
parser.add_argument('--save', action='store_true', help='Save output video')
args = parser.parse_args()

if args.video is not None:
    video_path= args.video
    cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

elif args.camera:
    cap = cv2.VideoCapture(0)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

elif args.image is not None:
    image_path = args.image
    frame = cv2.imread(image_path)
    height, width, _ = frame.shape

if args.save and (args.video or args.camera):
    # save in mp4 format
    run_id = time.strftime("%Y-%m-%d_%H-%M-%S")
    save_path = f"samples/output/{run_id}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(save_path, fourcc, fps, (width, height))
    

run_name = args.run
run_path = f"runs/{run_name}/"

train_summary = json.load(open(run_path + "train_summary.json"))

MODEL = train_summary["config"]["MODEL"]
IMAGE_SIZE = train_summary["config"]["IMAGE_SIZE"]

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


model = get_model(MODEL)
model = load_model(model, run_path + "best_model.pth")
model.to(DEVICE)






def process_frame(frame):
    frame_orig = frame.copy()

    frame_t = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_t = cv2.resize(frame_t, (IMAGE_SIZE, IMAGE_SIZE))
    frame_t = frame_t / 255.0
    frame_t = torch.from_numpy(frame_t).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)

    with torch.no_grad():
        model.eval()
        output = model(frame_t)
        output = output.cpu().numpy()
        output = output.reshape(-1, 2)

        keypoints = {
            'left_mouth_corner': None,
            'top_lip': None,
            'right_mouth_corner': None,
            'bottom_lip': None,
            'all_top_lip': None,
            'all_bottom_lip': None,
        }
           
        all_up_lip = []
        all_bottom_lip = []
        for i in range(output.shape[0]):

            if i == 48:
                keypoints['left_mouth_corner'] = np.array([output[i, 0], output[i, 1]])
            elif i == 51:
                keypoints['top_lip'] = np.array([output[i, 0], output[i, 1]])
                keypoints['right_mouth_corner'] = np.array([output[i, 0], output[i, 1]])
            elif i == 57:
                keypoints['bottom_lip'] = np.array([output[i, 0], output[i, 1]])

            if i >= 48 and i <= 54:
                all_up_lip.append([output[i, 0], output[i, 1]])
            elif i >= 55 and i <= 60 or i == 64:
                all_bottom_lip.append([output[i, 0], output[i, 1]])

            if i <= 16:
                color = (0, 0, 255)
            elif i <= 26:
                color = (0, 255, 255)
            elif i <= 35:
                color = (0, 255, 0)
            elif i <= 47:
                color = (255, 255, 0)
            else:
                color = (255, 0, 0)

            circle_size = (height + width) // 300
            cv2.circle(frame, (int(output[i, 0] * width), int(output[i, 1] * height)), circle_size, color, -1)
        
        keypoints['all_top_lip'] = np.array(all_up_lip)
        keypoints['all_bottom_lip'] = np.array(all_bottom_lip)

        MAR = calculate_MAR(keypoints)
        angle_left, angle_right = mouth_corner_angle(keypoints)
        curvature_bottom_lip = compute_curvature_for_lip(keypoints['all_bottom_lip'])
        curvature_top_lip = compute_curvature_for_lip(keypoints['all_top_lip'])
        curvature_ratio = curvature_bottom_lip / curvature_top_lip

        emotion = classify_emotion(MAR, angle_left, angle_right, curvature_ratio)


    frame = cv2.addWeighted(frame, 0.4, frame_orig, 0.6, 0)
    
    #put emotion text upper left corner of the frame
    cv2.putText(frame, emotion, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)



    return frame


if args.video or args.camera:
    while(True):
        
        ret, frame = cap.read()
        if not ret:
            break

        frame = process_frame(frame)

        cv2.imshow('Facial Keypoints Detection',frame)
        if cv2.waitKey(50) & 0xFF == ord('q'):
            break

        if args.save:
            out.write(frame)

    cap.release()
    if args.save:
        out.release()
    cv2.destroyAllWindows()

elif args.image:
    frame = process_frame(frame)
    cv2.imshow('frame',frame)

    if args.save:
        run_id = time.strftime("%Y-%m-%d_%H-%M-%S")
        save_path = f"samples/output/{run_id}.jpg"
        cv2.imwrite(save_path, frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()