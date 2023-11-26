import torch
from torchvision.transforms import v2
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import json
import argparse 
import time

from models.model import get_model
from utils import get_emotion
from utils import load_model, draw_face_mask


parser = argparse.ArgumentParser(description='Test facial keypoint detection model')
parser.add_argument('--video', type=str, default=None, help='Path to video file')
parser.add_argument('--image', type=str, default=None, help='Path to image file')
parser.add_argument('--camera', action='store_true', help='Use camera')
parser.add_argument('--run', type=str, default="2023-11-25_19-53-09", help='Run name')
parser.add_argument('--save', action='store_true', help='Save output video')
args = parser.parse_args()

global fps


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
    #if there is no samples/output folder, create it
    if not os.path.exists("samples/output"):
        os.mkdir("samples/output")
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
model = torch.jit.script(model)
model.to(DEVICE)


transforms_test = v2.Compose([
    v2.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225]),
    v2.ToImage(),
    v2.ToDtype(torch.float),
    v2.Resize((IMAGE_SIZE, IMAGE_SIZE), antialias=True),
])

def process_frame(frame, framecount):
    frame_orig = frame.copy()

    frame_t = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # frame_t = cv2.resize(frame_t, (IMAGE_SIZE, IMAGE_SIZE))
    frame_t = frame_t / 255.0
    # frame_t = torch.from_numpy(frame_t).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)
    # frame_t = np.expand_dims(frame_t, axis=0)
    frame_t = transforms_test(frame_t).to(DEVICE)
    frame_t = frame_t.unsqueeze(0)

    with torch.no_grad():
        model.eval()
        output = model(frame_t)
        output = output.cpu().numpy()
        output = output.reshape(-1, 2)

    frame = draw_face_mask(frame, output)
    emotion, results = get_emotion(output)

    #
    results["frame"] = str(framecount)



    #put emotion text upper left corner of the frame
    cv2.putText(frame, emotion, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    return frame, results


if args.video or args.camera:
    framecount = 0
    print("FPS: ", fps)
    results = []
    while(True):
        
        ret, frame = cap.read()
        if not ret:
            break
        
        framecount += 1

        # frame, result = process_frame(frame, framecount)
        frame, result = process_frame(frame, framecount)
        results.append(result)
        
            
        cv2.imshow('Facial Keypoints Detection',frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

        if args.save:
            out.write(frame)
    run_id = time.strftime("%Y-%m-%d_%H-%M-%S")
    save_path = f"samples/output/results_{run_id}.json"
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=4)

    cap.release()
    if args.save:
        out.release()
    cv2.destroyAllWindows()

elif args.image:
    frame = process_frame(frame)
    cv2.imshow('frame',frame)

    if args.save:
        #if there is no samples/output folder, create it
        if not os.path.exists("samples/output"):
            os.mkdir("samples/output")
        run_id = time.strftime("%Y-%m-%d_%H-%M-%S")
        save_path = f"samples/output/{run_id}.jpg"
        cv2.imwrite(save_path, frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()