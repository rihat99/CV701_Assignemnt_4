import torch

from matplotlib import pyplot as plt
import numpy as np
import sklearn
from scipy.interpolate import interp1d
from scipy.integrate import quad
import cv2

def plot_results(train_data, val_data, label, save_dir):

    plt.figure(figsize=(6, 6))
    plt.title(label)
    plt.xlabel('Epoch')
    plt.ylabel(label)
    plt.plot(train_data, label=f'Train {label}')
    plt.plot(val_data, label=f'Validation {label}')
    plt.legend()
    plt.savefig(save_dir + f'/{label}.png')

def plot_several_results(all_results, variable_name):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    i = 0
    for key, value in all_results.items():
        if i == 0: color = 'blue'
        elif i == 1: color = 'orange'
        elif i == 2: color = 'green'
        else : color = 'red'

        ax1.plot(value['train_loss'], label=f'{variable_name}={key}, train loss', color=color, )
        ax1.plot(value['val_loss'], label=f'{variable_name}={key}, val loss', color=color, linestyle='dashed')
        ax1.legend()

        ax2.plot(value['train_acc'], label=f'{variable_name}={key}, train acc', color=color, )
        ax2.plot(value['val_acc'], label=f'{variable_name}={key}, val acc', color=color, linestyle='dashed')
        ax2.legend()

        i+=1

    plt.show()

def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(model, path):
    model.load_state_dict(torch.load(path))
    return model


# Define a function to fit a quadratic curve to the keypoints of the lips
def fit_quadratic_curve(keypoints):
    # Fit a quadratic function (y = ax^2 + bx + c) to the keypoints
    x = keypoints[:, 0]
    y = keypoints[:, 1]
    coefficients = np.polyfit(x, y, 2)  # Fit a second-degree polynomial
    return coefficients

# Define a function to calculate curvature of the quadratic curve at a specific point
def curvature_of_quadratic(coefficients, x):
    # Curvature formula for a quadratic function (y = ax^2 + bx + c) is given by:
    # K = |2a| / (1 + (2ax + b)^2)^(3/2)
    a, b, _ = coefficients
    curvature = np.abs(2 * a) / (1 + (2 * a * x + b) ** 2) ** 1.5
    return curvature

# Define a function to compute the curvature for multiple points along the lip
def compute_curvature_for_lip(keypoints):
    # We will compute the curvature at multiple points along the lip and take the average
    coefficients = fit_quadratic_curve(keypoints)
    # Generate x values along the lip
    x_values = np.linspace(keypoints[:, 0].min(), keypoints[:, 0].max(), 100)
    # Calculate the curvature for each x value
    curvatures = [curvature_of_quadratic(coefficients, x) for x in x_values]
    # Return the average curvature
    return np.mean(curvatures)

def calculate_MAR(keypoints):
    # Calculate the Mouth Aspect Ratio (MAR)
    width = np.linalg.norm(keypoints['left_mouth_corner'] - keypoints['right_mouth_corner'])
    height = np.linalg.norm(keypoints['top_lip'] - keypoints['bottom_lip'])
    MAR = height / width
    return MAR

def mouth_corner_angle(keypoints):
    # Calculate the angle at the mouth corners, assuming a simple 2D model
    vec1 = keypoints['top_lip'] - keypoints['left_mouth_corner']
    vec2 = keypoints['bottom_lip'] - keypoints['left_mouth_corner']
    angle_left = np.degrees(np.arctan2(np.linalg.norm(np.cross(vec1, vec2)), np.dot(vec1, vec2)))
    
    vec3 = keypoints['top_lip'] - keypoints['right_mouth_corner']
    vec4 = keypoints['bottom_lip'] - keypoints['right_mouth_corner']
    angle_right = np.degrees(np.arctan2(np.linalg.norm(np.cross(vec3, vec4)), np.dot(vec3, vec4)))
    
    return angle_left, angle_right

def classify_emotion(MAR, angle_left, angle_right, curvature_ratio):

    """

    Simple rule-based classification based on MAR, mouth corner angles, and curvature ratio.

    """

    # Emotion classification based on multiple factors: MAR, angles, and curvature ratio

    if MAR > 0.3 and angle_left < 160 and angle_right < 160 and curvature_ratio > 1:

        return 'Positive'

    elif MAR < 0.2 and angle_left > 170 and angle_right > 170 and curvature_ratio < 1:

        return 'Negative'

    else:

        return 'Neutral'


def get_emotion(output):
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

    keypoints['all_top_lip'] = np.array(all_up_lip)
    keypoints['all_bottom_lip'] = np.array(all_bottom_lip)

    MAR = calculate_MAR(keypoints)
    angle_left, angle_right = mouth_corner_angle(keypoints)
    curvature_bottom_lip = compute_curvature_for_lip(keypoints['all_bottom_lip'])
    curvature_top_lip = compute_curvature_for_lip(keypoints['all_top_lip'])
    curvature_ratio = curvature_bottom_lip / curvature_top_lip

    emotion = classify_emotion(MAR, angle_left, angle_right, curvature_ratio)

    return emotion


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
    

def draw_face_mask(frame, output):
    frame_orig = frame.copy()

    height, width, _ = frame.shape

    for i in range(output.shape[0]):
        if i <= 16:
            color = (0, 0, 255)
            color_2 = (0, 0, 100)

            if i < 16:
                cv2.line(frame, (int(output[i, 0] * width), int(output[i, 1] * height)),
                         (int(output[i + 1, 0] * width), int(output[i + 1, 1] * height)), color_2, 2)

        elif i <= 26:
            color = (0, 255, 255)
            color_2 = (0, 100, 100)

            if i < 21:
                cv2.line(frame, (int(output[i, 0] * width), int(output[i, 1] * height)),
                         (int(output[i + 1, 0] * width), int(output[i + 1, 1] * height)), color_2, 2)
            
            if i > 21 and i < 26:
                cv2.line(frame, (int(output[i, 0] * width), int(output[i, 1] * height)),
                         (int(output[i + 1, 0] * width), int(output[i + 1, 1] * height)), color_2, 2)

        elif i <= 35:
            color = (0, 255, 0)
            color_2 = (0, 100, 0)

            if i < 35:
                cv2.line(frame, (int(output[i, 0] * width), int(output[i, 1] * height)),
                         (int(output[i + 1, 0] * width), int(output[i + 1, 1] * height)), color_2, 2)
                
            if i == 30:
                cv2.line(frame, (int(output[35, 0] * width), int(output[35, 1] * height)),
                         (int(output[i, 0] * width), int(output[i, 1] * height)), color_2, 2)

        elif i <= 47:
            color = (255, 255, 0)
            color_2 = (100, 100, 0)

            if i < 41:
                cv2.line(frame, (int(output[i, 0] * width), int(output[i, 1] * height)),
                         (int(output[i + 1, 0] * width), int(output[i + 1, 1] * height)), color_2, 2)
            
            if i > 41 and i < 47:
                cv2.line(frame, (int(output[i, 0] * width), int(output[i, 1] * height)),
                         (int(output[i + 1, 0] * width), int(output[i + 1, 1] * height)), color_2, 2)
                
            if i == 36:
                cv2.line(frame, (int(output[41, 0] * width), int(output[41, 1] * height)),
                         (int(output[i, 0] * width), int(output[i, 1] * height)), color_2, 2)
                
            if i == 42:
                cv2.line(frame, (int(output[47, 0] * width), int(output[47, 1] * height)),
                         (int(output[i, 0] * width), int(output[i, 1] * height)), color_2, 2)

        else:
            color = (255, 0, 0)
            color_2 = (100, 0, 0)

            if i < 59:
                cv2.line(frame, (int(output[i, 0] * width), int(output[i, 1] * height)),
                         (int(output[i + 1, 0] * width), int(output[i + 1, 1] * height)), color_2, 2)
                
            if i == 48:
                cv2.line(frame, (int(output[59, 0] * width), int(output[59, 1] * height)),
                         (int(output[i, 0] * width), int(output[i, 1] * height)), color_2, 2)
                
            if i > 59 and i < 67:
                cv2.line(frame, (int(output[i, 0] * width), int(output[i, 1] * height)),
                         (int(output[i + 1, 0] * width), int(output[i + 1, 1] * height)), color_2, 2)
            
            if i == 60:
                cv2.line(frame, (int(output[67, 0] * width), int(output[67, 1] * height)),
                         (int(output[i, 0] * width), int(output[i, 1] * height)), color_2, 2)

        circle_size = (height + width) // 300
        cv2.circle(frame, (int(output[i, 0] * width), int(output[i, 1] * height)), circle_size, color, -1)

    frame = cv2.addWeighted(frame, 0.4, frame_orig, 0.6, 0)

    return frame
