import torch

from matplotlib import pyplot as plt
import numpy as np
import sklearn
from scipy.interpolate import interp1d
from scipy.integrate import quad
import cv2

dictionary_of_templates={
    "positive":[cv2.imread('templates/Rikhat/Positive/frame0.jpg',0)[600:700, 300:560],
                cv2.imread('templates/Rikhat/Positive/frame421.jpg',0)[660:770, 310:570],
                cv2.imread('templates/Rikhat/Positive/frame98.jpg',0)[600:700, 300:560],
                cv2.imread('templates/Rikhat/Positive/frame210.jpg',0)[600:700, 220:430],
                cv2.imread('templates/Rikhat/Positive/frame255.jpg',0)[630:720, 470:700],
                # Nuren
                cv2.imread('templates/Nuren/Positive/frame0.jpg',0)[600:680, 320:540],
                cv2.imread('templates/Nuren/Positive/frame37.jpg',0)[600:690, 320:540],
                cv2.imread('templates/Nuren/Positive/frame176.jpg',0)[580:670, 490:680],
                cv2.imread('templates/Nuren/Positive/frame248.jpg',0)[580:680, 160:350],
                cv2.imread('templates/Nuren/Positive/frame338.jpg',0)[670:770, 290:510],
                cv2.imread('templates/Nuren/Positive/frame451.jpg',0)[570:670, 310:530],
                cv2.imread('templates/Nuren/Positive/frame551.jpg',0)[570:660, 280:510]

                ],

                
    "negative":[cv2.imread('templates/Rikhat/Negative/frame0.jpg',0)[450:520, 320:520],
                cv2.imread('templates/Rikhat/Negative/frame67.jpg',0)[440:510, 320:520],
                cv2.imread('templates/Rikhat/Negative/frame135.jpg',0)[380:460, 330:520],
                cv2.imread('templates/Rikhat/Negative/frame163.jpg',0)[450:530, 560:685],
                cv2.imread('templates/Rikhat/Negative/frame210.jpg',0)[520:590, 190:340],
                cv2.imread('templates/Rikhat/Negative/frame525.jpg',0)[550:640, 320:540],
                # Nuren
                cv2.imread('templates/Nuren/Negative/frame0.jpg',0)[540:610, 330:480],
                cv2.imread('templates/Nuren/Negative/frame34.jpg',0)[550:620, 320:490],
                cv2.imread('templates/Nuren/Negative/frame73.jpg',0)[560:620, 430:600],
                cv2.imread('templates/Nuren/Negative/frame121.jpg',0)[560:630, 240:420],
                cv2.imread('templates/Nuren/Negative/frame210.jpg',0)[550:620, 420:590],
                cv2.imread('templates/Nuren/Negative/frame342.jpg',0)[570:640, 240:380],
                cv2.imread('templates/Nuren/Negative/frame364.jpg',0)[580:640, 280:440],
                cv2.imread('templates/Nuren/Negative/frame384.jpg',0)[580:640, 310:490]
                ]
}

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


def fit_quadratic_curve(keypoints):
    x = keypoints[:, 0]
    y = keypoints[:, 1]
    coefficients = np.polyfit(x, y, 2)  # Fit a second-degree polynomial
    return coefficients

def curvature_of_quadratic(coefficients, x):
    a, b, _ = coefficients
    curvature = np.abs(2 * a) / (1 + (2 * a * x + b) ** 2) ** 1.5
    return curvature

def compute_curvature_for_lip(keypoints):
    coefficients = fit_quadratic_curve(keypoints)
    x_values = np.linspace(keypoints[:, 0].min(), keypoints[:, 0].max(), 100)
    curvatures = [curvature_of_quadratic(coefficients, x) for x in x_values]
    return np.mean(curvatures)

def calculate_MAR(keypoints):
    width = np.linalg.norm(keypoints['left_mouth_corner'] - keypoints['right_mouth_corner'])
    height = np.linalg.norm(keypoints['top_lip'] - keypoints['bottom_lip'])
    MAR = height / width
    return MAR

def mouth_corner_angle(keypoints):
    vec1 = keypoints['top_lip'] - keypoints['left_mouth_corner']
    vec2 = keypoints['bottom_lip'] - keypoints['left_mouth_corner']
    angle_left = np.degrees(np.arctan2(np.linalg.norm(np.cross(vec1, vec2)), np.dot(vec1, vec2)))
    
    vec3 = keypoints['top_lip'] - keypoints['right_mouth_corner']
    vec4 = keypoints['bottom_lip'] - keypoints['right_mouth_corner']
    angle_right = np.degrees(np.arctan2(np.linalg.norm(np.cross(vec3, vec4)), np.dot(vec3, vec4)))

    angle_up = np.degrees(np.arctan2(np.linalg.norm(np.cross(vec1, vec3)), np.dot(vec1, vec3)))
    angle_bottom = np.degrees(np.arctan2(np.linalg.norm(np.cross(vec2, vec4)), np.dot(vec2, vec4)))
    
    return angle_left, angle_right, angle_up, angle_bottom

def classify_emotion_with_rule_based(MAR, angle_left, angle_right, angle_up, angle_bottom, curvature_ratio):
    combined_feature = curvature_ratio * ((angle_left) / (angle_bottom))
    if combined_feature > 0.43:
        return 'positive'
    return 'negative'

def classifying_with_template_matching(cropped_mouth, templates_dictionary):
  method = eval('cv2.TM_CCOEFF_NORMED')
  positive_emotion_response=[]
  negative_emotion_response=[]
  for template in templates_dictionary["positive"]:
    res = cv2.matchTemplate(cropped_mouth,template,method)
    _, max_val, _, _ = cv2.minMaxLoc(res)
    positive_emotion_response.append(max_val)

  for template in templates_dictionary["negative"]:
    res = cv2.matchTemplate(cropped_mouth,template,method)
    _, max_val, _, _ = cv2.minMaxLoc(res)
    negative_emotion_response.append(max_val)
  
  predicted_emotion= "positive" if max(positive_emotion_response)>\
    max(negative_emotion_response) else "negative"
  print("predicted emotion is ", predicted_emotion)
  return predicted_emotion

def get_emotion(output, frame_orig, use_template_matching, width, height):
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
        elif i == 54:
            keypoints['right_mouth_corner'] = np.array([output[i, 0], output[i, 1]])
        elif i == 57:
            keypoints['bottom_lip'] = np.array([output[i, 0], output[i, 1]])

        if i >= 61 and i <= 63:
            all_up_lip.append([output[i, 0], output[i, 1]])
        elif i >= 65 and i <= 67:
            all_bottom_lip.append([output[i, 0], output[i, 1]])

    keypoints['all_top_lip'] = np.array(all_up_lip)
    keypoints['all_bottom_lip'] = np.array(all_bottom_lip)

    MAR = calculate_MAR(keypoints)
    angle_left, angle_right, angle_up, angle_bottom = mouth_corner_angle(keypoints)
    curvature_bottom_lip = compute_curvature_for_lip(keypoints['all_bottom_lip'])
    curvature_top_lip = compute_curvature_for_lip(keypoints['all_top_lip'])
    curvature_ratio = curvature_bottom_lip / curvature_top_lip

    #create a dictionary with all the values
    results = {
        'MAR': str(MAR),
        'angle_left': str(angle_left),
        'angle_right': str(angle_right),
        'angle_up': str(angle_up),
        'angle_bottom': str(angle_bottom),
        'curvature_ratio': str(curvature_ratio),
    }

    if use_template_matching:
        cropped_mouth = cv2.cvtColor(frame_orig, cv2.COLOR_BGR2GRAY)
        cropped_mouth=cropped_mouth[int(keypoints['left_mouth_corner'][1]*height-
                                        0.1*height):
                                    int(keypoints['left_mouth_corner'][1]*height+
                                        0.1*height), 
                                    int(keypoints['left_mouth_corner'][0]*width-
                                        0.05*width):
                                    int(keypoints['right_mouth_corner'][0]*width+
                                        0.1*width)]
        
        predicted_emotion=classifying_with_template_matching(cropped_mouth, 
                        {emotion:[cv2.resize(template, 
                        (cropped_mouth.shape[1]-10, cropped_mouth.shape[0]-10))
                        for template in template_list] for emotion, template_list
                        in dictionary_of_templates.items()})
        
        emotion=predicted_emotion
    else:
        emotion = classify_emotion_with_rule_based(MAR, angle_left, angle_right, angle_up, angle_bottom, curvature_ratio)

    return emotion, results


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

    colors_map = {
        "jaw": (255, 0, 0),
        "jaw_line": (200, 200, 200),
        "eyebrow": (255, 255, 0),
        "eyebrow_line": (200, 200, 200),
        "nose": (0, 255, 0),
        "nose_line": (200, 200, 200),
        "eye": (0, 255, 255),
        "eye_line": (200, 200, 200),
        "mouth": (0, 0, 255),
        "mouth_line": (200, 200, 200),
    }

    for i in range(output.shape[0]):
        if i <= 16: # jaw
            color = colors_map["jaw"]
            color_2 = colors_map["jaw_line"]

            if i < 16:
                cv2.line(frame, (int(output[i, 0] * width), int(output[i, 1] * height)),
                         (int(output[i + 1, 0] * width), int(output[i + 1, 1] * height)), color_2, 2)

        elif i <= 26:
            color = colors_map["eyebrow"]
            color_2 = colors_map["eyebrow_line"]

            if i < 21:
                cv2.line(frame, (int(output[i, 0] * width), int(output[i, 1] * height)),
                         (int(output[i + 1, 0] * width), int(output[i + 1, 1] * height)), color_2, 2)
            
            if i > 21 and i < 26:
                cv2.line(frame, (int(output[i, 0] * width), int(output[i, 1] * height)),
                         (int(output[i + 1, 0] * width), int(output[i + 1, 1] * height)), color_2, 2)

        elif i <= 35:
            color = colors_map["nose"]
            color_2 = colors_map["nose_line"]

            if i < 35:
                cv2.line(frame, (int(output[i, 0] * width), int(output[i, 1] * height)),
                         (int(output[i + 1, 0] * width), int(output[i + 1, 1] * height)), color_2, 2)
                
            if i == 30:
                cv2.line(frame, (int(output[35, 0] * width), int(output[35, 1] * height)),
                         (int(output[i, 0] * width), int(output[i, 1] * height)), color_2, 2)

        elif i <= 47:
            color = colors_map["eye"]
            color_2 = colors_map["eye_line"]

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
            color = colors_map['mouth']
            color_2 = colors_map["mouth_line"]

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
