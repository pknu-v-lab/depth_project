from __future__ import absolute_import, division, print_function

import cv2
import os
import sys
import glob
import argparse
import numpy as np
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt

from layers import *


import networks
import torch
from torchvision import transforms, datasets

# load a Midas model for depth estimation
model_type = "DPT_Hybrid"  

dpt = torch.hub.load("intel-isl/Midas", model_type)

# Move model to GPU if available
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
dpt.to(device)
dpt.eval()

# load transforms to resize and normalize the image
midas_transforms = torch.hub.load("intel-isl/Midas", "transforms")

transform = midas_transforms.dpt_transform


#from layers import disp_to_depth

K = np.array([[0.58, 0, 0.5],
              [0, 1.92, 0.5],
                [0, 0, 1]])

inv_K = np.linalg.pinv(K)

"mono+stereo_640x192"


num_pose_frames = 2



loaded_dict_enc = torch.load("encoder.pth", map_location=device)
loaded_dict_pose_enc = torch.load("pose_encoder.pth", map_location=device)


encoder = networks.ResnetEncoder(18, False)
pose_encoder = networks.ResnetEncoder(18, False, 2)

filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
filtered_dict_pose_enc = {k: v for k, v in loaded_dict_pose_enc.items() if k in pose_encoder.state_dict()}

encoder.load_state_dict(filtered_dict_enc)
encoder.to(device)
encoder.eval()



pose_encoder.load_state_dict(filtered_dict_pose_enc)
pose_encoder.to(device)
pose_encoder.eval()

pose_decoder = networks.PoseDecoder(pose_encoder.num_ch_enc, num_input_features=1, num_frames_to_predict_for=2)
pose_decoder.to(device)
pose_decoder.eval()
    
feed_height = loaded_dict_enc['height']
feed_width = loaded_dict_enc['width']


img1 = pil.open("10.jpg").convert('RGB')
img2 = pil.open("11.jpg").convert('RGB')
img3 = pil.open("12.jpg").convert('RGB')

origin_img2 = img2

original_width, original_height = img1.size
pose_img1 = img1.resize((feed_width, feed_height), pil.LANCZOS)
pose_img2 = img2.resize((feed_width, feed_height), pil.LANCZOS)
pose_img3 = img3.resize((feed_width, feed_height), pil.LANCZOS)


pose_img1 = transforms.ToTensor()(pose_img1).unsqueeze(0)
pose_img2 = transforms.ToTensor()(pose_img2).unsqueeze(0)
pose_img3 = transforms.ToTensor()(pose_img3).unsqueeze(0)

pose_img1 = pose_img1.to(device)
pose_img2 = pose_img2.to(device)
pose_img3 = pose_img3.to(device)



pose_feats = {-1 : pose_img1, 0: pose_img2, 1: pose_img3}


img2 = transform(np.array(img2)).to(device)

with torch.no_grad():
    pose_inputs = [pose_feats[0], pose_feats[1]]
    pose_inputs = [pose_encoder(torch.cat(pose_inputs, 1))]

    axisangle, translation = pose_decoder(pose_inputs)
    T = transformation_from_parameters(axisangle[:, 1], translation[:, 1])
    axisangle = rot_from_axisangle(axisangle[:, 1])[:, :3, :3].cpu().numpy()
    translation = get_translation_matrix(translation[:, 1])[:, :3, -1].cpu().numpy()

    prediction = dpt(img2)


    prediction = torch.nn.functional.interpolate(
        prediction.unsqueeze(1),
        size=(original_height,original_width),
        mode="bicubic",
        align_corners=False,
    ).squeeze()

output = prediction.cpu().numpy()
output = cv2.normalize(output, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_64F)

img2 = torch.nn.functional.interpolate(
                img2, (original_height, original_width), mode="bilinear", align_corners=False)

img2 = img2.cpu().numpy()


homo_size = (3, 3, original_height, original_width)
homo =  np.zeros(homo_size)

for i in range(original_height):
    for j in range(original_width):
        homo[i][j] = K * (axisangle + (translation / output[i][j]) * np.array([0, 0, 1])) * inv_K

result = np.einsum('...ij,...j', homo, img2)
result = result.astype(np.uint8)
save_result = pil.fromarray(result)

file_path = 'result1.jpg'  # Replace with the desired file path
save_result.save(file_path)


plt.imshow(output)
























def read_calib_file(path):
    """Read KITTI calibration file
    (from https://github.com/hunse/kitti)
    """
    float_chars = set("0123456789.e+- ")
    data = {}
    with open(path, 'r') as f:
        for line in f.readlines():
            key, value = line.split(':', 1)
            value = value.strip()
            data[key] = value
            if float_chars.issuperset(value):
                # try to cast to float array
                try:
                    data[key] = np.array(list(map(float, value.split(' '))))
                except ValueError:
                    # casting error: data[key] already eq. value, so pass
                    pass

    return data

# Load calibration data from calib_cam_to_cam.txt
calib_data = read_calib_file("./kitti_data/2011_10_03/calib_cam_to_cam.txt")


# Extract the left-to-right (and right-to-left) camera transformation matrices
R_left= calib_data["R_02"].reshape(3, 3)  # Rotation matrix from left to right camera
T_left = calib_data["T_02"].reshape((-1, 1))  # Translation vector from left to right camera






