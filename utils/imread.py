
import torch
import cv2

def imread(path):
    image_numpy = cv2.imread(path)#HxBxC
    image_tensor = torch.from_numpy(image_numpy).permute(2,0,1)#CxHxW
    return image_tensor.unsqueeze(0)
