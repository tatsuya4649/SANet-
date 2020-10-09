
import torch
import cv2


_DEFAULT_SIZE = 512

def imresize(image):
    height = image.shape[0]
    width = image.shape[1]
    if height < width:
        image = cv2.resize(image,(int(_DEFAULT_SIZE),int(_DEFAULT_SIZE*(height/width))))
    else:
        image = cv2.resize(image,(int(_DEFAULT_SIZE*(width/height)),int(_DEFAULT_SIZE)))
    return image


def imread(path):
    image_numpy = cv2.imread(path)#HxBxC
    image_numpy = cv2.cvtColor(image_numpy,cv2.COLOR_BGR2RGB)
    image_numpy = imresize(image_numpy)
    image_tensor = torch.from_numpy(image_numpy).permute(2,0,1)#CxHxW
    image_tensor = image_tensor.float() / 255.
    return image_tensor.unsqueeze(0)
