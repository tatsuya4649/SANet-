import sys,os
sys.path.append(os.getcwd() + '/net')
sys.path.append(os.getcwd() + '/utils')
import torch
from net import network as net
import argparse
import imread
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser(description='test SANet++')
parser.add_argument('-sc',default='models/vgg_stylized_content_iter_485000_63.pth',help='vgg_stylized_content model parameter file path')
parser.add_argument('-ss',default='models/vgg_stylized_style_iter_485000_63.pth',help='vgg_stylized_style model parameter file path')
parser.add_argument('-t',default='models/sa_module_iter_485000_63.pth',help='Transform model parameter file path')
parser.add_argument('-v',default='models/vgg_normalised.pth',help='VGG19 model parameter file path')
parser.add_argument('-d',default='models/decoder_iter_485000_63.pth',help='Decoder model parameter file path')
parser.add_argument('-c',default='images/brad_pitt.jpg',help='content image path')
parser.add_argument('-s',default='images/goph.jpg',help='style image path')

args = parser.parse_args()
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print('device => "{}"'.format(device))

network = net.Net(args.v,args.d,args.sc,args.ss,args.t)
network = network.eval().to(device)
content_image = imread.imread(args.c).to(device)#BxCxHxW
style_image= imread.imread(args.s).to(device)

print('generate image now...')
network_output = network.generate_image(content=content_image,style=style_image)
print('generate image !!!')
fig = plt.figure(figsize=(30,10))
content_fig = plt.subplot(1,3,1)
style_fig = plt.subplot(1,3,2)
generated_fig = plt.subplot(1,3,3)
content_fig.imshow(content_image[0].permute(1,2,0).cpu().detach().numpy())
style_fig.imshow(style_image[0].permute(1,2,0).cpu().detach().numpy())
generated_fig.imshow(network_output[0].permute(1,2,0).cpu().detach().numpy())
plt.show()
print(network_output)
print(network_output[0].permute(1,2,0).cpu().detach().numpy().shape)
