import sys,os
sys.path.append(os.getcwd() + '/net')
sys.path.append(os.getcwd() + '/utils')
import torch
from net import network as net
import argparse
import imread

parser = argparse.ArgumentParser(description='test SANet++')
parser.add_argument('-sc',default='models/vgg_stylized_content_iter_485000_63.pth',help='vgg_stylized_content model parameter file path')
parser.add_argument('-ss',default='models/vgg_stylized_style_iter_485000_63.pth',help='vgg_stylized_style model parameter file path')
parser.add_argument('-t',default='models/sa_module_iter_485000_63.pth',help='Transform model parameter file path')
parser.add_argument('-v',default='models/vgg_normalised.pth',help='VGG19 model parameter file path')
parser.add_argument('-d',default='models/decoder_iter_485000_63.pth',help='Decoder model parameter file path')
parser.add_argument('-c',default='',help='content image path')
parser.add_argument('-s',default='',help='style image path')

args = parser.parse_args()

network = net.Net(args.v,args.d,args.sc,args.ss,args.t)
content_image = imread.imread(args.c)#BxCxHxW
style_image= imread.imread(args.s)

network_output = network(content=content_image,style=style_image)
print(network_output.shape)
