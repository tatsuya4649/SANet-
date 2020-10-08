import torch
from net import network
import argparse
import utils.imread

parser = argparse.ArgumentParser(description='test SANet++')
parser.add_argument('-sc',default='moedls/vgg_stylized_content_iter_485000_63.pth',help='vgg_stylized_content model parameter file path')
parser.add_argument('-ss',default='moedls/vgg_stylized_content_iter_485000_63.pth',help='vgg_stylized_content model parameter file path')
parser.add_argument('-t',default='moedls/sa_module_iter_485000_63.pth',help='Transform model parameter file path')
parser.add_argument('-c',default='',help='content image path')
parser.add_argument('-s',default='',help='style image path')

args = parser.parse_args()

network = net.Net(args.sc,args.ss,args.t)
content_image = imread.imread(args.c)#BxCxHxW
style_image= imread.imread(args.s)

network_output = network(content=content_image,style=style_image)
print(network_output.shape)
