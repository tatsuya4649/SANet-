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
parser.add_argument('-d',default='models/decoder_iter_485000_63.pth',help='Decoder model parameter file path',type=str)
parser.add_argument('-c',"--content_image",default='brad_pitt.jpg',help='content image path',type=str)
parser.add_argument('-s',"--style_image",default='in1.jpg',help='style image path',type=str)
parser.add_argument('-a',"--alpha",default=1.0,help='transform alpha',type=float)

args = parser.parse_args()
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print('device => "{}"'.format(device))

content_image_path = "images/{}".format(args.content_image)
style_image_path = "images/{}".format(args.style_image)
network = net.Net(args.v,args.d,args.sc,args.ss,args.t)
network = network.eval().to(device)
content_image = imread.imread(content_image_path).to(device)#BxCxHxW
style_image= imread.imread(style_image_path).to(device)

print('generate image now...')
network_output = network.generate_image(content=content_image,style=style_image,alpha=args.alpha)
print('generate image !!!')
fig = plt.figure(figsize=(30,10))
content_fig = plt.subplot(1,3,1)
style_fig = plt.subplot(1,3,2)
generated_fig = plt.subplot(1,3,3)
content_fig.imshow(content_image[0].permute(1,2,0).cpu().detach().numpy())
style_fig.imshow(style_image[0].permute(1,2,0).cpu().detach().numpy())
generated_fig.imshow(network_output[0].permute(1,2,0).cpu().detach().numpy())
plt.show()
