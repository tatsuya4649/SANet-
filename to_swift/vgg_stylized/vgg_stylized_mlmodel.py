
import torch
import torch.nn as nn
import sys,os
sys.path.append('../..')
import net.stylized_feats as sf

class VGG_Stylized_Mlmodel(nn.Module):
	def __init__(self):
		super().__init__()
		self.vgg_stylized_content = sf.VGG_Stylized_Content()
		self.vgg_stylized_style = sf.VGG_Stylized_Style()
	def forward(self,content4,content5,style4,style5):
		new_content4 = self.vgg_stylized_content.vgg_stylized_conv_4(content4)
		new_content5 = self.vgg_stylized_content.vgg_stylized_conv_5(content5)
		new_style4 = self.vgg_stylized_style.vgg_stylized_conv_4_style(style4)
		new_style5 = self.vgg_stylized_style.vgg_stylized_conv_5_style(style5)
		return new_content4,new_content5,new_style4,new_style5
