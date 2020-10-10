"""
making vgg mlmodel (.py -> .mlmodel)
"""

import torch
import torch.nn as nn
import sys,os
sys.path.append("../..")
import net.vgg as vgg

_DEFAULT_VGG_PATH = "../../models/vgg_normalised.pth"
model = vgg.VGG19(_DEFAULT_VGG_PATH)
model.eval()

class VGG_Mlmodel(nn.Module):
	def __init__(self):
		super().__init__()
		self.model = model
	def forward(self,content,style):
		content4,content5 = self.model(content)
		style4,style5 = self.model(style)
		return content4,content5,style4,style5

