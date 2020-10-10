
import torch
import torch.nn as nn
import sys,os
sys.path.append("../..")
import net.sanet as sanet

_DEFAULT_PATH = "../../models/sa_module_iter_485000_63.pth"
transform = sanet.Transform(512,512)
transform.load_state_dict(torch.load(_DEFAULT_PATH))
transform.eval()

class Transform_Mlmodel(nn.Module):
	def __init__(self):
		super().__init__()
		self.transform = transform
	def forward(self,content4,style4,content5,style5,alpha):
		self.alpha = alpha
		transform_output_cc = self.transform(content4,content4,content5,content5)
		transform_output_cs = self.transform(content4,style4,content5,style5)
		transform_output = self.alpha * transform_output_cs + (1-self.alpha) * transform_output_cc
		return transform_output
