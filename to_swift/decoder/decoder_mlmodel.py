

import torch
import torch.nn as nn
import sys,os
sys.path.append("../..")
import net.decoder as decoder

_DEFAULT_PATH = "../../models/decoder_iter_485000_63.pth"
decoder = decoder.Decoder(_DEFAULT_PATH)

class Decoder_Mlmodel(nn.Module):
	def __init__(self):
		super().__init__()
		self.decoder = decoder
	def forward(self,input):
		decoder_output = self.decoder(input)
		return decoder_output
