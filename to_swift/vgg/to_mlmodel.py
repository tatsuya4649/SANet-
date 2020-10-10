"""
making vgg mlmodel (.py -> .mlmodel)
"""

import torch
import vgg_mlmodel as vm
import coremltools
import sys
sys.path.append('..')
import reflection_pad

vgg_mlmodel = vm.VGG_Mlmodel()
vgg_mlmodel.eval()
ex_content = torch.rand(1,3,512,512)
ex_style = torch.rand(1,3,512,512)
trace_model = torch.jit.trace(vgg_mlmodel,(ex_content,ex_style))
_SCALE = 1.0/255.0
model = coremltools.convert(
	trace_model,
	source = "pytorch",
	inputs = [coremltools.ImageType(name="content",shape=ex_content.shape,scale=_SCALE),coremltools.ImageType(name="style",shape=ex_style.shape,scale=_SCALE)]
)
model.save("../../mlmodels/sanet_vgg_512.mlmodel")
