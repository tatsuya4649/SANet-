
import coremltools
import torch
import transform_mlmodel as tm
import sys,os
sys.path.append('..')
import reflection_pad

transform_mlmodel = tm.Transform_Mlmodel()
transform_mlmodel.eval()

ex_content4 = torch.rand(1,512,64,64)
ex_content5 = torch.rand(1,512,32,32)
ex_style4 = torch.rand(1,512,64,64)
ex_style5 = torch.rand(1,512,32,32)
ex_alpha = torch.rand(1)

trace_model = torch.jit.trace(transform_mlmodel,(ex_content4,ex_style4,ex_content5,ex_style5,ex_alpha))
model = coremltools.convert(
	trace_model,
	source="pytorch",
	inputs=[
		coremltools.TensorType(name="content_4",shape=ex_content4.shape),
		coremltools.TensorType(name="style_4",shape=ex_style4.shape),
		coremltools.TensorType(name="content_5",shape=ex_content5.shape),
		coremltools.TensorType(name="style_5",shape=ex_style5.shape),
		coremltools.TensorType(name="alpha",shape=ex_alpha.shape)
	])
model.save("../../mlmodels/sanet_transform_512.mlmodel")
