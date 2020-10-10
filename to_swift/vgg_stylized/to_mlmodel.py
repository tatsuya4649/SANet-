import torch
import coremltools
import vgg_stylized_mlmodel as vs

vs_mlmodel = vs.VGG_Stylized_Mlmodel()
_CONTENT_PARAMETER_PATH = "../../models/vgg_stylized_content_iter_485000_63.pth"
_STYLE_PARAMETER_PATH = "../../models/vgg_stylized_style_iter_485000_63.pth"
vs_mlmodel.vgg_stylized_content.load_state_dict(torch.load(_CONTENT_PARAMETER_PATH))
vs_mlmodel.vgg_stylized_style.load_state_dict(torch.load(_STYLE_PARAMETER_PATH))
vs_mlmodel.eval()

ex_content4 = torch.rand(1,512,64,64)
ex_content5 = torch.rand(1,512,32,32)
ex_style4 = torch.rand(1,512,64,64)
ex_style5 = torch.rand(1,512,32,32)

trace_model = torch.jit.trace(vs_mlmodel,(ex_content4,ex_style4,ex_content5,ex_style5))
model = coremltools.convert(
	trace_model,
	source = "pytorch",
	inputs = [	
			coremltools.TensorType(name="content_4",shape=ex_content4.shape),
			coremltools.TensorType(name="style_4",shape=ex_style4.shape),
			coremltools.TensorType(name="content_5",shape=ex_content5.shape),
			coremltools.TensorType(name="style_5",shape=ex_style5.shape)
		]
)
model.save("../../mlmodels/64/sanet_vgg_stylized_512.mlmodel")	
