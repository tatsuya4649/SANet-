import torch
import coremltools
import sys
sys.path.append('..')
import reflection_pad
import decoder_mlmodel as dm

decoder_mlmodel = dm.Decoder_Mlmodel()
decoder_mlmodel.eval()
ex_input = torch.rand(1,512,64,64)
trace_model = torch.jit.trace(decoder_mlmodel,(ex_input))
model = coremltools.convert(
	trace_model,
	source = "pytorch",
	inputs = [coremltools.TensorType(name="sanet",shape=ex_input.shape)]
)
model.save("../../mlmodels/64/sanet_decoder_512.mlmodel")
