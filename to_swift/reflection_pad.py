import coremltools
from coremltools.converters.mil import register_torch_op
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.frontend.torch.ops import _get_inputs

@register_torch_op
def reflection_pad2d(context,node):
	inputs = _get_inputs(context,node)
	output = mb.pad(x=inputs[0],pad=[1,1,1,1],mode="reflect")
	context.add(output,node.name)
