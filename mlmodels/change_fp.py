"""
*.mlmodel FP64 => FP16
"""
import coremltools
import argparse

parser = argparse.ArgumentParser(description='to change "FP" 64 => 16')
parser.add_argument('-f','--file',help='fp64.mlmodel')
args = parser.parse_args()

_MLMODEL_PATH = "64/{}".format(args.file)
_NAME = "{}_16.mlmodel".format(args.file.split(".")[0])
_MLMODEL_16_PATH = "16/{}".format(_NAME)
model_64 = coremltools.utils.load_spec(_MLMODEL_PATH)
model_16 = coremltools.utils.convert_neural_network_spec_weights_to_fp16(model_64)
coremltools.utils.save_spec(model_16,_MLMODEL_16_PATH)
