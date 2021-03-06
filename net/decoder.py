"""

decoder is to chage transfered feature => generated image

"""
import torch
import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self,parameters_path=None):
        super().__init__()
        self.decoder = self.making_decoder()
        if parameters_path is not None:
            self.decoder.load_state_dict(torch.load(parameters_path))
    def making_decoder(self):
        decoder = nn.Sequential(
                nn.ReflectionPad2d((1,1,1,1)),
                nn.Conv2d(512,256,(3,3)),
                nn.ReLU(),
                nn.Upsample(scale_factor=2,mode='nearest'),
                nn.ReflectionPad2d((1,1,1,1)),
                nn.Conv2d(256,256,(3,3)),
                nn.ReLU(),
                nn.ReflectionPad2d((1,1,1,1)),
                nn.Conv2d(256,256,(3,3)),
                nn.ReLU(),
                nn.ReflectionPad2d((1,1,1,1)),
                nn.Conv2d(256,256,(3,3)),
                nn.ReLU(),
                nn.ReflectionPad2d((1,1,1,1)),
                nn.Conv2d(256,128,(3,3)),
                nn.ReLU(),
                nn.Upsample(scale_factor=2,mode='nearest'),
                nn.ReflectionPad2d((1,1,1,1)),
                nn.Conv2d(128,128,(3,3)),
                nn.ReLU(),
                nn.ReflectionPad2d((1,1,1,1)),
                nn.Conv2d(128,64,(3,3)),
                nn.ReLU(),
                nn.Upsample(scale_factor=2,mode='nearest'),
                nn.ReflectionPad2d((1,1,1,1)),
                nn.Conv2d(64,64,(3,3)),
                nn.ReLU(),
                nn.ReflectionPad2d((1,1,1,1)),
                nn.Conv2d(64,32,(3,3)),
                nn.Upsample(scale_factor=2,mode='nearest'),
                nn.ReflectionPad2d((1,1,1,1)),
                nn.Conv2d(32,32,(3,3)),
                nn.ReLU(),
                nn.ReflectionPad2d((1,1,1,1)),
                nn.Conv2d(32,16,(3,3)),
                nn.Upsample(scale_factor=2,mode='nearest'),
                nn.ReflectionPad2d((1,1,1,1)),
                nn.Conv2d(16,16,(3,3)),
                nn.ReLU(),
                nn.ReflectionPad2d((1,1,1,1)),
                nn.Conv2d(16,3,(3,3)),
        )
        return decoder
    def forward(self,input):
        return self.decoder(input)


if __name__ == "__main__":
    print("Hello,{}".format(__file__))
    _DEFAULT_PATH = '../models/decoder_iter_485000_63.pth'
    decoder = Decoder(_DEFAULT_PATH)
    rand = torch.rand(1,512,16,16)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print('--------------Decoder test----------------')
    print('device => {}'.format(device))
    print('Decoder parameter file path => {}'.format(_DEFAULT_PATH))
    decoder_output = decoder(rand)
    print("input.shape => {}".format(rand.shape))
    print("decoder_output.shape => {}".format(decoder_output.shape))
    print('------------------------------------------')
    print('clear!!!')
