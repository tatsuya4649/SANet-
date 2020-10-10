"""

vgg file is to make VGG19 Feature extractor.

why extractor is VGG19?
 ==> https://reiinakano.com/2019/06/21/robust-neural-style-transfer.html

"""

import torch
import torch.nn as nn

class VGG19(nn.Module):
    def __init__(self,parameters_path=None):
        super().__init__()
        self.vgg = self.making_vgg()
        if parameters_path is not None:
            self.vgg.load_state_dict(torch.load(parameters_path)) 
        enc_layers = list(self.vgg.children())
        self.enc4 = nn.Sequential(*enc_layers[:31])
        self.enc5 = nn.Sequential(*enc_layers[31:44])
        for name in ['enc4','enc5']:
            for param in getattr(self,name).parameters():
                param.requires_grad = False

    def making_vgg(self):
        vgg = nn.Sequential(
                nn.Conv2d(3,3,(1,1)),
                nn.ReflectionPad2d((1,1,1,1)),
                nn.Conv2d(3,64,(3,3)),
                nn.ReLU(),
                nn.ReflectionPad2d((1,1,1,1)),
                nn.Conv2d(64,64,(3,3)),
                nn.ReLU(), #relu1-2
                nn.MaxPool2d((2,2),(2,2),(0,0),ceil_mode=True),
                nn.ReflectionPad2d((1,1,1,1)),
                nn.Conv2d(64,128,(3,3)),
                nn.ReLU(), #relu2-1
                nn.ReflectionPad2d((1,1,1,1)),
                nn.Conv2d(128,128,(3,3)),
                nn.ReLU(),#relu2-2
                nn.MaxPool2d((2,2),(2,2),(0,0),ceil_mode=True),
                nn.ReflectionPad2d((1,1,1,1)),
                nn.Conv2d(128,256,(3,3)),
                nn.ReLU(),#relu3-1
                nn.ReflectionPad2d((1,1,1,1)),
                nn.Conv2d(256,256,(3,3)),
                nn.ReLU(),#relu3-2
                nn.ReflectionPad2d((1,1,1,1)),
                nn.Conv2d(256,256,(3,3)),
                nn.ReLU(),#relu3-3
                nn.ReflectionPad2d((1,1,1,1)),
                nn.Conv2d(256,256,(3,3)),
                nn.ReLU(),#relu3-4
                nn.MaxPool2d((2,2),(2,2),(0,0),ceil_mode=True),
                nn.ReflectionPad2d((1,1,1,1)),
                nn.Conv2d(256,512,(3,3)),
                nn.ReLU(),#relu4-1
                nn.ReflectionPad2d((1,1,1,1)),
                nn.Conv2d(512,512,(3,3)),
                nn.ReLU(),#relu4-2
                nn.ReflectionPad2d((1,1,1,1)),
                nn.Conv2d(512,512,(3,3)),
                nn.ReLU(),#relu4-3
                nn.ReflectionPad2d((1,1,1,1)),
                nn.Conv2d(512,512,(3,3)),
                nn.ReLU(),#relu4-4
                nn.MaxPool2d((2,2),(2,2),(0,0),ceil_mode=True),
                nn.ReflectionPad2d((1,1,1,1)),
                nn.Conv2d(512,512,(3,3)),
                nn.ReLU(),#relu5-1
                nn.ReflectionPad2d((1,1,1,1)),
                nn.Conv2d(512,512,(3,3)),
                nn.ReLU(),#relu5-2
                nn.ReflectionPad2d((1,1,1,1)),
                nn.Conv2d(512,512,(3,3)),
                nn.ReLU(),#relu5-3
                nn.ReflectionPad2d((1,1,1,1)),
                nn.Conv2d(512,512,(3,3)),
                nn.ReLU() #relu5-4
        )
        return vgg

    def forward(self,input):
        enc4_output = self.enc4(input)
        enc5_output = self.enc5(enc4_output)
        return enc4_output,enc5_output

_DEFAULT_PATH = '../models/vgg_normalised.pth'
if __name__ == "__main__":
    print("Hello,{}".format(__file__))
    encoder = VGG19(_DEFAULT_PATH)
    rand = torch.rand(1,3,512,512)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print("-----------------VGG19 test-------------------")
    print('device => {}'.format(device))
    print('VGG19 parameter file path => {}'.format(_DEFAULT_PATH))
    print("input shape => {}".format(rand.shape))
    enc4_output,enc5_output = encoder(rand)
    print("enc4_output.shape => {}".format(enc4_output.shape))
    print("enc5_output.shape => {}".format(enc5_output.shape))
    print('-------------------------------------------------')
    print('clear!!!')
