"""
Net is a model with 3step Architectuer(extractor,transfer,generator).
"""

import torch
import torch.nn as nn
from decoder import Decoder
from vgg import VGG19
from stylized_feats import VGG_Stylized_Content,VGG_Stylized_Style
from sanet import Transform
import sys
sys.path.append('../utils/')
import loss_func as lf

class Net(nn.Module):
    def __init__(self,vgg_path=None,decoder_path=None,stylized_content_path=None,stylized_style_path=None,transform_path=None):
        super().__init__()
        self.vgg_stylized_content = VGG_Stylized_Content()
        self.vgg_stylized_style = VGG_Stylized_Style()
        if stylized_content_path is not None:
            self.vgg_stylized_content.load_state_dict(torch.load(stylized_content_path))
        if stylized_style_path is not None:
            self.vgg_stylized_style.load_state_dict(torch.load(stylized_style_path))
        self.transform = Transform(512,512)
        if transform_path is not None:
            self.transform.load_state_dict(torch.load(transform_path))
        self.encoder = VGG19()
        if vgg_path is not None:
            self.encoder.vgg.load_state_dict(torch.load(vgg_path))

        self.decoder = Decoder()
        if decoder_path is not None:
            self.decoder.decoder.load_state_dict(torch.load(decoder_path))

        self.mse_loss = nn.MSELoss()

        enc_layers = list(self.encoder.vgg.children())
        self.enc_1 = nn.Sequential(*enc_layers[:4])
        self.enc_2 = nn.Sequential(*enc_layers[4:11])
        self.enc_3 = nn.Sequential(*enc_layers[11:18])
        self.enc_4 = nn.Sequential(*enc_layers[18:31])
        self.enc_5 = nn.Sequential(*enc_layers[31:44])
        for name in ['enc_1','enc_2','enc_3','enc_4','enc_5']:
            for param in getattr(self,name).parameters():
                param.requiers_grad = False

    def encode_with_intermediate(self,input):
        results = [input]
        for i in range(5):
            func = getattr(self,'enc_{}'.format(i+1))
            results.append(func(results[-1]))
        return results[1:]

    def calc_content_loss(self,input,target):
        assert(input.size() == target.size())
        return self.mse_loss(input,target)

    def calc_style_loss(self,input,target):
        assert(input.size() == target.size())
        input_mean,input_std = calc_mean_std(input)
        target_mean,target_std = calc_mean_std(target)
        return self.mse_loss(input_mean,target_mean) + self.mse_loss(input_std,target_std)
    def small_images(self,images):
        return torch.nn.functional.interpolate(images,scale_factor=0.25,mode='bicubic')

    def generate_image(self,content,style,alpha=1.0):
        content_feats_4,content_feats_5 = self.encoder(content)
        style_feats_4,style_feats_5 = self.encoder(style)
        new_content_feats_4,new_content_feats_5 = self.vgg_stylized_content([content_feats_4,content_feats_5])
        new_style_feats_4,new_style_feats_5 = self.vgg_stylized_style([style_feats_4,style_feats_5])
        transform_output_cs = self.transform(new_content_feats_4,new_style_feats_4,new_content_feats_5,new_style_feats_5)
        transform_output_cc = self.transform(new_content_feats_4,new_content_feats_4,new_content_feats_5,new_content_feats_5)
        transform_output = alpha * transform_output_cs + (1-alpha) * transform_output_cc
        decoder_output = self.decoder(transform_output)
        return decoder_output

    def forward(self,content,style):
        content_feats_4,content_feats_5 = self.encoder(content)
        style_feats_4,style_feats_5 = self.encoder(style)
        new_content_feats_4,new_content_feats_5 = self.vgg_stylized_content([content_feats_4,content_feats_5])
        new_style_feats_4,new_style_feats_5 = self.vgg_stylized_style([style_feats_4,style_feats_5])
        transform_output = self.transform(new_content_feats_4,new_style_feats_4,new_content_feats_5,new_style_feats_5)
        decoder_output = self.decoder(transform_output)
        Ics = self.small_images(decoder_output)
        Ics_feats = self.encode_with_intermediate(Ics)
        content_feats = self.encode_with_intermediate(content)
        loss_c = self.calc_content_loss(lf.normal(Ics_feats[0]),lf.normal(content_feats[0])) + self.calc_content_loss(lf.normal(Ics_feats[1]),lf.normal(content_feats[1])) + self.calc_content_loss(lf.normal(Ics_feats[2]),lf.normal(content_feats[2])) + self.calc_content_loss(lf.normal(Ics_feats[3]),lf.normal(content_feats[3])) + self.calc_content_loss(lf.normal(Ics_feats[4]),lf.normal(content_feats[4]))
        loss_s = self.calc_style_loss(Ics_feats[0],style_feats[0]) + self.calc_style_loss(Ics_feats[1],style_feats[1]) + self.calc_style_loss(Ics_feats[2],style_feats[2]) + self.calc_style_loss(Ics_feats[3],style_feats[3]) + self.calc_style_loss(Ics_feats[4],style_feats[4])

        Icc = self.decoder(self.transform(content_feats[-2],content_feats[-2],content_feats[-1],content_feats[-1]))
        Iss = self.decoder(self.transform(style_feats[-2],style_feats[-2],style_feats[-1],style_feats[-1]))

        loss_lambda1 = self.calc_content_loss(Icc,content) + self.calc_content_loss(Iss,style)
        
        Icc_feats = self.encode_with_intermediate(Icc)
        Iss_feats = self.encode_with_intermediate(Iss)

        loss_lambda2 = 0
        for i in range(5):
            loss_lambda2 += self.calc_content_loss(Icc_feats[i],content_featss[i]) + self.calc_content_loss(Iss_feats[i],style_feats[i])
        return loss_c,loss_s,loss_lambda1,loss_lambda2
        

if __name__ == "__main__":
    print("Hello,{}".format(__file__))
    content_rand = torch.rand(1,3,512,512)
    style_rand = torch.rand(1,3,512,512)
    print('-------------------- Net test -------------------')
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print('device => {}'.format(device))
    _DEFAULT_VGG_STYLIZED_CONTENT_PATH = '../models/vgg_stylized_content_iter_485000_63.pth'
    _DEFAULT_VGG_STYLIZED_STYLE_PATH = '../models/vgg_stylized_style_iter_485000_63.pth'
    _DEFAULT_TRANSFORM_PATH = '../models/sa_module_iter_485000_63.pth'
    print("vgg_stylized_content parameter file path => {}".format(_DEFAULT_VGG_STYLIZED_CONTENT_PATH))
    print("vgg_stylized_style parameter file path => {}".format(_DEFAULT_VGG_STYLIZED_STYLE_PATH))
    print("transform parameter file path => {}".format(_DEFAULT_TRANSFORM_PATH))
    net = Net(_DEFAULT_VGG_STYLIZED_CONTENT_PATH,_DEFAULT_VGG_STYLIZED_STYLE_PATH,_DEFAULT_TRANSFORM_PATH)
    print("content_rand.shape => {}".format(content_rand.shape))
    print("style_rand.shape => {}".format(style_rand.shape))
    #loss_c,loss_s,loss_lambda1,loss_lambda2 = net(content_rand,style_rand)
    image = net.generate_image(content_rand,style_rand)
    print("image.shape => {}".format(image.shape))
    print('-------------------------------------------------')
    print('clear!!!')
