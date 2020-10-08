"""

Styleize_feats is a model to make SANet easier to understand when passing features to SANet

"""

import torch
import torch.nn as nn

class VGG_Attention_Content(nn.Module):
    """
    For Content image feats
    """
    def __init__(self,dim):
        super().__init__()
        self.cnn = nn.Conv2d(dim,dim,(1,1))
    def forward(self,feats):
        output = self.cnn(feats)
        return output

class VGG_Attention_Style(nn.Module):
    """
    For Style image feats
    """
    def __init__(self,dim):
        super().__init__()
        self.cnn = nn.Conv2d(dim,dim,(1,1))
    def forward(self,feats):
        output = self.cnn(feats)
        return output

vgg_stylized_conv_5_content = VGG_Attention_Content(dim=512)
vgg_stylized_conv_4_content = VGG_Attention_Content(dim=512)
vgg_stylized_conv_3_content = VGG_Attention_Content(dim=256)
vgg_stylized_conv_2_content = VGG_Attention_Content(dim=128)
vgg_stylized_conv_1_content = VGG_Attention_Content(dim=64)

vgg_stylized_conv_5_style = VGG_Attention_Style(dim=512)
vgg_stylized_conv_4_style = VGG_Attention_Style(dim=512)
vgg_stylized_conv_3_style = VGG_Attention_Style(dim=256)
vgg_stylized_conv_2_style = VGG_Attention_Style(dim=128)
vgg_stylized_conv_1_style = VGG_Attention_Style(dim=64)

class VGG_Stylized_Content(nn.Module):
    def __init__(self,start=3):
        super().__init__()
        self.vgg_stylized_conv_5 = vgg_stylized_conv_5_content
        self.vgg_stylized_conv_4 = vgg_stylized_conv_4_content
        self.vgg_stylized_conv_3 = vgg_stylized_conv_3_content
        self.vgg_stylized_conv_2 = vgg_stylized_conv_2_content
        self.vgg_stylized_conv_1 = vgg_stylized_conv_1_content
    def forward(self,inputs):
        results = []
        j = 0
        for i in range(3,5):
            func = getattr(self,'vgg_stylized_conv_{}'.format(i+1))
            results.append(func(inputs[j]))
            j += 1
        return results

class VGG_Stylized_Style(nn.Module):
    def __init__(self,start=3):
        super().__init__()
        self.vgg_stylized_conv_5_style = vgg_stylized_conv_5_style
        self.vgg_stylized_conv_4_style = vgg_stylized_conv_4_style
        self.vgg_stylized_conv_3_style = vgg_stylized_conv_3_style
        self.vgg_stylized_conv_2_style = vgg_stylized_conv_2_style
        self.vgg_stylized_conv_1_style = vgg_stylized_conv_1_style
    def forward(self,inputs):
        results = []
        j = 0
        for i in range(3,5):
            func = getattr(self,'vgg_stylized_conv_{}_style'.format(i+1))
            results.append(func(inputs[j]))
            j += 1
        return results


if __name__ == "__main__":
    print("Hello,{}".format(__file__))
    content = VGG_Stylized_Content()
    style = VGG_Stylized_Style()
    rand = [torch.rand(1,512,32,32),torch.rand(1,512,16,16)]
    content_output = content(rand)
    style_output = style(rand)
    for content in content_output:
        print(content.shape)
    for style in style_output:
        print(style.shape)
