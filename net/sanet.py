"""

SANet is a model to change content image feature to style image features.

Transform is a model VGG 4 and 5 layers of SANet.

"""
import torch
import torch.nn as nn

class SANet(nn.Module):
    def __init__(self,in_planes):
        super().__init__()
        self.f = nn.Conv2d(in_planes,in_planes,(1,1))
        self.g = nn.Conv2d(in_planes,in_planes,(1,1))
        self.h = nn.Conv2d(in_planes,in_planes,(1,1))
        self.sm = nn.Softmax(dim = -1)
        self.out_conv = nn.Conv2d(in_planes,in_planes,(1,1))

    def new_sqrt(self,feat):
        """
        for transfering to Swift
        """
        return torch.pow(feat,0.5)

    def new_var(self,feat):
        """
        for transering to Swift
        """
        mu = torch.mead(feat,dim = 2)
        mu = torch.reshape(x=mu,shape=(mu.shape[0],mu.shape[1],1))
        sub = feat - mui
        sub = sub * sub
        var = torch.mean(sub,dim=2)
        shape = feat.shape[2]
        var = (shape/(shape - 1)) * var
        return var

    def calc_mean_std(self,feat,eps=1e-5):
        size = feat.size()
        assert (len(size)==4)
        N,C = size[:2]
        feat_var = self.new_var(feat.view(N,C,-1)) + eps
        feat_std = self.new_sqrt(feat_var).view(N,C,1,1)
        feat_mean = feat.view(N,C,-1).mean(dim=2).view(N,C,1,1)
        return feat_mean,feat_std

    def mean_variance_norm(self,feat):
        size = feat.size()
        mean,std = self.calc_mean_std(feat)
        normalized_feat = (feat - mean.expand(size)) / std.expand(size)
        return normalized_feat

    def forward(self,content,style):
        F = self.f(self.mean_variance_norm(content))
        G = self.g(self.mean_variance_norm(style))
        H = self.h(style)
        b,c,h,w = F.size()
        F = F.view(b,c,int(w*h)).permute(0,2,1)
        b,c,h,w = G.size()
        G = G.view(b,c,int(w*h))
        S = torch.bmm(F,G)
        S = self.sm(S)
        b,c,h,w = H.size()
        H = H.view(b,c,int(w*h))
        S = S.permute(0,2,1)
        O = torch.bmm(H,S)
        b,c,h,w = content.size()
        O = O.view(b,c,h,w)
        O = self.out_conv(O)
        O += content
        return O
