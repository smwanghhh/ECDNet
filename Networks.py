from torch import nn
from torch.nn import functional as F
import torch
from torchvision import models
import math
from torchsummary import summary
import skimage
import numpy as np

class Encoder(nn.Module):
    def __init__(self, img_size, z_app, z_geo, num_class, drop_rate=0):
        super(Encoder, self).__init__()
        self.drop_rate = drop_rate
        pretrain = True
        if pretrain:
#            self.feature_app = nn.Sequential(model_define(), nn.Flatten())
            self.feature_geo = nn.Sequential(model_define(), nn.Flatten())

        else:
            self.feature_geo = nn.Sequential(*list(models.resnet18(pretrained= True).children())[:-1], nn.Flatten())
        self.feature_app = nn.Sequential(*list(models.resnet18(pretrained= True).children())[:-1], nn.Flatten())
        self.fc_app =  nn.Linear(512, z_app)
        self.fc_geo = nn.Linear(512, z_geo)
        self.fc_num = nn.Sequential(nn.Dropout(0.5), nn.Linear(z_geo, num_class))



    def forward(self, x):
        z_app = self.feature_app(x)
        z_geo = self.feature_geo(x)
        # fea1 = self.feature_app[:-2](x)
        # fea2 = self.feature_geo[:-2](x)
        if self.drop_rate > 0:
            z_app =  nn.Dropout(0.5)(z_app)
            z_geo =  nn.Dropout(0.5)(z_geo)

        z_app = self.fc_app(z_app)
        z_geo = self.fc_geo(z_geo)
        logit = self.fc_num(z_geo)
        return z_app, z_geo, logit

    def get_parameters(self):

        parameter = [{"params": self.feature_app.parameters()},
                        {"params": self.fc_app.parameters()},
                        {"params": self.feature_geo.parameters()},
                        {"params": self.fc_geo.parameters()},
                        {"params": self.fc_num.parameters()}]


        return parameter

def model_define():
    pretrain_dict = torch.load('./models/resnet18_pretrained_on_msceleb.pth.tar')['state_dict']
    # model = nn.Sequential(*list(models.resnet18().children())[:-2])
    model = models.resnet18(num_classes= 12666)
    model_dict = model.state_dict()
    pre_dict = {}
    dict ={}
    for k, v in pretrain_dict.items():
        pre_dict[k] = v
    for k, v in model_dict.items():
        dict[k] = v

    keys = list(dict.keys())
    values = list(pre_dict.values())
    for i in range(len(keys)):
        dict[keys[i]] = values[i]
    model_dict.update(dict)
    model.load_state_dict(model_dict)
    model = nn.Sequential(*list(model.children())[:-1])
    return model

class Percep_model(nn.Module):
    def __init__(self):
        super(Percep_model, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slices1 = nn.Sequential()
        self.slices2 = nn.Sequential()
        self.slices3 = nn.Sequential()
        self.slices4 = nn.Sequential()
        self.slices5 = nn.Sequential()
        for x in range(2):
            self.slices1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slices2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slices3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slices4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slices5.add_module(str(x), vgg_pretrained_features[x])
        for param in self.parameters():
            param.requires_grad = False

        self.slices1 = self.slices1.cuda()
        self.slices2 = self.slices2.cuda()
        self.slices3 = self.slices3.cuda()
        self.slices4 = self.slices4.cuda()
        self.slices5 = self.slices5.cuda()

        # pretrain_dict = torch.load('./models/resnet18_pretrained_on_msceleb.pth.tar')['state_dict']
        # # model = nn.Sequential(*list(models.resnet18().children())[:-2])
        # model = models.resnet18(num_classes= 12666)
        # model_dict = model.state_dict()
        # pre_dict = {}
        # dict ={}
        # for k, v in pretrain_dict.items():
        #     pre_dict[k] = v
        # for k, v in model_dict.items():
        #     dict[k] = v
        #
        # keys = list(dict.keys())
        # values = list(pre_dict.values())
        # for i in range(len(keys)):
        #     dict[keys[i]] = values[i]
        # model_dict.update(dict)
        # model.load_state_dict(model_dict)
        # self.model = nn.Sequential(*list(model.children())[:1])

    def forward(self, img):
        out1 = self.slices1(img)
        out2 = self.slices2(out1)
        out3 = self.slices3(out2)
        out4 = self.slices4(out3)
        out5 = self.slices5(out4)
        return out1, out2, out3, out4, out5

class Decoder(nn.Module):
    def __init__(self, z_dim, img_size):
        super(Decoder, self).__init__()
        self.fmd = 7
        self.ch = [128, 64, 32, 16]
        self.chb = [int(i * 5 / 8) for i in self.ch]
        self.kernel = [4, 4, 4, 4, 4]#[5, 5, 3, 3, 2]
        self.stride = [2, 2, 2, 2, 2]#[3, 2, 2, 2, 1]
        # self.fmd=7
        self.img_size = img_size


        self.fc_geo = nn.Linear(z_dim, self.fmd * self.fmd * self.ch[0])
        self.deconv_geo = nn.Sequential()
        for i in range(len(self.ch)-1):
            self.deconv_geo.add_module('deconv%s'%str(i), nn.ConvTranspose2d(self.ch[i], self.ch[i+1], self.kernel[i], self.stride[i] , 1))
            self.deconv_geo.add_module('relu%s' % str(i), nn.ReLU())
        self.deconv_geo.add_module('deconv%s' % str(len(self.ch)), nn.ConvTranspose2d(self.ch[-1], 2, self.kernel[-1], self.stride[-1], 1))
        self.deconv_geo.add_module('tanh', nn.Tanh())



        self.fc_app = nn.Linear(z_dim, self.fmd * self.fmd * self.chb[0])
        self.deconv_app = nn.Sequential()
        for i in range(len(self.ch)-1):
            self.deconv_app.add_module('deconv%s' % str(i),
                                       nn.ConvTranspose2d(self.chb[i], self.chb[i + 1], self.kernel[i], self.stride[i], 1))
            self.deconv_app.add_module('relu%s' % str(i), nn.ReLU())
        self.deconv_app.add_module('deconv%s' % str(len(self.chb)),
                                   nn.ConvTranspose2d(self.chb[-1], 3, self.kernel[-1], self.stride[-1], 1))
        self.deconv_app.add_module('sigmoid', nn.Sigmoid())
        # self.fc_geo = nn.Linear(z_dim, self.fmd * self.fmd * self.ch[0])
        # self.deconv_geo = nn.Sequential()
        # # self.deconv_geo.add_module('bn', nn.BatchNorm2d(self.ch[0]))
        # for i in range(int(math.sqrt(self.img_size // self.fmd))):
        #     self.deconv_geo.add_module('up'+str(i), nn.Upsample(scale_factor=2))
        #     if i == 0:
        #         self.deconv_geo.add_module('conv'+str(i), nn.Conv2d(self.ch[i], self.ch[i], 3, stride=1, padding=1))
        #     else:
        #         self.deconv_geo.add_module('conv' + str(i), nn.Conv2d(self.ch[i-1], self.ch[i], 3, stride=1, padding=1))
        #     # self.deconv_geo.add_module('bn'+str(i), nn.BatchNorm2d(self.ch[i]))
        #     self.deconv_geo.add_module('relu'+str(i), nn.ReLU())
        # self.deconv_geo.add_module('conv', nn.Conv2d(self.ch[-1], 2, 3, stride=1, padding=1))
        # self.deconv_geo.add_module('tanh', nn.Tanh())
        #
        #
        # self.fc_app = nn.Linear(z_dim, self.fmd * self.fmd * self.chb[0])
        # self.deconv_app = nn.Sequential()
        # # self.deconv_app.add_module('bn', nn.BatchNorm2d(self.chb[0]))
        # for i in range(int(math.sqrt(self.img_size // self.fmd))):
        #     self.deconv_app.add_module('up'+str(i), nn.Upsample(scale_factor=2))
        #     if i == 0:
        #         self.deconv_app.add_module('conv'+str(i), nn.Conv2d(self.chb[i], self.chb[i], 3, stride=1, padding=1))
        #     else:
        #         self.deconv_app.add_module('conv' + str(i), nn.Conv2d(self.chb[i-1], self.chb[i], 3, stride=1, padding=1))
        #     # self.deconv_app.add_module('bn'+str(i), nn.BatchNorm2d(self.chb[i], 0.8))
        #     self.deconv_app.add_module('relu'+str(i), nn.ReLU())
        # self.deconv_app.add_module('conv', nn.Conv2d(self.chb[-1], 3, 3, stride=1, padding=1))
        # self.deconv_app.add_module('tanh', nn.Sigmoid())

    def forward(self, z_app, z_geo):
        z_geo = self.fc_geo(z_geo)
        z_geo = torch.reshape(z_geo, (-1, self.ch[0], self.fmd, self.fmd))
        geo = self.deconv_geo(z_geo)
        geo = torch.reshape(geo, (-1, 2, self.img_size, self.img_size))

        z_app = self.fc_app(z_app)
        z_app = torch.reshape(z_app, (-1, self.chb[0], self.fmd, self.fmd))
        app = self.deconv_app(z_app)
        return app, geo

    def get_parameters(self):

        parameter = [{"params": self.fc_geo.parameters()},
                     {"params": self.deconv_geo.parameters()},
                     {"params": self.fc_app.parameters()},
                     {"params": self.deconv_app.parameters()},]
        return parameter


class FER_model(nn.Module):
    def __init__(self, img_size, num_class, drop_rate=0.5):
        super(FER_model, self).__init__()
        pretrain = False
        if pretrain:
            self.feature = nn.Sequential(model_define(), nn.Flatten())
        else:
            self.feature1 = nn.Sequential(*list(models.resnet18(pretrained=False).children())[:-2])
            self.feature2 = nn.Sequential(nn.Conv2d(512, 512, 1, 1, 0), nn.ReLU(), nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten())
        self.fc = nn.Sequential(nn.Dropout(drop_rate), nn.Linear(512, 128), nn.Dropout(drop_rate), nn.Linear(128, num_class))
        #self.fc = nn.Sequential(nn.Linear(512, num_class))

    def forward(self, x):
        fea = self.feature1(x)
        fea = self.feature2(fea)
        fea = self.fc(fea)
        return fea

    def get_parameters(self):

        parameter = [{"params": self.feature1.parameters()},
                     {"params": self.feature2.parameters()},
                         {"params": self.fc.parameters()}]

        return parameter
