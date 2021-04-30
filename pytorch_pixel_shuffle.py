import torch.nn as nn
import torch
import functools
from torch.nn import init
def pixel_shuffle():
    netG = nn.Sequential(
            nn.Conv2d(16, 64, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.1, True),
        )
    weights_init_kaiming_ = functools.partial(weights_init_kaiming, scale=1)
    netG.apply(weights_init_kaiming_)
    if isinstance(netG, nn.DataParallel):
        netG = netG.module
    netG.eval().cuda()
    dummy_input = torch.randn(1, 16, 200, 200).cuda()
    dynamic_axes = { 
                'color_input':  {2: 'dy_num',3:'dy_num'},
                'color_output':{2: 'dy_num',3:'dy_num'}
                }
    
    torch.onnx.export(netG, dummy_input, r"./pixel_shuffle222.onnx",opset_version=10, export_params=True,input_names = ['color_input'],output_names=['color_output'],dynamic_axes=dynamic_axes)
    a = input("   qingshuru: ") 
def weights_init_kaiming(m, scale=1):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        m.weight.data *= scale
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        m.weight.data *= scale
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.constant_(m.weight.data, 1.0)
        init.constant_(m.bias.data, 0.0)

pixel_shuffle()
