import torch
import torch.nn as nn
from Convolution import conv3x3
from Residuals import BasicBlock

class LowToHighGenerator(nn.Module):
    def __init__(self, ngpu=1):
        super(LowToHighGenerator, self).__init__()
        self.ngpu = ngpu
        res_units = [256, 128, 96]
        inp_res_units = [
            [256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256,
             256], [256, 128, 128], [128, 96, 96]]

        self.layers_set = []
        self.layers_set_up = []
        self.layers_set_final = nn.ModuleList()
        self.layers_set_final_up = nn.ModuleList()

        self.a1 = nn.Sequential(nn.Conv2d(256, 128, 1, 1))
        self.a2 = nn.Sequential(nn.Conv2d(128, 96, 1, 1))

        self.layers_in = conv3x3(3, 256)

        layers = []
        for ru in range(len(res_units) - 1):
            nunits = res_units[ru]
            curr_inp_resu = inp_res_units[ru]
            self.layers_set.insert(ru, [])
            self.layers_set_up.insert(ru, [])

            if ru == 0:
                num_blocks_level = 12
            else:
                num_blocks_level = 3

            for j in range(num_blocks_level):
                # if curr_inp_resu[j]==3:
                self.layers_set[ru].append(BasicBlock(curr_inp_resu[j], nunits))
                # else:
                # layers.append(MyBlock(curr_inp_resu[j], nunits))

            self.layers_set_up[ru].append(nn.Upsample(scale_factor=2, mode='bilinear',align_corners=True))

            self.layers_set_up[ru].append(nn.BatchNorm2d(nunits))
            self.layers_set_up[ru].append(nn.ReLU(True))
            self.layers_set_up[ru].append(nn.ConvTranspose2d(nunits, nunits, kernel_size=1, stride=1))
            self.layers_set_final.append(nn.Sequential(*self.layers_set[ru]))
            self.layers_set_final_up.append(nn.Sequential(*self.layers_set_up[ru]))

        nunits = res_units[-1]
        layers.append(conv3x3(inp_res_units[-1][0], nunits))
        layers.append(nn.ReLU(True))

        layers.append(nn.Conv2d(inp_res_units[-1][1], nunits, kernel_size=1, stride=1))
        layers.append(nn.ReLU(True))

        layers.append(nn.Conv2d(nunits, 3, kernel_size=1, stride=1))
        layers.append(nn.Tanh())

        self.main = nn.Sequential(*layers)

    def forward(self, input):
        x = self.layers_in(input)
        for ru in range(len(self.layers_set_final)):
            if ru == 0:
                temp = self.layers_set_final[ru](x)
                x = x + temp
            elif ru == 1:
                temp = self.layers_set_final[ru](x)
                temp2 = self.a1(x)
                x = temp + temp2
            elif ru == 2:
                temp = self.layers_set_final[ru](x)
                temp2 = self.a2(x)
                x = temp + temp2
            x = self.layers_set_final_up[ru](x)

        x = self.main(x)

        return x


class HighToLowGenerator(nn.Module):
    def __init__(self, ngpu=1):
        super(HighToLowGenerator, self).__init__()
        self.ngpu = ngpu
        res_units = [96, 128, 256, 512, 256, 128]
        inp_res_units = [
            [96, 96], [96, 128], [128, 256], [256, 512], [512, 256],[256, 128]]

        self.layers_set = []
        self.layers_set_up = []
        self.layers_set_final = nn.ModuleList()
        self.layers_set_final_up = nn.ModuleList()

        self.layers_in = conv3x3(4, 96)

        layers = []

        num_blocks_level = 2

        for ru in range(4):
            nunits = res_units[ru]
            curr_inp_resu = inp_res_units[ru]
            self.layers_set.insert(ru, [])
            self.layers_set_up.insert(ru, [])
            for j in range(num_blocks_level):
                self.layers_set[ru].append(BasicBlock(curr_inp_resu[j], nunits))
            
            self.layers_set_up[ru].append(nn.AvgPool2d(2, 2))
            
            self.layers_set_up[ru].append(nn.BatchNorm2d(nunits))
            self.layers_set_up[ru].append(nn.ReLU(True))
            self.layers_set_up[ru].append(nn.ConvTranspose2d(nunits, nunits, kernel_size=1, stride=1))
            self.layers_set_final.append(nn.Sequential(*self.layers_set[ru]))
            self.layers_set_final_up.append(nn.Sequential(*self.layers_set_up[ru]))
    
        for ru in range(4,len(res_units)):
            nunits = res_units[ru]
            curr_inp_resu = inp_res_units[ru]
            self.layers_set.insert(ru, [])
            self.layers_set_up.insert(ru, [])
            for j in range(num_blocks_level):
                self.layers_set[ru].append(BasicBlock(curr_inp_resu[j], nunits))
                
            self.layers_set_up[ru].append(nn.Upsample(scale_factor=2, mode='bilinear',align_corners=True))

            self.layers_set_up[ru].append(nn.BatchNorm2d(nunits))
            self.layers_set_up[ru].append(nn.ReLU(True))
            self.layers_set_up[ru].append(nn.ConvTranspose2d(nunits, nunits, kernel_size=1, stride=1))
            self.layers_set_final.append(nn.Sequential(*self.layers_set[ru]))
            self.layers_set_final_up.append(nn.Sequential(*self.layers_set_up[ru]))

        layers.append(nn.Conv2d(res_units[-1], 3, kernel_size=1, stride=1))
        layers.append(nn.Tanh())
        
        self.main = nn.Sequential(*layers)

    def forward(self, input, noise):
        noise_img = noise.view(noise.size(0), noise.size(1), 1, 1).expand(
                noise.size(0), noise.size(1), input.size(2), input.size(3))
        input = torch.cat([input, noise_img], 1)
        x = self.layers_in(input)
        for ru in range(len(self.layers_set_final)):
            x = self.layers_set_final[ru](x)
            x = self.layers_set_final_up[ru](x)
        x = self.main(x)

        return x