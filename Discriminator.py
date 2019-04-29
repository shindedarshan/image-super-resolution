import torch.nn as nn
import torch
from Convolution import conv3x3
from Residuals import BasicBlock
    
class LowToHighDiscriminator(nn.Module):
    def __init__(self, ngpu=1):
        super(LowToHighDiscriminator, self).__init__()
        self.ngpu = ngpu
        res_units = [256, 128, 64, 32, 1]
        inp_res_units = [
            [256, 256], [256, 128], [128, 64], [64, 32], [32, 1]]

        self.layers_set = []
        self.layers_set_down = []
        self.layers_set_final = nn.ModuleList()
        self.layers_set_final_down = nn.ModuleList()

        self.layers_in = conv3x3(3, 256)

        layers = []
        for ru in range(len(res_units) - 1):
            nunits = res_units[ru]
            curr_inp_resu = inp_res_units[ru]
            self.layers_set.insert(ru, [])
            self.layers_set_down.insert(ru, [])

            if ru == 0:
                num_blocks_level = 2
            else:
                num_blocks_level = 1

            for j in range(num_blocks_level):
                self.layers_set[ru].append(BasicBlock(curr_inp_resu[j], nunits, nobn = True))

            self.layers_set_down[ru].append(nn.MaxPool2d(2,2))

            self.layers_set_down[ru].append(nn.ReLU(True))
            self.layers_set_down[ru].append(nn.ConvTranspose2d(nunits, nunits, kernel_size=1, stride=1))
            self.layers_set_final.append(nn.Sequential(*self.layers_set[ru]))
            self.layers_set_final_down.append(nn.Sequential(*self.layers_set_down[ru]))

        nunits = res_units[-1]
        layers.append(conv3x3(inp_res_units[-1][0], nunits))
        layers.append(nn.ReLU(True))

        layers.append(nn.Conv2d(inp_res_units[-1][1], nunits, kernel_size=1, stride=1))
        layers.append(nn.ReLU(True))

        layers.append(nn.Conv2d(nunits, 3, kernel_size=1, stride=1))
        layers.append(nn.Sigmoid())

        self.main = nn.Sequential(*layers)
        
    def forward(self, input):
        batch_size = input.shape[0]
        x = self.layers_in(input)
        for ru in range(len(self.layers_set_final)):
            x = self.layers_set_final[ru](x)
            x = self.layers_set_final_down[ru](x)
        x = self.main(x)
        x = x.view(-1)
        fc = nn.Linear(x.shape[0], batch_size)
        fc.to(torch.device('cuda:0'))
        out = fc(x)        
        return out
    
class HighToLowDiscriminator(nn.Module):
    def __init__(self, ngpu=1):
        super(HighToLowDiscriminator, self).__init__()
        self.ngpu = ngpu
        res_units = [256, 128, 96]
        inp_res_units = [
            [256, 256, 256, 256], [256, 128], [128, 96]]

        self.layers_set = []
        self.layers_set_down = []
        self.layers_set_final = nn.ModuleList()
        self.layers_set_final_down = nn.ModuleList()

        self.layers_in = conv3x3(3, 256)

        layers = []
        for ru in range(len(res_units) - 1):
            nunits = res_units[ru]
            curr_inp_resu = inp_res_units[ru]
            self.layers_set.insert(ru, [])
            self.layers_set_down.insert(ru, [])

            if ru == 0:
                num_blocks_level = 4
            else:
                num_blocks_level = 1

            for j in range(num_blocks_level):
                self.layers_set[ru].append(BasicBlock(curr_inp_resu[j], nunits, nobn = True))

            self.layers_set_down[ru].append(nn.MaxPool2d(2,2))

            self.layers_set_down[ru].append(nn.ReLU(True))
            self.layers_set_down[ru].append(nn.ConvTranspose2d(nunits, nunits, kernel_size=1, stride=1))
            self.layers_set_final.append(nn.Sequential(*self.layers_set[ru]))
            self.layers_set_final_down.append(nn.Sequential(*self.layers_set_down[ru]))

        nunits = res_units[-1]
        layers.append(conv3x3(inp_res_units[-1][0], nunits))
        layers.append(nn.ReLU(True))

        layers.append(nn.Conv2d(inp_res_units[-1][1], nunits, kernel_size=1, stride=1))
        layers.append(nn.ReLU(True))

        layers.append(nn.Conv2d(nunits, 3, kernel_size=1, stride=1))
        layers.append(nn.Sigmoid())

        self.main = nn.Sequential(*layers)
        
    def forward(self, input):
        batch_size = input.shape[0]
        x = self.layers_in(input)
        for ru in range(len(self.layers_set_final)):
            x = self.layers_set_final[ru](x)
            x = self.layers_set_final_down[ru](x)
        x = self.main(x)
        x = x.view(-1)
        fc = nn.Linear(x.shape[0], batch_size)
        fc.to(torch.device('cuda:0'))
        out = fc(x)        
        return out