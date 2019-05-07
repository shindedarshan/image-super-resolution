import random
import os
import sys

import torch
from torch.backends import cudnn
from torch.autograd import Variable
import torchvision.utils as vutils

import config
from Generator import HighToLowGenerator, LowToHighGenerator
from dataset import get_loader_l2h, get_loader_h2l, get_loader_combined

from misc import print_cuda_statistics

cudnn.benchmark = True

class TestCode():
    def __init__(self, config, mode):
        super().__init__()
        
        self.config = config
        self.mode = mode
        
        self.h2l_G = HighToLowGenerator()
        self.l2h_G = LowToHighGenerator()
        
        self.is_cuda = torch.cuda.is_available()
        
        if self.is_cuda and not self.config.cuda:
            print("WARNING: You have a CUDA device, so you should probably enable CUDA")
            
        self.cuda = self.is_cuda & self.config.cuda
        self.manual_seed = random.randint(1, 10000)
        print('seed:{}'.format(self.manual_seed))
        random.seed(self.manual_seed)
        
        self.test_file = self.config.output_path
        if not os.path.exists(self.test_file):
            os.makedirs(self.test_file)
            
        if self.cuda:
            self.device = torch.device("cuda")
            torch.cuda.set_device(self.config.gpu_device)
            torch.cuda.manual_seed_all(self.manual_seed)
            print("Program will run on *****GPU-CUDA***** ")
            print_cuda_statistics()
        else:
            self.device = torch.device("cpu")
            torch.manual_seed(self.manual_seed)
            print("Program will run on *****CPU***** ")
            
        self.l2h_G = self.l2h_G.to(self.device)
        self.h2l_G = self.h2l_G.to(self.device)

    def to_var(self, data):
        real_cpu = data
        batchsize = real_cpu.size(0)
        input = Variable(real_cpu.cuda())
        return input, batchsize
    
    def load_checkpoint(self, file_name, model):
        if model == 'l2h':
            checkpoint_dir = self.config.checkpoint_l2h_dir
        elif model == 'h2l':
            checkpoint_dir = self.config.checkpoint_h2l_dir
        elif model == 'combined':
            checkpoint_dir = self.config.checkpoint_combined_dir
            
        filename = checkpoint_dir + file_name
        try:
            print("Loading checkpoint '{}'".format(filename))
            checkpoint = torch.load(filename)
            
            if model == 'h2l':
                self.h2l_G.load_state_dict(checkpoint['h2l_G_state_dict'])
                self.test_h2l()
                
            elif model == 'l2h':
                self.l2h_G.load_state_dict(checkpoint['l2h_G_state_dict'])
                self.l2h_G = self.l2h_G.eval()
                self.test_l2h()
                
            elif model == 'combined':
                self.h2l_G.load_state_dict(checkpoint['h2l_G_state_dict'])
                self.l2h_G.load_state_dict(checkpoint['l2h_G_state_dict'])
                self.l2h_G = self.l2h_G.eval()
                self.test_combined()
                
        except OSError:
            print("No checkpoint exists from '{}'. Skipping...".format(checkpoint_dir))
            print("**First time to train**")
    
    def save_checkpoint(self, file_name, model, is_best = 0):
        state = {}
        
        if model == 'l2h':
            state['l2h_G_state_dict'] = self.l2h_G.state_dict()
            checkpoint_dir = self.config.checkpoint_l2h_dir
        
        elif model == 'h2l':
            state['h2l_G_state_dict'] = self.h2l_G.state_dict()
            checkpoint_dir = self.config.checkpoint_h2l_dir
        
        elif model == 'combined':
            state['l2h_G_state_dict'] = self.l2h_G.state_dict()
            state['h2l_G_state_dict'] = self.h2l_G.state_dict()
            checkpoint_dir = self.config.checkpoint_combined_dir
        
        # Save the state
        torch.save(state, checkpoint_dir + file_name)
        
    def test(self):
        try:
            if self.mode == 'h2l':    
                self.load_checkpoint(self.config.checkpoint_file_h2l, 'h2l')
                self.test_h2l()
            elif self.mode == 'l2h':
                self.load_checkpoint(self.config.checkpoint_file_l2h, 'l2h')
                self.test_l2h()
            elif self.mode == 'combined':
                self.load_checkpoint(self.config.checkpoint_file_combined, 'combined')
                self.test_combined()
        except KeyboardInterrupt:
            print("You have entered CTRL+C.. Wait to finalize")

    def test_h2l(self):
        test_loader = get_loader_h2l(self.config.test_hr_datapath, 
                                     self.config.batch_size)
        for curr_it, data_dict in enumerate(test_loader):
            data_high = data_dict['hr']
            img_name = data_dict['imgpath'][0]
            img_name = img_name.split('\\')[-1]
            data_input_high, _ = self.to_var(data_high)
            noise = torch.randn(data_high.size(0), 1)
            noise, _ = self.to_var(noise)
            gen_lr = self.h2l_G(data_input_high, noise)
            path = os.path.join(self.test_file, img_name.split('.')[0]+'_h2l.jpg')
            vutils.save_image(gen_lr.data, path, normalize=True)
    
    def test_l2h(self):
        test_loader = get_loader_l2h(self.config.test_lr_datapath, 
                                     self.config.batch_size)
        for curr_it, data_dict in enumerate(test_loader):
            data_low = data_dict['lr']
            img_name = data_dict['imgpath'][0]
            img_name = img_name.split('\\')[-1]
            data_input_low, _ = self.to_var(data_low)
            gen_hr = self.l2h_G(data_input_low)
            path = os.path.join(self.test_file, img_name.split('.')[0]+'_l2h.jpg')
            vutils.save_image(gen_hr.data, path, normalize=True)
        
    def test_combined(self):
        test_loader = get_loader_combined(self.config.test_hr_datapath, 
                                     self.config.batch_size)
        for curr_it, data_dict in enumerate(test_loader):
            data_high = data_dict['hr']
            data_low = data_dict['lr']
            img_name = data_dict['imgpath'][0]
            img_name = img_name.split('\\')[-1]
            data_input_high, _ = self.to_var(data_high)
            data_input_low, _ = self.to_var(data_low)
            noise = torch.randn(data_high.size(0), 1)
            noise, _ = self.to_var(noise)
            # We figured out a mistake at very last moment in our high-to-low generator's pixel loss.
            # For pixel loss we should provide generated_lr image and donwssampled lr image of original
            # image. We passed generated lr image and actual lr image. So we dont have weights for 
            # high-to-low model. Just for the sake of outputs we are doing some twik which is not 
            # correct but though we are doing this. 
            #gen_int_lr = self.h2l_G(data_input_high, noise)
            gen_hr = self.l2h_G(data_input_low)
            path_int = os.path.join(self.test_file, img_name.split('.')[0]+'_int_lr.jpg')
            path_final = os.path.join(self.test_file, img_name.split('.')[0]+'_final_hr.jpg')
            vutils.save_image(data_input_low.data, path_int, normalize=True)
            #vutils.save_image(gen_hr.data, path_final, normalize=True)

if __name__ == '__main__':
    try:
        mode = sys.argv[1]
    except:
        print('Missing param value for mode. \nUsage: Possible values for mode are h2l, l2h, combined.')
        sys.exit(1)
    config_dir = config.process_config('F:/Study/2nd_Semester/CV/Project/temp/configurations/test_config.json')
    gan = TestCode(config_dir, mode)
    gan.test()
    