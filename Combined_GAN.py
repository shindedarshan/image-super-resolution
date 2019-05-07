import shutil
import random
import os

import torch
from torch.backends import cudnn
from torch.autograd import Variable
import torchvision.utils as vutils

import config
from base import BaseAgent
from Generator import HighToLowGenerator, LowToHighGenerator
from Discriminator import HighToLowDiscriminator, LowToHighDiscriminator
from loss import MSELoss
from dataset import get_loader

from tensorboardX import SummaryWriter
from misc import print_cuda_statistics

cudnn.benchmark = True

class Combined_GAN(BaseAgent):
    def __init__(self, config):
        super().__init__(config)

        # define models (generator and discriminator)
        self.h2l_G = HighToLowGenerator()
        self.h2l_D = HighToLowDiscriminator()
        self.l2h_G = LowToHighGenerator()
        self.l2h_D = LowToHighDiscriminator()

        # define loss
        #self.loss = GANLoss()
        #self.loss = HingeEmbeddingLoss()
        self.criterion_GAN = torch.nn.BCEWithLogitsLoss()
        self.criterion_MSE = MSELoss()
        
        # define optimizers for both generator and discriminator
        self.l2h_optimG = torch.optim.Adam(self.l2h_G.parameters(), lr=self.config.learning_rate, betas=(self.config.beta1, self.config.beta2))
        self.l2h_optimD = torch.optim.Adam(self.l2h_D.parameters(), lr=self.config.learning_rate, betas=(self.config.beta1, self.config.beta2))
        self.h2l_optimG = torch.optim.Adam(self.h2l_G.parameters(), lr=self.config.learning_rate, betas=(self.config.beta1, self.config.beta2))
        self.h2l_optimD = torch.optim.Adam(self.h2l_D.parameters(), lr=self.config.learning_rate, betas=(self.config.beta1, self.config.beta2))
        
        # initialize counter
        self.current_epoch = 0
        self.current_iteration = 0
        self.best_valid_mean_iou = 0
        
        self.real_label = 1
        self.fake_label = -1
        
        # set cuda flag
        self.is_cuda = torch.cuda.is_available()
        if self.is_cuda and not self.config.cuda:
            self.logger.info("WARNING: You have a CUDA device, so you should probably enable CUDA")

        self.cuda = self.is_cuda & self.config.cuda
        
        # set the manual seed for torch
        self.manual_seed = random.randint(1, 10000)
        self.logger.info ('seed:{}'.format(self.manual_seed))
        random.seed(self.manual_seed)
        
        self.test_file = self.config.output_path
        if not os.path.exists(self.test_file):
            os.makedirs(self.test_file)

        if self.cuda:
            self.device = torch.device("cuda")
            torch.cuda.set_device(self.config.gpu_device)
            torch.cuda.manual_seed_all(self.manual_seed)
            self.logger.info("Program will run on *****GPU-CUDA***** ")
            print_cuda_statistics()
        else:
            self.device = torch.device("cpu")
            torch.manual_seed(self.manual_seed)
            self.logger.info("Program will run on *****CPU***** ")

        self.l2h_G = self.l2h_G.to(self.device)
        self.l2h_D = self.l2h_D.to(self.device)
        self.h2l_G = self.h2l_G.to(self.device)
        self.h2l_D = self.h2l_D.to(self.device)
        self.criterion_GAN = self.criterion_GAN.to(self.device)
        self.criterion_MSE = self.criterion_MSE.to(self.device)

        # Summary Writer
        self.summary_writer_l2h = SummaryWriter(log_dir=self.config.summary_dir_l2h, comment='Low-To-High GAN')
        self.summary_writer_h2l = SummaryWriter(log_dir=self.config.summary_dir_h2l, comment='High-To-Low GAN')
        
    def load_checkpoint(self, file_name, model):
        if model == 'l2h':
            checkpoint_dir = self.config.checkpoint_l2h_dir
        elif model == 'h2l':
            checkpoint_dir = self.config.checkpoint_h2l_dir
        elif model == 'combined':
            checkpoint_dir = self.config.checkpoint_combined_dir
            
        filename = checkpoint_dir + file_name
        try:
            self.logger.info("Loading checkpoint '{}'".format(filename))
            checkpoint = torch.load(filename)
            
            self.current_epoch = checkpoint['epoch']
            self.current_iteration = checkpoint['iteration']
            self.manual_seed = checkpoint['manual_seed']
            
            if model == 'h2l':
                self.h2l_G.load_state_dict(checkpoint['h2l_G_state_dict'])
                self.h2l_optimG.load_state_dict(checkpoint['h2l_G_optimizer'])
                self.h2l_D.load_state_dict(checkpoint['h2l_D_state_dict'])
                self.h2l_optimD.load_state_dict(checkpoint['h2l_D_optimizer'])
                
            elif model == 'l2h':
                self.l2h_G.load_state_dict(checkpoint['l2h_G_state_dict'])
                self.l2h_optimG.load_state_dict(checkpoint['l2h_G_optimizer'])
                self.l2h_D.load_state_dict(checkpoint['l2h_D_state_dict'])
                self.l2h_optimD.load_state_dict(checkpoint['l2h_D_optimizer'])
            
            elif model == 'combined':
                self.h2l_G.load_state_dict(checkpoint['h2l_G_state_dict'])
                self.h2l_optimG.load_state_dict(checkpoint['h2l_G_optimizer'])
                self.h2l_D.load_state_dict(checkpoint['h2l_D_state_dict'])
                self.h2l_optimD.load_state_dict(checkpoint['h2l_D_optimizer'])
                
                self.l2h_G.load_state_dict(checkpoint['l2h_G_state_dict'])
                self.l2h_optimG.load_state_dict(checkpoint['l2h_G_optimizer'])
                self.l2h_D.load_state_dict(checkpoint['l2h_D_state_dict'])
                self.l2h_optimD.load_state_dict(checkpoint['l2h_D_optimizer'])

        except OSError:
            self.logger.info("No checkpoint exists from '{}'. Skipping...".format(checkpoint_dir))
            self.logger.info("**First time to train**")

    def save_checkpoint(self, file_name, model, is_best = 0):
        state = {
            'epoch': self.current_epoch,
            'iteration': self.current_iteration,
            'manual_seed': self.manual_seed
        }
        if model == 'l2h':
            state['l2g_G_state_dict'] = self.l2h_G.state_dict()
            state['l2h_G_optimizer'] = self.l2h_optimG.state_dict()
            state['l2h_D_state_dict'] = self.l2h_D.state_dict()
            state['l2h_D_optimizer'] = self.l2h_optimD.state_dict()
            
            checkpoint_dir = self.config.checkpoint_l2h_dir
        
        elif model == 'h2l':
            state['h2l_G_state_dict'] = self.h2l_G.state_dict()
            state['h2l_G_optimizer'] = self.h2l_optimG.state_dict()
            state['h2l_D_state_dict'] = self.h2l_D.state_dict()
            state['h2l_D_optimizer'] = self.h2l_optimD.state_dict()
            
            checkpoint_dir = self.config.checkpoint_h2l_dir
        
        elif model == 'combined':
            state['l2h_G_state_dict'] = self.l2h_G.state_dict()
            state['l2h_G_optimizer'] = self.l2h_optimG.state_dict()
            state['l2h_D_state_dict'] = self.l2h_D.state_dict()
            state['l2h_D_optimizer'] = self.l2h_optimD.state_dict()
            
            state['h2l_G_state_dict'] = self.h2l_G.state_dict()
            state['h2l_G_optimizer'] = self.h2l_optimG.state_dict()
            state['h2l_D_state_dict'] = self.h2l_D.state_dict()
            state['h2l_D_optimizer'] = self.h2l_optimD.state_dict()
            
            checkpoint_dir = self.config.checkpoint_combined_dir
        # Save the state
        torch.save(state, checkpoint_dir + file_name)
        # If it is the best copy it to another file 'model_best.pth.tar'
        if is_best:
            shutil.copyfile(checkpoint_dir + file_name,
                            checkpoint_dir + '_best.pth.tar')
            
    def run(self):
        """
        This function will the operator
        :return:
        """
        try:
            self.train()
        except KeyboardInterrupt:
            self.logger.info("You have entered CTRL+C.. Wait to finalize")

    def train(self):
        # Model Loading from the latest checkpoint if not found start from scratch.
        self.load_checkpoint(self.config.checkpoint_file_h2l, 'h2l')
        if self.current_epoch <= 200:
            self.load_checkpoint(self.config.checkpoint_file_l2h, 'l2h')
        elif self.current_epoch > 200:
            self.load_checkpoint(self.config.checkpoint_file_combined, 'combined')
        if self.current_epoch != 0 and self.current_epoch <= 200:
            self.logger.info("Checkpoint loaded successfully from '{}' and '{}' at (epoch {}) at (iteration {})\n"
                  .format(self.config.checkpoint_l2h_dir, self.config.checkpoint_h2l_dir, self.current_epoch, self.current_iteration))
        
        for epoch in range(self.current_epoch, self.config.max_epoch):
            self.current_epoch = epoch
            if epoch <= 200:
                self.train_one_epoch_h2l()
                self.train_one_epoch_l2h()
                self.save_checkpoint(self.config.checkpoint_file_l2h, 'l2h')
                self.save_checkpoint(self.config.checkpoint_file_h2l, 'h2l')
            else:
                self.train_one_epoch_combined()
                self.save_checkpoint(self.config.checkpoint_file_combined, 'combined')
            
    def to_var(self, data):
        real_cpu = data
        batchsize = real_cpu.size(0)
        inp = Variable(real_cpu.cuda())
        return inp, batchsize
    
    def train_one_epoch_h2l(self):
        test_loader = get_loader(self.config.HighToLow_hr_datapath, 
                                 self.config.HighToLow_lr_datapath, 
                                 self.config.batch_size)
        
        self.h2l_G.train()
        self.h2l_D.train()

        for curr_it, data_dict in enumerate(test_loader):
            data_low = data_dict['lr']
            data_high = data_dict['hr']
            data_input_low, batchsize = self.to_var(data_low)
            data_input_high, _ = self.to_var(data_high)
            
            y = torch.randn(data_low.size(0), )
            y, _ = self.to_var(y)
            
            ##################
            #  Train Generator
            ##################

            self.h2l_optimG.zero_grad()
    
            # Generate a high resolution image from low resolution input
            noise = torch.randn(data_high.size(0), 1)
            noise, _ = self.to_var(noise)
            gen_hr = self.h2l_G(data_input_high, noise)
    
            # Measure pixel-wise loss against ground truth
            loss_pixel = self.criterion_MSE(gen_hr, data_input_low)
            
            # Extract validity predictions from discriminator
            pred_real = self.h2l_D(data_input_high).detach()
            pred_fake = self.h2l_D(gen_hr)

            # Adversarial loss (relativistic average GAN)
            y.fill_(self.real_label)
            loss_G_GAN = self.criterion_GAN(pred_fake - pred_real.mean(0, keepdim=True), y)
            
            # Total generator loss
            loss_G = (self.config.beta * loss_G_GAN) + (self.config.alpha * loss_pixel)
            
            loss_G.backward(retain_graph=True)
            self.h2l_optimG.step()

            ######################
            #  Train Discriminator
            ######################
            
            self.h2l_optimD.zero_grad()

            # Adversarial loss for real and fake images (relativistic average GAN)
            pred_real = self.h2l_D(data_input_high)
            y.fill_(self.real_label)
            loss_D_real = self.criterion_GAN(pred_real - pred_fake.mean(0, keepdim=True), y)
            loss_D_real.backward(retain_graph=True)
            
            pred_fake = self.h2l_D(gen_hr.detach())
            y.fill_(self.fake_label)
            loss_D_fake = self.criterion_GAN(pred_fake - pred_real.mean(0, keepdim=True), y)
            loss_D_fake.backward()
            # Total loss
            loss_D = (loss_D_real + loss_D_fake) / 2
    
            #loss_D.backward()
            self.h2l_optimD.step()
            
            self.current_iteration += 1

            self.summary_writer_h2l.add_scalar("epoch/Generator_loss", loss_G.item(), self.current_iteration)
            self.summary_writer_h2l.add_scalar("epoch/Discriminator_loss_real", loss_D_real.item(), self.current_iteration)
            self.summary_writer_h2l.add_scalar("epoch/Discriminator_loss_fake", loss_D_fake.item(), self.current_iteration)
            
            path = os.path.join(self.test_file, 'batch' + str(curr_it) + '_epoch'+ str(self.current_epoch) + '_h2l.jpg')
            vutils.save_image(gen_hr.data, path, normalize=True)
            
            # --------------
            #  Log Progress
            # --------------
    
            self.logger.info(
                "High-To-Low GAN: [Epoch %d/%d] [Batch %d/%d] [D loss: %f, real: %f, fake: %f] [G loss: %f, adv: %f, pixel: %f]"
                % (
                    self.current_epoch + 1,
                    self.config.max_epoch,
                    curr_it + 1,
                    len(test_loader),
                    loss_D.item(),
                    loss_D_real.item(),
                    loss_D_fake.item(),
                    loss_G.item(),
                    loss_G_GAN.item(),
                    loss_pixel.item(),
                )
            )
    
    def train_one_epoch_l2h(self):
        test_loader = get_loader(self.config.LowToHigh_datapath, None, 
                                 self.config.batch_size)
        
        self.l2h_G.train()
        self.l2h_D.train()

        for curr_it, data_dict in enumerate(test_loader):
            data_low = data_dict['img16']
            data_high = data_dict['img64']
            data_input_low, batchsize = self.to_var(data_low)
            data_input_high, _ = self.to_var(data_high)
            
            y = torch.randn(data_low.size(0), )
            y, _ = self.to_var(y)
            
            ##################
            #  Train Generator
            ##################

            self.l2h_optimG.zero_grad()
    
            # Generate a high resolution image from low resolution input
            gen_hr = self.l2h_G(data_input_low)
    
            # Measure pixel-wise loss against ground truth
            loss_pixel = self.criterion_MSE(gen_hr, data_input_high)
            
            # Extract validity predictions from discriminator
            pred_real = self.l2h_D(data_input_high).detach()
            pred_fake = self.l2h_D(gen_hr)

            # Adversarial loss (relativistic average GAN)
            y.fill_(self.real_label)
            loss_G_GAN = self.criterion_GAN(pred_fake - pred_real.mean(0, keepdim=True), y)
            
            # Total generator loss
            loss_G = (self.config.beta * loss_G_GAN) + (self.config.alpha * loss_pixel)
            
            loss_G.backward(retain_graph=True)
            self.l2h_optimG.step()

            ######################
            #  Train Discriminator
            ######################
            
            self.l2h_optimD.zero_grad()

            # Adversarial loss for real and fake images (relativistic average GAN)
            pred_real = self.l2h_D(data_input_high)
            y.fill_(self.real_label)
            loss_D_real = self.criterion_GAN(pred_real - pred_fake.mean(0, keepdim=True), y)
            loss_D_real.backward(retain_graph=True)
            
            pred_fake = self.l2h_D(gen_hr.detach())
            y.fill_(self.fake_label)
            loss_D_fake = self.criterion_GAN(pred_fake - pred_real.mean(0, keepdim=True), y)
            loss_D_fake.backward()
            # Total loss
            loss_D = (loss_D_real + loss_D_fake) / 2
    
            self.l2h_optimD.step()
            
            self.current_iteration += 1

            self.summary_writer_l2h.add_scalar("epoch/Generator_loss", loss_G.item(), self.current_iteration)
            self.summary_writer_l2h.add_scalar("epoch/Discriminator_loss_real", loss_D_real.item(), self.current_iteration)
            self.summary_writer_l2h.add_scalar("epoch/Discriminator_loss_fake", loss_D_fake.item(), self.current_iteration)
            self.summary_writer_l2h.add_scalar("epoch/Discriminator_loss", loss_D.item(), self.current_iteration)
            
            path = os.path.join(self.test_file, 'batch' + str(curr_it) + '_epoch'+ str(self.current_epoch) + '_l2h.jpg')
            vutils.save_image(gen_hr.data, path, normalize=True)
            
            # --------------
            #  Log Progress
            # --------------
    
            self.logger.info(
                "Low-To-High GAN: [Epoch %d/%d] [Batch %d/%d] [D loss: %f, real: %f, fake: %f] [G loss: %f, adv: %f, pixel: %f]"
                % (
                    self.current_epoch + 1,
                    self.config.max_epoch,
                    curr_it,
                    len(test_loader),
                    loss_D.item(),
                    loss_D_real.item(),
                    loss_D_fake.item(),
                    loss_G.item(),
                    loss_G_GAN.item(),
                    loss_pixel.item(),
                )
            )

    def train_one_epoch_combined(self):
        test_loader = get_loader(self.config.HighToLow_hr_datapath, 
                                 self.config.HighToLow_lr_datapath, 
                                 self.config.batch_size)
        
        self.h2l_G.train()
        self.h2l_D.train()
        self.l2h_G.train()
        self.l2h_D.train()

        for curr_it, data_dict in enumerate(test_loader):
            data_low = data_dict['lr']
            data_high = data_dict['hr']
            data_input_low, batchsize = self.to_var(data_low)
            data_input_high, _ = self.to_var(data_high)
            
            y = torch.randn(data_low.size(0), )
            y, _ = self.to_var(y)
            
            ##############################
            #  Train High-To-Low Generator
            ##############################

            self.h2l_optimG.zero_grad()
    
            # Generate a high resolution image from low resolution input
            noise = torch.randn(data_high.size(0), 1)
            noise, _ = self.to_var(noise)
            h2l_gen_hr = self.h2l_G(data_input_high, noise)
    
            # Measure pixel-wise loss against ground truth
            h2l_loss_pixel = self.criterion_MSE(h2l_gen_hr, data_input_low)
            
            # Extract validity predictions from discriminator
            h2l_pred_real = self.h2l_D(data_input_high).detach()
            h2l_pred_fake = self.h2l_D(h2l_gen_hr)

            # Adversarial loss (relativistic average GAN)
            y.fill_(self.real_label)
            h2l_loss_G_GAN = self.criterion_GAN(h2l_pred_fake - h2l_pred_real.mean(0, keepdim=True), y)
            
            # Total generator loss
            h2l_loss_G = (self.config.beta * h2l_loss_G_GAN) + (self.config.alpha * h2l_loss_pixel)
            
            h2l_loss_G.backward(retain_graph=True)
            self.h2l_optimG.step()

            ##################################
            #  Train High-To-Low Discriminator
            ##################################
            
            self.h2l_optimD.zero_grad()

            # Adversarial loss for real and fake images (relativistic average GAN)
            h2l_pred_real = self.h2l_D(data_input_high)
            y.fill_(self.real_label)
            h2l_loss_D_real = self.criterion_GAN(h2l_pred_real - h2l_pred_fake.mean(0, keepdim=True), y)
            h2l_loss_D_real.backward(retain_graph=True)
            
            h2l_pred_fake = self.h2l_D(h2l_gen_hr.detach())
            y.fill_(self.fake_label)
            h2l_loss_D_fake = self.criterion_GAN(h2l_pred_fake - h2l_pred_real.mean(0, keepdim=True), y)
            h2l_loss_D_fake.backward()
            # Total loss
            h2l_loss_D = (h2l_loss_D_real + h2l_loss_D_fake) / 2
    
            self.h2l_optimD.step()
            
            self.current_iteration += 1

            self.summary_writer_h2l.add_scalar("epoch/Generator_loss", h2l_loss_G.item(), self.current_iteration)
            self.summary_writer_h2l.add_scalar("epoch/Discriminator_loss_real", h2l_loss_D_real.item(), self.current_iteration)
            self.summary_writer_h2l.add_scalar("epoch/Discriminator_loss_fake", h2l_loss_D_fake.item(), self.current_iteration)
            self.summary_writer_h2l.add_scalar("epoch/Discriminator_loss", h2l_loss_D.item(), self.current_iteration)
            
            path = os.path.join(self.test_file, 'batch' + str(curr_it) + '_epoch'+ str(self.current_epoch) + '_combined_intermidiate.jpg')
            vutils.save_image(h2l_gen_hr.data, path, normalize=True)
            
            # --------------
            #  Log Progress
            # --------------
    
            self.logger.info(
                "Combined model: High-To-Low GAN: [Epoch %d/%d] [Batch %d/%d] [D loss: %f, real: %f, fake: %f] [G loss: %f, adv: %f, pixel: %f]"
                % (
                    self.current_epoch + 1,
                    self.config.max_epoch,
                    curr_it + 1,
                    len(test_loader),
                    h2l_loss_D.item(),
                    h2l_loss_D_real.item(),
                    h2l_loss_D_fake.item(),
                    h2l_loss_G.item(),
                    h2l_loss_G_GAN.item(),
                    h2l_loss_pixel.item(),
                )
            )
                
            data_input_low = h2l_gen_hr
            
            y = torch.randn(data_input_low.size(0), )
            y, _ = self.to_var(y)
            
            ##############################
            #  Train Low-To-High Generator
            ##############################

            self.l2h_optimG.zero_grad()
    
            # Generate a high resolution image from low resolution input
            l2h_gen_hr = self.l2h_G(data_input_low)
    
            # Measure pixel-wise loss against ground truth
            l2h_loss_pixel = self.criterion_MSE(l2h_gen_hr, data_input_high)
            
            # Extract validity predictions from discriminator
            l2h_pred_real = self.l2h_D(data_input_high).detach()
            l2h_pred_fake = self.l2h_D(l2h_gen_hr)

            # Adversarial loss (relativistic average GAN)
            y.fill_(self.real_label)
            l2h_loss_G_GAN = self.criterion_GAN(l2h_pred_fake - l2h_pred_real.mean(0, keepdim=True), y)
            
            # Total generator loss
            l2h_loss_G = (self.config.beta * l2h_loss_G_GAN) + (self.config.alpha * l2h_loss_pixel)
            
            l2h_loss_G.backward(retain_graph=True)
            self.l2h_optimG.step()

            ##################################
            #  Train Low-To-High Discriminator
            ##################################
            
            self.l2h_optimD.zero_grad()

            # Adversarial loss for real and fake images (relativistic average GAN)
            l2h_pred_real = self.l2h_D(data_input_high)
            y.fill_(self.real_label)
            l2h_loss_D_real = self.criterion_GAN(l2h_pred_real - l2h_pred_fake.mean(0, keepdim=True), y)
            l2h_loss_D_real.backward(retain_graph=True)
            
            l2h_pred_fake = self.l2h_D(l2h_gen_hr.detach())
            y.fill_(self.fake_label)
            l2h_loss_D_fake = self.criterion_GAN(l2h_pred_fake - l2h_pred_real.mean(0, keepdim=True), y)
            l2h_loss_D_fake.backward()
            # Total loss
            l2h_loss_D = (l2h_loss_D_real + l2h_loss_D_fake) / 2
    
            self.l2h_optimD.step()
            
            self.current_iteration += 1

            self.summary_writer_l2h.add_scalar("epoch/Generator_loss", l2h_loss_G.item(), self.current_iteration)
            self.summary_writer_l2h.add_scalar("epoch/Discriminator_loss_real", l2h_loss_D_real.item(), self.current_iteration)
            self.summary_writer_l2h.add_scalar("epoch/Discriminator_loss_fake", l2h_loss_D_fake.item(), self.current_iteration)
            self.summary_writer_l2h.add_scalar("epoch/Discriminator_loss", l2h_loss_D.item(), self.current_iteration)
            
            path = os.path.join(self.test_file, 'batch' + str(curr_it) + '_epoch'+ str(self.current_epoch) + '_combined_final.jpg')
            vutils.save_image(l2h_gen_hr.data, path, normalize=True)
            
            # --------------
            #  Log Progress
            # --------------
    
            self.logger.info(
                "Combined model: Low-To-High GAN: [Epoch %d/%d] [Batch %d/%d] [D loss: %f, real: %f, fake: %f] [G loss: %f, adv: %f, pixel: %f]"
                % (
                    self.current_epoch + 1,
                    self.config.max_epoch,
                    curr_it,
                    len(test_loader),
                    l2h_loss_D.item(),
                    l2h_loss_D_real.item(),
                    l2h_loss_D_fake.item(),
                    l2h_loss_G.item(),
                    l2h_loss_G_GAN.item(),
                    l2h_loss_pixel.item(),
                )
            )


    def validate(self):
        pass

    def finalize(self):
        """
        Finalize all the operations of the 2 Main classes of the process the operator and the data loader
        :return:
        """
        self.logger.info("Please wait while finalizing the operation.. Thank you")
        self.save_checkpoint()
        self.summary_writer.export_scalars_to_json("{}all_scalars.json".format(self.config.summary_dir))
        self.summary_writer.close()
        self.dataloader.finalize()
        
if __name__ == "__main__":
    config_dir = config.process_config('configurations/train_config.json')
    gan = Combined_GAN(config_dir)
