from __future__ import print_function, division

from keras.layers import Input, Dense, Reshape, Flatten
from keras.layers import BatchNormalization, Convolution2D, AveragePooling2D, ReLU, Conv2DTranspose, UpSampling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
from keras.optimizers import RMSprop
from keras import losses, backend

import tensorflow as tf

import matplotlib.pyplot as plt
from random import sample
import os
from PIL import Image
from glob import glob
import numpy as np

class GAN():
    def __init__(self):
        
        # Directory where the low resolution and high resolution images are stored
        self.basepath = 'F:/Study/2nd_Semester/CV/Project/'
        self.LR_dir = self.basepath + 'wider_lnew/'
        self.HR_dir = self.basepath + 'celeba-dataset/img_align_celeba/'

        # Shape of real high resolution image
        self.real_img_rows = 218
        self.real_img_cols = 178
        self.real_channels = 3
        self.real_img_shape = (self.real_img_rows, self.real_img_cols, self.real_channels)

        # Shape of generated/ real low resolution image
        self.gen_img_rows = 16
        self.gen_img_cols = 16
        self.gen_channels = 3
        self.gen_img_shape = (self.gen_img_rows, self.gen_img_cols, self.gen_channels)
        
        optimizer = RMSprop(lr=0.0002, rho=0.9)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss=losses.binary_crossentropy, optimizer=optimizer, metrics=['accuracy'])

        # Build and compile the generator
        self.generator = self.build_generator()
        self.generator.compile(loss=losses.binary_crossentropy, optimizer=optimizer)

        # The generator takes high resolution image as input 
        z = Input(shape=self.real_img_shape)
        generated_img = self.generator(z)

        # For the GAN model we will only train the generator
        self.discriminator.trainable = False

        # valid takes generated images as input and determines validity
        valid = self.discriminator(generated_img)

        # The GAN model (stacked generator and discriminator) takes
        # high resolution image as input => generates low resolution images => determines validity 
        self.GAN = Model(z, valid)
        self.GAN.compile(loss='binary_crossentropy', optimizer=optimizer)
    
    def conv3x3(self, outC, stride=1):
        # 3x3 convolution with padding
        return Convolution2D(outC, kernel_size=(3,3), padding='same', use_bias=False)
    
    def build_resid_block(self, outC, operation, bn = False):
        block = Sequential()
        
        if bn:
            block.add(BatchNormalization(momentum=0.8))
        block.add(ReLU())
        if operation == 'downsample':
            block.add(self.conv3x3(outC))
        elif operation == 'upsample':
             Conv2DTranspose(outC, kernel_size=(4,4), strides = 2, padding = 'same')
            
        if bn:
            block.add(BatchNormalization(momentum=0.8))
        block.add(ReLU())
        if operation == 'downsample':
            block.add(AveragePooling2D(2,2))
            block.add(self.conv3x3(outC))
        elif operation == 'upsample':
            block.add(UpSampling2D(size = (2,2)))
            block.add(self.conv3x3(outC))
        
        return block#Model(inp, out)
    
    def build_generator(self):
        model = Sequential()
        
        # Input layer        
        model.add(Flatten(input_shape=self.real_img_shape))
        
        model.add(Reshape((218, 178, 3), input_shape=(218 * 178 * 3,)))
        model.add(self.conv3x3(256))
        model.add(self.build_resid_block(64, 'downsample'))
        model.add(self.build_resid_block(64, 'downsample'))
        model.add(self.build_resid_block(64, 'downsample'))
        model.add(self.build_resid_block(64, 'downsample'))
        model.add(self.build_resid_block(64, 'upsample'))
        model.add(self.build_resid_block(64, 'upsample'))
        model.add(self.conv3x3(3))
        model.add(Flatten())
        # Output layer
        model.add(Dense(np.prod(self.gen_img_shape), activation='tanh'))
        
        model.add(Reshape(self.gen_img_shape))

        model.summary()
        
        # Taking input and output of the model 
        HR_Image = Input(shape=self.real_img_shape)
        gen_LR_Image = model(HR_Image)

        return Model(HR_Image, gen_LR_Image)


    def build_discriminator(self):
        model = Sequential()
        
        # Input layer
        model.add(Flatten(input_shape=self.gen_img_shape))
        
        # 1st Fully connected hidden layer
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        
        # 2nd Fully connected hidden layer
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        
        # Output layer
        model.add(Dense(1, activation='sigmoid'))
        
        model.summary()
        
        # Taking input and output of the model
        gen_LR_Image = Input(shape=self.gen_img_shape)
        validity = model(gen_LR_Image)

        return Model(gen_LR_Image, validity)
    
    
    def get_image(self, image_path, mode):
        image = Image.open(image_path)
        return np.array(image.convert(mode))
    
    
    def get_batch(self, image_files, mode):
        data_batch = np.array(
            [self.get_image(sample_file, mode) for sample_file in image_files])
        return data_batch    
    
    
    def train(self, epochs, batch_size=128, save_interval=50):
        
        # Input the low resolution images from the directory
        X_train = self.get_batch(sample(glob(os.path.join(self.LR_dir, '*.jpg')), 5000), 'RGB')

        #Rescale -1 to 1
        X_train = (X_train.astype(np.float32) - 127.5) / 127.5

        half_batch = int(batch_size / 2)

        # Array initialization for logging of the losses
        d_loss_logs_r = []
        d_loss_logs_f = []
        g_loss_logs = []
        
        config = tf.ConfigProto(device_count = {'GPU': 1 , 'CPU': 56} )
        config.gpu_options.allow_growth = True
        session = tf.Session(config=config)
        backend.set_session(session)

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half the batch size of high resolution images
            idx = np.random.randint(0, X_train.shape[0], half_batch)
            real_LR_Images = X_train[idx]
            HR_Images = self.get_batch(sample(glob(os.path.join(self.HR_dir, '*.jpg')),half_batch), 'RGB')
            
            # Generate a half batch of new low resolution images
            gen_LR_Images = self.generator.predict(HR_Images)

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(real_LR_Images, np.ones((half_batch, 1)))
            d_loss_fake = self.discriminator.train_on_batch(gen_LR_Images, np.zeros((half_batch, 1)))
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            HR_Images = self.get_batch(sample(glob(os.path.join(self.HR_dir, '*.jpg')),batch_size), 'RGB')
            
            # The generator wants the discriminator to label the generated samples as valid (ones)
            valid_y = np.array([1] * batch_size)

            # Train the generator
            g_loss = self.GAN.train_on_batch(HR_Images, valid_y)

            # Print the progress
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

            # Store the losses
            d_loss_logs_r.append([epoch, d_loss[0]])
            d_loss_logs_f.append([epoch, d_loss[1]])
            g_loss_logs.append([epoch, g_loss])

            # If at save interval => save generated low resolution images of some samples high resolution images
            if epoch % save_interval == 0:
                self.save_imgs(epoch)
        
        self.save_imgs(epochs)
        
        d_loss_logs_r = np.array(d_loss_logs_r)
        d_loss_logs_f = np.array(d_loss_logs_f)
        g_loss_logs = np.array(g_loss_logs)

        # At the end of training plot the losses vs epochs
        plt.plot(d_loss_logs_r[:,0], d_loss_logs_r[:,1], label="Discriminator Loss - Real")
        plt.plot(d_loss_logs_f[:,0], d_loss_logs_f[:,1], label="Discriminator Loss - Fake")
        plt.plot(g_loss_logs[:,0], g_loss_logs[:,1], label="Generator Loss")
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('GAN')
        plt.grid(True)
        plt.show() 
        plt.savefig('gen/loss_graph.png')
        
    def save_imgs(self, epoch):
        r, c = 5, 5
        HR_Images = self.get_batch(sample(glob(os.path.join(self.HR_dir, '*.jpg')),r * c), 'RGB')
        
        gen_LR_Images = self.generator.predict(HR_Images)

        # Rescale images 0 - 1
        gen_LR_Images = (1/2.5) * gen_LR_Images + 0.5
        
        # Storing original high resolution images
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(HR_Images[cnt, :,:,:])
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig(self.basepath + "gen/orig_%d.png" % epoch, bbox_inches="tight")
        plt.close()
        
        # Storing generated low resolution images
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_LR_Images[cnt, :,:,:])
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig(self.basepath + "gen/%d.png" % epoch, bbox_inches="tight")
        plt.close()

if __name__ == '__main__':
    gan = GAN()
    gan.train(epochs = 1000, batch_size = 32, save_interval = 50)