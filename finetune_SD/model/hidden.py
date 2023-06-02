import numpy as np
import torch
import torch.nn as nn

from options import HiDDenConfiguration
from model.discriminator import Discriminator
from model.encoder_decoder import EncoderDecoder
from vgg_loss import VGGLoss
from noise_layers.noiser import Noiser
import imgaug.augmenters as iaa
import matplotlib.pyplot as plt
import torchvision
from model.lamb import Lamb

import math
from PerceptualSimilarity.src.loss.loss_provider import LossProvider
from noise_layers.jpeg_compression import yuv2rgb_tensor
import torch


def combine_linear_layer(l1,l2):
    w1=l1.weight
    b1=l1.bias
    w2=l2.weight
    b2=l2.bias

    w3=w2@w1
    b3=w2@b1+b2

    return nn.Parameter(w3),nn.Parameter(b3)

class Hidden:
    def __init__(self, configuration: HiDDenConfiguration, device: torch.device, noiser: Noiser, tb_logger,train_options,w_path):
        """
        :param configuration: Configuration for the net, such as the size of the input image, number of channels in the intermediate layers, etc.
        :param device: torch.device object, CPU or GPU
        :param noiser: Object representing stacked noise layers.
        :param tb_logger: Optional TensorboardX logger object, if specified -- enables Tensorboard logging
        """
        super(Hidden, self).__init__()
        self.accu_step=configuration.accu_step
        warm_up_iter = int(train_options.number_of_epochs*configuration.data_len/(float(train_options.batch_size*self.accu_step)*5.0))
        lr_max = 1e-4
        lr_min = 1e-6
        T_max=int(train_options.number_of_epochs*configuration.data_len/float(train_options.batch_size*self.accu_step))
        self.encoder_decoder = EncoderDecoder(configuration, noiser).to(device)
        #self.discriminator = Discriminator(configuration).to(device)
        
        #self.optimizer_discrim = torch.optim.Adam(self.discriminator.parameters())

        self.encoder_decoder.eval()
        self.encoder_decoder.encoder.encoder.requires_grad_(False)
        self.encoder_decoder.encoder.quant_conv.requires_grad_(False)
        self.encoder_decoder.decoder.requires_grad_(False)
        
        for module in self.encoder_decoder.encoder.modules():
            # print(module)
            if isinstance(module, nn.GroupNorm):
                if hasattr(module, 'weight'):
                    module.weight.requires_grad_(False)
                if hasattr(module, 'bias'):
                    module.bias.requires_grad_(False)
                #module.eval()
        

        self.whitening_layer=torch.load(w_path).to(device)
        self.whitening_layer.eval()
        self.whitening_layer.requires_grad_(False)
        

        #w_temp,b_temp=combine_linear_layer(self.encoder_decoder.decoder.linear,self.whitening_layer)
        



        if configuration.opt_type=="adam":
            self.optimizer_enc_dec = torch.optim.Adam(filter(lambda p: p.requires_grad, self.encoder_decoder.encoder.parameters()))
        else:
            self.optimizer_enc_dec = Lamb(filter(lambda p: p.requires_grad, self.encoder_decoder.encoder.parameters()))
        

        lambda0 = lambda cur_iter: cur_iter / warm_up_iter if  cur_iter < warm_up_iter else \
        (lr_min + 0.5*(lr_max-lr_min)*(1.0+math.cos( (cur_iter-warm_up_iter)/(T_max-warm_up_iter)*math.pi)))/lr_max
        self.scheduler=torch.optim.lr_scheduler.LambdaLR(self.optimizer_enc_dec, lr_lambda=lambda0)
        self.scheduler.last_epoch=int((train_options.start_epoch-1)*configuration.data_len/float(train_options.batch_size))-1


        if configuration.use_vgg:
            self.vgg_loss = VGGLoss(3, 1, False)
            self.vgg_loss.to(device)
        else:
            self.vgg_loss = None

        self.config = configuration
        self.device = device


        self.bce_with_logits_loss = nn.BCEWithLogitsLoss().to(device)
        self.mse_loss = nn.MSELoss().to(device)
        self.perceptual_loss=LossProvider().get_loss_function('watson-vgg', colorspace='RGB', pretrained=True, reduction='sum').to(device)

        if configuration.loss_type=="mse":
            self.message_loss=self.mse_loss
        else:
            self.message_loss=self.bce_with_logits_loss
        # Defined the labels used for training the discriminator/adversarial loss
        self.cover_label = 1
        self.encoded_label = 0

        self.tb_logger = tb_logger
        if tb_logger is not None:
            from tensorboard_logger import TensorBoardLogger
            #encoder_final = self.encoder_decoder.encoder._modules['final_layer']
            #encoder_final.weight.register_hook(tb_logger.grad_hook_by_name('grads/encoder_out'))
            #decoder_final = self.encoder_decoder.decoder._modules['linear']
            #decoder_final.weight.register_hook(tb_logger.grad_hook_by_name('grads/decoder_out'))
            #discrim_final = self.discriminator._modules['linear']
            #discrim_final.weight.register_hook(tb_logger.grad_hook_by_name('grads/discrim_out'))

        self.optimizer_enc_dec.zero_grad()

    def train_on_batch(self, batch: list):
        """
        Trains the network on a single batch consisting of images and messages
        :param batch: batch of training data, in the form [images, messages]
        :return: dictionary of error metrics from Encoder, Decoder, and Discriminator on the current batch
        """
        images, messages,step = batch

        batch_size = images.shape[0]
        self.encoder_decoder.eval()
        
        self.encoder_decoder.encoder.decoder.train()
        #self.encoder_decoder.encoder.encoder.requires_grad_(False)
        #self.encoder_decoder.encoder.quant_conv.requires_grad_(False)


        #self.encoder_decoder.decoder.eval()
        for module in self.encoder_decoder.encoder.modules():
            # print(module)
            if isinstance(module, nn.GroupNorm):
                if hasattr(module, 'weight'):
                    module.weight.requires_grad_(False)
                if hasattr(module, 'bias'):
                    module.bias.requires_grad_(False)
                module.eval()
        
        for module in self.encoder_decoder.encoder.modules():
            # print(module)
            if isinstance(module, nn.Dropout):
                module.eval()
        

        '''
        for name, param in self.encoder_decoder.encoder.decoder.named_parameters():
            
            if param.requires_grad==False:
                print(name)
        
        raise NotImplementedError
        '''
        #self.discriminator.train()
        #with torch.enable_grad():
        # ---------------- Train the discriminator -----------------------------
        #self.optimizer_discrim.zero_grad()
        # train on cover
        #d_target_label_cover = torch.full((batch_size, 1), self.cover_label, device=self.device,dtype=torch.float32)
        #d_target_label_encoded = torch.full((batch_size, 1), self.encoded_label, device=self.device,dtype=torch.float32)
        #g_target_label_encoded = torch.full((batch_size, 1), self.cover_label, device=self.device,dtype=torch.float32)

        #d_on_cover = self.discriminator(images)
        #d_loss_on_cover = self.bce_with_logits_loss(d_on_cover, d_target_label_cover)
        #d_loss_on_cover.backward()

        
        # train on fake
        encoded_images, noised_images, decoded_messages = self.encoder_decoder(images, messages)
        #decoded_messages=self.whitening_layer(decoded_messages)/2+0.5
        decoded_messages=self.whitening_layer(decoded_messages)
        #decoded_messages=(decoded_messages-0.5)/6.+0.5
        '''
        for name, param in self.encoder_decoder.encoder.encoder.named_parameters():
            if param.requires_grad==True:
                
                print('encoder',name)

        for name, param in self.encoder_decoder.encoder.decoder.named_parameters():
            if param.requires_grad==True:
                print('decoder',name)
        '''
        
        #d_on_encoded = self.discriminator(encoded_images.detach())
        #d_loss_on_encoded = self.bce_with_logits_loss(d_on_encoded, d_target_label_encoded)

        #d_loss_on_encoded.backward()
        #self.optimizer_discrim.step()

        # --------------Train the generator (encoder-decoder) ---------------------
        
        # target label for encoded images should be 'cover', because we want to fool the discriminator
        #d_on_encoded_for_enc = self.discriminator(encoded_images)
        #g_loss_adv = self.bce_with_logits_loss(d_on_encoded_for_enc, g_target_label_encoded)
        
        
        if self.vgg_loss == None:
            g_loss_enc = self.perceptual_loss(encoded_images, images)/batch_size
            #g_loss_enc = self.mse(encoded_images, images)
        else:
            vgg_on_cov = self.vgg_loss(images)
            vgg_on_enc = self.vgg_loss(encoded_images)
            g_loss_enc = self.mse_loss(vgg_on_cov, vgg_on_enc)
        
        g_loss_dec = self.message_loss(decoded_messages, messages)
        g_loss =  self.config.encoder_loss * g_loss_enc + self.config.decoder_loss * g_loss_dec

        (g_loss/self.accu_step).backward()

        if step%self.accu_step==0:
            self.optimizer_enc_dec.step()
            self.optimizer_enc_dec.zero_grad()
            self.scheduler.step()
        

        decoded_rounded = decoded_messages.detach().cpu().numpy().round().clip(0, 1)
        bitwise_avg_err = np.sum(np.abs(decoded_rounded - messages.detach().cpu().numpy())) / (
                batch_size * messages.shape[1])

        losses = {
            'loss           ': g_loss.detach().item(),
            'encoder_mse    ': g_loss_enc.detach().item(),
            'dec_mse        ': g_loss_dec.detach().item(),
            'bitwise-error  ': bitwise_avg_err,
            #'adversarial_bce': g_loss_adv.item(),
            #'discr_cover_bce': d_loss_on_cover.item(),
            #'discr_encod_bce': d_loss_on_encoded.item()
        }
        return losses, (encoded_images, noised_images, decoded_messages)

    def validate_on_batch(self, batch: list):
        """
        Runs validation on a single batch of data consisting of images and messages
        :param batch: batch of validation data, in form [images, messages]
        :return: dictionary of error metrics from Encoder, Decoder, and Discriminator on the current batch
        """
        # if TensorboardX logging is enabled, save some of the tensors.
        

        images, messages = batch

        batch_size = images.shape[0]

        self.encoder_decoder.eval()
        #self.discriminator.eval()
        with torch.no_grad():
            #d_target_label_cover = torch.full((batch_size, 1), self.cover_label, device=self.device,dtype=torch.float32)
            #d_target_label_encoded = torch.full((batch_size, 1), self.encoded_label, device=self.device,dtype=torch.float32)
            #g_target_label_encoded = torch.full((batch_size, 1), self.cover_label, device=self.device,dtype=torch.float32)

            #d_on_cover = self.discriminator(images)
            #d_loss_on_cover = self.bce_with_logits_loss(d_on_cover, d_target_label_cover)

            encoded_images, noised_images, decoded_messages = self.encoder_decoder(images, messages)
            #decoded_messages=self.whitening_layer(decoded_messages)/2+0.5
            decoded_messages=self.whitening_layer(decoded_messages)
            #decoded_messages=(decoded_messages-0.5)/6.+0.5
            #d_on_encoded = self.discriminator(encoded_images)
            #d_loss_on_encoded = self.bce_with_logits_loss(d_on_encoded, d_target_label_encoded)

            #d_on_encoded_for_enc = self.discriminator(encoded_images)
            #g_loss_adv = self.bce_with_logits_loss(d_on_encoded_for_enc, g_target_label_encoded)
            
            if self.vgg_loss is None:
                g_loss_enc = self.perceptual_loss(encoded_images, images)/batch_size
                #g_loss_enc = self.mse(encoded_images, images)
            else:
                vgg_on_cov = self.vgg_loss(images)
                vgg_on_enc = self.vgg_loss(encoded_images)
                g_loss_enc = self.mse_loss(vgg_on_cov, vgg_on_enc)
            
            g_loss_dec = self.message_loss(decoded_messages, messages)
            g_loss =  self.config.encoder_loss * g_loss_enc + self.config.decoder_loss * g_loss_dec

        decoded_rounded = decoded_messages.detach().cpu().numpy().round().clip(0, 1)
        bitwise_avg_err = np.sum(np.abs(decoded_rounded - messages.detach().cpu().numpy())) / (
                batch_size * messages.shape[1])

        losses = {
            'loss           ': g_loss.detach().item(),
            'encoder_mse    ': g_loss_enc.detach().item(),
            'dec_mse        ': g_loss_dec.detach().item(),
            'bitwise-error  ': bitwise_avg_err,
            #'adversarial_bce': g_loss_adv.item(),
            #'discr_cover_bce': d_loss_on_cover.item(),
            #'discr_encod_bce': d_loss_on_encoded.item()
        }
        return losses, (encoded_images, noised_images, decoded_messages)

    def validate_on_batch_jpeg(self, batch: list,q):
        """
        Runs validation on a single batch of data consisting of images and messages
        :param batch: batch of validation data, in form [images, messages]
        :return: dictionary of error metrics from Encoder, Decoder, and Discriminator on the current batch
        """
        # if TensorboardX logging is enabled, save some of the tensors.
        if self.tb_logger is not None:
            encoder_final = self.encoder_decoder.encoder._modules['final_layer']
            self.tb_logger.add_tensor('weights/encoder_out', encoder_final.weight)
            decoder_final = self.encoder_decoder.decoder._modules['linear']
            self.tb_logger.add_tensor('weights/decoder_out', decoder_final.weight)
            discrim_final = self.discriminator._modules['linear']
            self.tb_logger.add_tensor('weights/discrim_out', discrim_final.weight)

        images, messages = batch

        batch_size = images.shape[0]
        aug=iaa.JpegCompression(compression=(q, q))
        self.encoder_decoder.eval()
        self.discriminator.eval()
        with torch.no_grad():
            d_target_label_cover = torch.full((batch_size, 1), self.cover_label, device=self.device,dtype=torch.float32)
            d_target_label_encoded = torch.full((batch_size, 1), self.encoded_label, device=self.device,dtype=torch.float32)
            g_target_label_encoded = torch.full((batch_size, 1), self.cover_label, device=self.device,dtype=torch.float32)

            d_on_cover = self.discriminator(images)
            d_loss_on_cover = self.bce_with_logits_loss(d_on_cover, d_target_label_cover)

            encoded_images, noised_images, decoded_messages = self.encoder_decoder(images, messages)
            
            
            

           
            temp_encoded_images=encoded_images.clone().detach()
            
            temp_encoded_images=np.clip(((temp_encoded_images[0].permute(1,2,0).cpu().numpy()+1)*255/2),0,255).astype(np.uint8)
            noised_images=aug.augment_image(temp_encoded_images)
            

            noised_images=torch.tensor(noised_images,dtype=torch.float).permute(2,0,1)/255*2-1
            
            decoded_messages=self.encoder_decoder.decoder(noised_images.unsqueeze(0).to(self.device))

            


            d_on_encoded = self.discriminator(encoded_images)
            d_loss_on_encoded = self.bce_with_logits_loss(d_on_encoded, d_target_label_encoded)

            d_on_encoded_for_enc = self.discriminator(encoded_images)
            g_loss_adv = self.bce_with_logits_loss(d_on_encoded_for_enc, g_target_label_encoded)

            if self.vgg_loss is None:
                g_loss_enc = self.mse_loss(encoded_images, images)
            else:
                vgg_on_cov = self.vgg_loss(images)
                vgg_on_enc = self.vgg_loss(encoded_images)
                g_loss_enc = self.mse_loss(vgg_on_cov, vgg_on_enc)

            g_loss_dec = self.mse_loss(decoded_messages, messages)
            g_loss = self.config.adversarial_loss * g_loss_adv + self.config.encoder_loss * g_loss_enc \
                     + self.config.decoder_loss * g_loss_dec

        decoded_rounded = decoded_messages.detach().cpu().numpy().round().clip(0, 1)
        bitwise_avg_err = np.sum(np.abs(decoded_rounded - messages.detach().cpu().numpy())) / (
                batch_size * messages.shape[1])

        losses = {
            'loss           ': g_loss.item(),
            'encoder_mse    ': g_loss_enc.item(),
            'dec_mse        ': g_loss_dec.item(),
            'bitwise-error  ': bitwise_avg_err,
            'adversarial_bce': g_loss_adv.item(),
            'discr_cover_bce': d_loss_on_cover.item(),
            'discr_encod_bce': d_loss_on_encoded.item()
        }
        return losses, (encoded_images, noised_images, decoded_messages)

    def to_stirng(self):
        return '{}'.format(str(self.encoder_decoder))
