import torch
import torch.nn as nn
from options import HiDDenConfiguration
from model.conv_bn_relu import ConvBNRelu


class Encoder(nn.Module):
    """
    Inserts a watermark into an image.
    """
    def __init__(self, config: HiDDenConfiguration):
        super(Encoder, self).__init__()
        self.H = config.H
        self.W = config.W
        self.conv_channels = config.encoder_channels
        self.num_blocks = config.encoder_blocks

        layers = [ConvBNRelu(3, self.conv_channels)]

        for _ in range(config.encoder_blocks-2):
            layer = ConvBNRelu(self.conv_channels, self.conv_channels)
            layers.append(layer)

        self.conv_layers = nn.Sequential(*layers)
        self.after_concat_layer = ConvBNRelu(self.conv_channels + 3 + config.message_length,
                                             self.conv_channels)

        self.final_layer = nn.Conv2d(self.conv_channels, 3, kernel_size=1)

        self.m=ConvBNRelu(self.conv_channels, self.conv_channels)
        self.v=ConvBNRelu(self.conv_channels, self.conv_channels)
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var/2)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, image, message,val_tag):
        mu=None
        log_var=None
        # First, add two dummy dimensions in the end of the message.
        # This is required for the .expand to work correctly
        expanded_message = message.unsqueeze(-1)
        expanded_message.unsqueeze_(-1)

        expanded_message = expanded_message.expand(-1,-1, self.H, self.W)
        encoded_image = self.conv_layers(image)
        # concatenate expanded message and image

        mu=self.m(encoded_image)
        log_var=self.v(encoded_image)

        encoded_image=mu
        concat = torch.cat([expanded_message, encoded_image, image], dim=1)
        im_w = self.after_concat_layer(concat)
        im_w = self.final_layer(im_w)

        mu=torch.flatten(mu, 1)
        log_var=torch.flatten(log_var, 1)
        noise_encoded_image = self.reparameterize(mu, log_var).view(-1,self.conv_channels,self.H,self.W)
        
        
        concat = torch.cat([expanded_message, noise_encoded_image, image], dim=1)
        im_n = self.after_concat_layer(concat)
        im_n = self.final_layer(im_n)


        
        return im_w,im_n,mu,log_var
