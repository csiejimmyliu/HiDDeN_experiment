import torch.nn as nn
from model.encoder import Encoder
from model.decoder import Decoder
from options import HiDDenConfiguration
from noise_layers.noiser import Noiser
import torch
import random

class EncoderDecoder(nn.Module):
    """
    Combines Encoder->Noiser->Decoder into single pipeline.
    The input is the cover image and the watermark message. The module inserts the watermark into the image
    (obtaining encoded_image), then applies Noise layers (obtaining noised_image), then passes the noised_image
    to the Decoder which tries to recover the watermark (called decoded_message). The module outputs
    a three-tuple: (encoded_image, noised_image, decoded_message)
    """
    def __init__(self, config: HiDDenConfiguration, noiser: Noiser):

        super(EncoderDecoder, self).__init__()
        self.encoder = Encoder(config)
        self.noiser = noiser

        self.decoder = Decoder(config)
        self.valtag=False
    
    def set_val_true(self):
        self.valtag=True
    def set_val_false(self):
        self.valtag=False

    def forward(self, image, message):
        encoded_image,noised_image,mu,log_var = self.encoder(image, message,val_tag=self.valtag)
        
        
        if self.valtag==True:
            #validation
            noised_and_cover = self.noiser([encoded_image, image])
            noised_image = noised_and_cover[0]
        
        '''
        encoded_image=noised_image.clone().detach()
        noised_and_cover = self.noiser([encoded_image, image])
        noised_image = noised_and_cover[0]
        '''
        if self.valtag==False:
            if random.randint(0, 1)==0:
                noised_image=encoded_image

            

        decoded_message = self.decoder(noised_image)
        return encoded_image, noised_image, decoded_message,mu,log_var
