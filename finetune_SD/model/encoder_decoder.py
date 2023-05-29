import torch.nn as nn
from model.encoder import Encoder
from model.decoder import Decoder
from options import HiDDenConfiguration
from noise_layers.noiser import Noiser
import torch

from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler,AutoencoderKL
import torch

from noise_layers.jpeg_compression import rgb2yuv_tensor
from diffusers.models.vae import DecoderOutput

model_id = "runwayml/stable-diffusion-v1-5"

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
        #pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
        #pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        self.encoder = AutoencoderKL.from_pretrained(model_id, subfolder='vae',torch_dtype=torch.float32)
        self.noiser = noiser
        self.alpha=config.alpha
        self.decoder = Decoder(config)
        '''
        self.encoder.encoder.requires_grad_(False)
        self.encoder.quant_conv.requires_grad_(False)
        self.decoder.requires_grad_(False)
        '''
        '''
        for name, param in self.encoder.decoder.named_parameters():
            if 'norm' in name:
                param.requires_grad=False
        '''
        '''
        for module in self.encoder.modules():
            # print(module)
            if isinstance(module, nn.GroupNorm):
                if hasattr(module, 'weight'):
                    module.weight.requires_grad_(False)
                if hasattr(module, 'bias'):
                    module.bias.requires_grad_(False)
                module.eval()
        '''

    
    def vae_forward(self,vae,sample):
        x = sample
        posterior = vae.encode(x).latent_dist
        z = posterior.mode()
        dec = vae.decode(z.clone().detach()).sample
        return DecoderOutput(sample=dec)
    
    def forward(self, image, message):
        #encoded_image = self.encoder(image, message)
        #watermark_noise=self.encoder(image).sample
        watermark_noise=self.vae_forward(self.encoder,image).sample
        #watermark_noise=torch.tanh(watermark_noise)
        #encoded_image=image+self.alpha*watermark_noise
        encoded_image=watermark_noise


        #to [-1,1] yuv
        yuv_encoded_image=encoded_image*2.0-1.0
        yuv_encoded_image=rgb2yuv_tensor(yuv_encoded_image)

        yuv_image=image*2.0-1.0
        yuv_image=rgb2yuv_tensor(yuv_image)
        
        noised_and_cover = self.noiser([yuv_encoded_image, yuv_image])
        noised_image = noised_and_cover[0]
        decoded_message = self.decoder(noised_image)
        return encoded_image, noised_image, decoded_message
