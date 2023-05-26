import torch.nn as nn
from model.encoder import Encoder
from model.decoder import Decoder
from options import HiDDenConfiguration
from noise_layers.noiser import Noiser
import torch

from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import torch

from noise_layers.jpeg_compression import rgb2yuv_tensor

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
        pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        self.encoder = pipe.vae
        self.noiser = noiser
        self.alpha=config.alpha
        self.decoder = Decoder(config)
        self.encoder.encoder.requires_grad_(False)
        self.encoder.quant_conv.requires_grad_(False)

    def forward(self, image, message):
        #encoded_image = self.encoder(image, message)
        watermark_noise=self.encoder(image).sample
        #watermark_noise=torch.tanh(watermark_noise)
        #encoded_image=image+self.alpha*watermark_noise
        encoded_image=watermark_noise


        #to [-1,1] yuv
        encoded_image=encoded_image*2.0-1.0
        encoded_image=rgb2yuv_tensor(encoded_image)

        yuv_image=image*2.0-1.0
        yuv_image=rgb2yuv_tensor(yuv_image)
        
        noised_and_cover = self.noiser([encoded_image, yuv_image])
        noised_image = noised_and_cover[0]
        decoded_message = self.decoder(noised_image)
        return encoded_image, noised_image, decoded_message
