import torchvision.transforms as T
import torchvision.transforms.functional as Fu
import torchvision
import os

from model.decoder import Decoder
from options import HiDDenConfiguration
import torch

import numpy as np
from utils import ImageFolderInstance
from average_meter import AverageMeter
from collections import defaultdict
from typing import Any, BinaryIO, List, Optional, Tuple, Union
from tqdm import tqdm
from noise_layers.crop import Test_Crop
import argparse
from noise_layers.jpeg_compression import *
from PIL import Image,ImageEnhance 
from torchvision import datasets
from augly.image import functional as aug_functional


parser = argparse.ArgumentParser(description='Test folder')
parser.add_argument('--wm_path', '-wm', required=True, type=str,help='The directory where the data is stored.')
parser.add_argument('--whitening', '-w', required=True, type=str,help='The directory where the data is stored.')
parser.add_argument('--folder', '-f', required=True, type=str,help='The directory where the data is stored.')
parser.add_argument('--batch_size', '-b', default=8, type=int, help='The batch size.')
parser.add_argument('--transform', '-t', default=None, type=str, help='The transform using')
parser.add_argument('--transform_factor', '-tf', default=2.0, type=float, help='The transform factor')

args = parser.parse_args()


'''
sharpness_ehc = ImageEnhance.Sharpness()
brightness_ehc = ImageEnhance.Brightness()
color_ehc = ImageEnhance.Color()
contrast_ehc = ImageEnhance.Contrast()
'''


class ImageFolderEnhance(datasets.ImageFolder):
    """Folder datasets which returns the index of the image as well
    """

    def __init__(self, root, transform=None, target_transform=None):
        super(ImageFolderEnhance, self).__init__(root, transform, target_transform)


    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if args.transform == 'sharpness':
            sample=ImageEnhance.Sharpness(sample).enhance(args.transform_factor)
        elif args.transform == 'brightness':
            sample=ImageEnhance.Brightness(sample).enhance(args.transform_factor)
        elif args.transform == 'color':
            sample=ImageEnhance.Color(sample).enhance(args.transform_factor)
        elif args.transform == 'contrast':
            sample=ImageEnhance.Contrast(sample).enhance(args.transform_factor)
        elif args.transform =='jpeg':
            sample=aug_functional.encoding_quality(sample, quality=int(args.transform_factor))
        
        #sample = sample.convert("YCbCr")
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target




wm_checkpoint_path=args.wm_path
whitening_layer_path=args.whitening
folder_path=args.folder

batch_size=args.batch_size
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

fix_message=torch.Tensor([1., 1., 1., 0., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0.,                                                                                                                                                                                                   
        1., 1., 0., 1., 0., 1., 1., 0., 1., 0., 0., 1., 1., 1., 0., 1., 1., 1.,                                                                                                                                                                                                                 
        0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0.])
message=fix_message.repeat(batch_size,1).to(device)




hidden_config = HiDDenConfiguration(H=512, W=512,
                                            message_length=48,
                                            encoder_blocks=4, encoder_channels=64,
                                            decoder_blocks=7, decoder_channels=64,
                                            use_discriminator=True,
                                            use_vgg=False,
                                            discriminator_blocks=3, discriminator_channels=64,
                                            decoder_loss=1,
                                            encoder_loss=0.2,
                                            adversarial_loss=1e-3,
                                            enable_fp16=False,
                                            alpha=0.3,
                                            loss_type='mse',
                                            opt_type='adam',
                                            data_len=5000,
                                            accu_step=1
                                            )

wm_decoder=Decoder(hidden_config)
wm_checkpoint=torch.load(wm_checkpoint_path)
decoder_dict=wm_decoder.state_dict()
pretrained_dict = {k[8:]: v for k, v in wm_checkpoint['enc-dec-model'].items() if (k[8:] in decoder_dict and 'encoder' not in k)}
wm_decoder.load_state_dict(pretrained_dict)
wm_decoder=wm_decoder.to(device)

whitening_layer=torch.load(whitening_layer_path).to(device)
whitening_layer.eval()


crop_size=164

noise_transform=T.CenterCrop(512)

if args.transform == 'crop':
    noise_transform=T.CenterCrop(512)
elif args.transform == 'resize':
    resize=T.Resize()

open_transform=T.Compose([
        
        noise_transform,
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])


dataset=ImageFolderEnhance(folder_path,open_transform)
data_loader=torch.utils.data.DataLoader(dataset, batch_size=batch_size,shuffle=False, num_workers=4)

#validation_losses = defaultdict(AverageMeter)

acc_list=[]
wm_decoder.eval()


with torch.no_grad():
    for image, _ in tqdm(data_loader):
        image = image.to(device)

        image=rgb2yuv_tensor(image)
        decoded_messages=wm_decoder(image)
        decoded_messages=whitening_layer(decoded_messages)
       

        decoded_rounded = decoded_messages.detach().cpu().numpy().round().clip(0, 1)
        bitwise_avg_err = np.sum(np.abs(decoded_rounded - message.detach().cpu().numpy())) / (batch_size * message.shape[1])

        acc_list.append(bitwise_avg_err)


np_acc=np.array(acc_list)
print(np_acc.mean())