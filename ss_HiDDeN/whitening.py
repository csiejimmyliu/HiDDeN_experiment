import os
import time
import pprint
import argparse
import torch
import numpy as np
import pickle
import utils
import csv

from model.hidden import Hidden
from noise_layers.noiser import Noiser
from average_meter import AverageMeter
from noise_argparser import NoiseArgParser

import torch.nn as nn


def write_validation_loss(file_name, losses_accu, experiment_name, epoch, write_header=False):
    with open(file_name, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if write_header:
            row_to_write = ['experiment_name', 'epoch'] + [loss_name.strip() for loss_name in losses_accu.keys()]
            writer.writerow(row_to_write)
        row_to_write = [experiment_name, epoch] + ['{:.4f}'.format(loss_avg.avg) for loss_avg in losses_accu.values()]
        writer.writerow(row_to_write)


def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    #device = torch.device('cpu')

    parser = argparse.ArgumentParser(description='Training of HiDDeN nets')
    # parser.add_argument('--size', '-s', default=128, type=int, help='The size of the images (images are square so this is height and width).')
    parser.add_argument('--data-dir', '-d', required=True, type=str, help='The directory where the data is stored.')
    parser.add_argument('--w_save', '-w', required=True, type=str, help='The path where the whitening model to save.')
    parser.add_argument('--runs_root', '-r', default=os.path.join('.', 'experiments'), type=str,
                        help='The root folder where data about experiments are stored.')
    parser.add_argument('--batch-size', '-b', default=1, type=int, help='Validation batch size.')
    parser.add_argument('--checkpoint_path', '-c', required=True, type=str, help='The path of wm decoder.')
    parser.add_argument('--noise', nargs='*', action=NoiseArgParser,
            help="Noise layers configuration. Use quotes when specifying configuration, e.g. 'cropout((0.55, 0.6), (0.55, 0.6))'")
    args = parser.parse_args()
    
    noise_config = args.noise if args.noise is not None else []



    print(f'Run folder: {args.runs_root}')
    options_file = os.path.join(args.runs_root, 'options-and-config.pickle')
    train_options, hidden_config, _ = utils.load_options(options_file)
    train_options.train_folder = os.path.join(args.data_dir, 'train')
    train_options.validation_folder = os.path.join(args.data_dir, 'train')
    train_options.batch_size = args.batch_size
    checkpoint, chpt_file_name = utils.load_checkpoint(args.checkpoint_path)
    print(f'Loaded checkpoint from file {chpt_file_name}')


    noiser = Noiser(noise_config,device)
    model = Hidden(hidden_config, device, noiser, tb_logger=None,train_options=train_options)
    utils.model_from_checkpoint(model, checkpoint)
    model.encoder_decoder.eval()


    print('Model loaded successfully. Starting validation run...')
    _, val_data = utils.get_data_loaders(hidden_config, train_options)
    file_count = len(val_data.dataset)
    print('File count: ',file_count)



    step = 0

    list_arr=[]

    with torch.no_grad():
        for image, _ in val_data:
            image = image.to(device)
            decoded_messages= model.encoder_decoder.decoder(image)

            if step==0:
                list_arr=decoded_messages.detach().cpu()
            else:
                list_arr=torch.cat( (list_arr,decoded_messages.detach().cpu()),0 )

            step += 1
            
    miu=torch.mean(list_arr, 0)
    conv_matrix=torch.cov(list_arr.T)
    conv_matrix=(conv_matrix.T+conv_matrix)/2

    L, V = torch.linalg.eig(conv_matrix)
    L_matrix=torch.diag_embed(1/torch.sqrt(L))
    b=-(L_matrix@V.T).to(torch.float)@miu
    w=(L_matrix@V.T).to(torch.float)

    whitening_layer=nn.Linear(hidden_config.message_length,hidden_config.message_length)
    whitening_layer.weight=nn.Parameter(w)
    whitening_layer.bias=nn.Parameter(b)
    whitening_layer.eval()

    torch.save(whitening_layer,args.w_save)
    


if __name__ == '__main__':
    main()