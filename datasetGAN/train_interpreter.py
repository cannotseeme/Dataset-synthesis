"""
Copyright (C) 2021 NVIDIA Corporation.  All rights reserved.
Licensed under The MIT License (MIT)

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
import sys
sys.path.append('..')
import torch
import torch.nn as nn
torch.manual_seed(0)
import scipy.misc
from torch_utils import misc
import json
from collections import OrderedDict
import numpy as np
from PIL import Image
import gc
import pickle
from models.stylegan2_ada import Generator
import copy
from numpy.random import choice
from torch.distributions import Categorical
import scipy.stats
from utils.utils import multi_acc, colorize_mask, get_label_stas, latent_to_image, oht_to_scalar, Interpolate
import torch.optim as optim
import argparse
import glob
from torch.utils.data import Dataset, DataLoader
import cv2
from multifca import MultiFca
import PIL.Image
import legacy
import dnnlib
device_ids = [0]
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class trainData(Dataset):

    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__(self):
        return len(self.X_data)

class trainData_inter(Dataset):

    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data

    def __getitem__(self, index):
        return [torch.FloatTensor(self.X_data[id][index]) for id in range(len(self.X_data))], self.y_data[index]

    def __len__(self):
        return len(self.X_data[0])

class pixel_classifier(nn.Module):
    def __init__(self, numpy_class, dim):
        super(pixel_classifier, self).__init__()
        if numpy_class < 32:
            self.layers = nn.Sequential(
                nn.Linear(dim, 128),
                nn.ReLU(),
                nn.BatchNorm1d(num_features=128),
                nn.Linear(128, 32),
                nn.ReLU(),
                nn.BatchNorm1d(num_features=32),
                nn.Linear(32, numpy_class),
                # nn.Sigmoid()
            )
        else:
            self.layers = nn.Sequential(
                nn.Linear(dim, 256),
                nn.ReLU(),
                nn.BatchNorm1d(num_features=256),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.BatchNorm1d(num_features=128),
                nn.Linear(128, numpy_class),
                # nn.Sigmoid()
            )


    def init_weights(self, init_type='normal', gain=0.02):
        '''
        initialize network's weights
        init_type: normal | xavier | kaiming | orthogonal
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/9451e70673400885567d08a9e97ade2524c700d0/models/networks.py#L39
        '''

        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)

                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

            elif classname.find('BatchNorm2d') != -1:
                nn.init.normal_(m.weight.data, 1.0, gain)
                nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)



    def forward(self, x):
        return self.layers(x)

def prepare_stylegan(args):

    if args['stylegan_ver'] == "1":
        if args['category'] == "car":
            resolution = 512
        else:
            assert "Not implementated!"

        avg_latent = np.load(args['average_latent'])
        avg_latent = torch.from_numpy(avg_latent).type(torch.FloatTensor).to(device)

        # kwargs for an Ada-StyleGAN2 generator
        G_kwargs = {
            "class_name": "models.stylegan2_ada.Generator",
            "z_dim": 512,
            "c_dim": 0,
            "w_dim": 512,
            "img_resolution": 512,
            "img_channels": 3,
            "mapping_kwargs": {
                "num_layers": 2
            },
            "synthesis_kwargs": {
                "channel_base": 32768,
                "channel_max": 512,
                "num_fp16_res": 0,
                "conv_clamp": 256,
            }
        }
        g_all = dnnlib.util.construct_class_by_name(**G_kwargs).train().requires_grad_(False).to(device)
        with dnnlib.util.open_url(args['stylegan_checkpoint']) as f:
            resume_data = legacy.load_network_pkl(f)
        for name, module in [('G_ema', g_all)]:
            misc.copy_params_and_buffers(resume_data[name], module, require_all=False)

    else:
        assert "Not implementated error"

    res = args['dim'][1]
    mode = args['upsample_mode']
    upsamplers = [nn.Upsample(scale_factor=res / 4, mode=mode),
                  nn.Upsample(scale_factor=res / 4, mode=mode),
                  nn.Upsample(scale_factor=res / 8, mode=mode),
                  nn.Upsample(scale_factor=res / 8, mode=mode),
                  nn.Upsample(scale_factor=res / 16, mode=mode),
                  nn.Upsample(scale_factor=res / 16, mode=mode),
                  nn.Upsample(scale_factor=res / 32, mode=mode),
                  nn.Upsample(scale_factor=res / 32, mode=mode),
                  nn.Upsample(scale_factor=res / 64, mode=mode),
                  nn.Upsample(scale_factor=res / 64, mode=mode),
                  nn.Upsample(scale_factor=res / 128, mode=mode),
                  nn.Upsample(scale_factor=res / 128, mode=mode),
                  nn.Upsample(scale_factor=res / 256, mode=mode),
                  nn.Upsample(scale_factor=res / 256, mode=mode)
                  ]

    if resolution > 256:
        upsamplers.append(nn.Upsample(scale_factor=res / 512, mode=mode))
        upsamplers.append(nn.Upsample(scale_factor=res / 512, mode=mode))

    if resolution > 512:
        upsamplers.append(Interpolate(res, 'bilinear'))
        upsamplers.append(Interpolate(res, 'bilinear'))

    return g_all, avg_latent, upsamplers

def generate_data(args, checkpoint_path, num_sample, start_step=0, vis=True):
    if args['category'] == 'car':
        from utils.data_util import car_20_palette as palette
    else:
        assert False
    if not vis:
        result_path = os.path.join(checkpoint_path, 'samples')
    else:
        result_path = os.path.join(checkpoint_path, 'vis_%d' % num_sample)
    if os.path.exists(result_path):
        pass
    else:
        os.system('mkdir -p %s' % result_path)
        print('Experiment folder created at: %s' % result_path)

    # prepare an Ada-StyleGAN2 generator
    g_all, avg_latent, upsamplers = prepare_stylegan(args)

    classifier_list = []
    interpreter_list = []
    for MODEL_NUMBER in range(args['model_num']):
        print('MODEL_NUMBER', MODEL_NUMBER)

        # load pixel classifier
        classifier = pixel_classifier(numpy_class=args['number_class'], dim=args['dim'][2])
        classifier = nn.DataParallel(classifier, device_ids=device_ids).cuda()
        checkpoint = torch.load(os.path.join(checkpoint_path, 'model_cls' + '.pth'))
        classifier.load_state_dict(checkpoint['model_state_dict'])
        classifier.eval()
        classifier_list.append(classifier)

        # load interpreter
        interpreter = MultiFca()
        interpreter = nn.DataParallel(interpreter, device_ids=device_ids).cuda()
        checkpoint = torch.load(os.path.join(checkpoint_path, 'model_inter' + '.pth'))
        interpreter.load_state_dict(checkpoint['model_state_dict'])
        interpreter.eval()
        interpreter_list.append(interpreter)

    softmax_f = nn.Softmax(dim=1)
    with torch.no_grad():
        latent_cache = []
        image_cache = []
        seg_cache = []
        entropy_calculate = []
        results = []
        np.random.seed(start_step)
        count_step = start_step
        print("num_sample: ", num_sample)

        for i in range(num_sample):
            if i % 100 == 0:
                print("Genearte", i, "Out of:", num_sample)
            curr_result = {}
            latent = np.random.randn(1, 512)
            curr_result['latent'] = latent
            latent = torch.from_numpy(latent).type(torch.FloatTensor).to(device)
            latent_cache.append(latent)
            dim_list = {4: 512, 8: 512, 16: 512, 32: 512, 64: 512, 128: 256, 256: 128, 512: 64}

            img, affine_layers = latent_to_image(g_all, upsamplers, latent, dim=args['dim'][1],
                                                     return_upsampled_layers=False, return_original_layers=True)

            if args['dim'][0] != args['dim'][1]:
                img = img[:, 32:224]
                for t, dim_mul in enumerate(dim_list.keys()):
                    # only for car
                    if dim_mul < 8:
                        affine_layers[t] = affine_layers[t][:, :, 0:int(dim_mul / 4 * 3)]
                    else:
                        affine_layers[t] = affine_layers[t][:, :, int(dim_mul / 8):int(dim_mul / 8 * 7)]
            else:
                img = img[0]

            image_cache.append(img)
            all_seg = []
            all_entropy = []
            mean_seg = None

            seg_mode_ensemble = []
            for MODEL_NUMBER in range(args['model_num']):
                classifier = classifier_list[MODEL_NUMBER]
                interpreter = interpreter_list[MODEL_NUMBER]
                img_seg = affine_layers
                img_seg = interpreter(img_seg)
                img_shape = img_seg.shape
                img_seg = img_seg.permute(0, 2, 3, 1).reshape(-1, img_shape[1])
                img_seg = classifier(img_seg).squeeze()
                entropy = Categorical(logits=img_seg).entropy()
                all_entropy.append(entropy)
                all_seg.append(img_seg)
                if mean_seg is None:
                    mean_seg = softmax_f(img_seg)
                else:
                    mean_seg += softmax_f(img_seg)
                img_seg_final = oht_to_scalar(img_seg)
                img_seg_final = img_seg_final.reshape(args['dim'][0], args['dim'][1], 1)
                img_seg_final = img_seg_final.cpu().detach().numpy()
                seg_mode_ensemble.append(img_seg_final)

            mean_seg = mean_seg / len(all_seg)
            full_entropy = Categorical(mean_seg).entropy()
            js = full_entropy - torch.mean(torch.stack(all_entropy), 0)
            top_k = js.sort()[0][- int(js.shape[0] / 10):].mean()
            entropy_calculate.append(top_k)

            img_seg_final = np.concatenate(seg_mode_ensemble, axis=-1)
            img_seg_final = scipy.stats.mode(img_seg_final, 2)[0].reshape(args['dim'][0], args['dim'][1])
            del (affine_layers)
            if vis:
                color_mask = 0.7 * colorize_mask(img_seg_final, palette) + 0.3 * img
                PIL.Image.fromarray(color_mask.squeeze().astype(np.uint8)).save(os.path.join(result_path, "vis_" + str(i) + '.jpg'))
                PIL.Image.fromarray(img.squeeze().astype(np.uint8)).save(os.path.join(result_path, "vis_" + str(i) + '_image.jpg'))
            else:
                seg_cache.append(img_seg_final)
                curr_result['uncertrainty_score'] = top_k.item()
                image_label_name = os.path.join(result_path, 'label_' + str(count_step) + '.jpg')
                image_name = os.path.join(result_path,  str(count_step) + '.jpg')
                js_name = os.path.join(result_path, str(count_step) + '.npy')
                img = img.squeeze()
                img = Image.fromarray(img.squeeze())
                img.save(image_name)
                np.save(js_name, img_seg_final)
                curr_result['image_name'] = image_name
                curr_result['image_label_name'] = image_label_name
                curr_result['js_name'] = js_name
                count_step += 1

                results.append(curr_result)
                if i % 1000 == 0 and i != 0:
                    with open(os.path.join(result_path, str(i) + "_" + str(start_step) + '.pickle'), 'wb') as f:
                        pickle.dump(results, f)

        with open(os.path.join(result_path, str(num_sample) + "_" + str(start_step) + '.pickle'), 'wb') as f:
            pickle.dump(results, f)

def prepare_data(args, palette):
    g_all, avg_latent, upsamplers = prepare_stylegan(args)
    latent_all = np.load(args['annotation_image_latent_path']).squeeze()
    print(latent_all.shape)
    latent_all = torch.from_numpy(latent_all).cuda()

    # load annotated mask
    mask_list = []
    im_list = []
    latent_all = latent_all[:args['max_training']]
    num_data = len(latent_all)

    for i in range(len(latent_all)):
        if i >= args['max_training']:
            break
        name = 'image_mask%0d.npy' % i

        im_frame = np.load(os.path.join( args['annotation_mask_path'] , name))
        mask = np.array(im_frame)
        mask = cv2.resize(np.squeeze(mask), dsize=(args['dim'][1], args['dim'][0]), interpolation=cv2.INTER_NEAREST)
        mask_list.append(mask)

        im_name = os.path.join( args['annotation_mask_path'], 'image_%d.jpg' % i)
        img = Image.open(im_name)
        img = img.resize((args['dim'][1], args['dim'][0]))

        im_list.append(np.array(img))

    # delete small annotation error
    for i in range(len(mask_list)):  # clean up artifacts in the annotation, must do
        for target in range(1, 50):
            if (mask_list[i] == target).sum() < 30:
                mask_list[i][mask_list[i] == target] = 0


    all_mask = np.stack(mask_list)

    # Generate ALL training data for training pixel classifier
    dim_list = {4: 512, 8: 512, 16: 512, 32: 512, 64: 512, 128: 256, 256: 128, 512: 64}  # resolution:channels
    all_feature_maps_train = []
    for dim_mul in dim_list.keys():
        all_feature_maps_train.append(np.zeros((len(latent_all), dim_list[dim_mul], dim_mul, dim_mul), dtype=np.float16))
    all_mask_train = np.zeros((len(latent_all), args['dim'][0], args['dim'][1]), dtype=np.float16)  # mask: len_all * dim[0] * dim[1]

    vis = []
    for i in range(len(latent_all)):
        gc.collect()
        latent_input = latent_all[i].float()
        img, feature_maps = latent_to_image(g_all, upsamplers, latent_input.unsqueeze(0), dim=args['dim'][1],
                                            return_upsampled_layers=False, return_original_layers=True, use_style_latents=args['annotation_data_from_w'])
        if args['dim'][0] != args['dim'][1]:
            img = img[:, 32:224]
            for t, dim_mul in enumerate(dim_list.keys()):
                if dim_mul < 8:
                    feature_maps[t] = feature_maps[t][:, :, 0:int(dim_mul / 4 * 3)]
                else:
                    feature_maps[t] = feature_maps[t][:, :, int(dim_mul/8):int(dim_mul/8*7)]
        mask = all_mask[i:i + 1]
        new_mask = np.squeeze(mask)

        for t, dim_mul in enumerate(dim_list.keys()):
            all_feature_maps_train[t][i: i+1] = feature_maps[t].cpu().detach().numpy().astype(np.float16)
        all_mask_train[i: i+1] = mask.astype(np.float16)
        img_show = cv2.resize(np.squeeze(img[0]), dsize=(args['dim'][1], args['dim'][1]), interpolation=cv2.INTER_NEAREST)
        curr_vis = np.concatenate( [im_list[i], img_show, colorize_mask(new_mask, palette)], 0 )
        vis.append(curr_vis)

    vis = np.concatenate(vis, 1)
    scipy.misc.imsave(os.path.join(args['exp_dir'], "train_data.jpg"), vis)

    return all_feature_maps_train, all_mask_train, num_data

def main(args):
    if args['category'] == 'car':
        from utils.data_util import car_20_palette as palette

    all_feature_maps_train_all, all_mask_train_all, num_data = prepare_data(args, palette)

    print(" *********************** Current number data " + str(num_data) + " ***********************")

    train_data_inter = trainData_inter(all_feature_maps_train_all, torch.FloatTensor(all_mask_train_all))
    train_loader_inter = DataLoader(dataset=train_data_inter, batch_size=1, shuffle=True)
    for MODEL_NUMBER in range(args['model_num']):

        gc.collect()
        # load pixel classifier
        classifier = pixel_classifier(numpy_class=args['number_class'], dim=args['dim'][2])
        classifier.init_weights()
        classifier = nn.DataParallel(classifier, device_ids=device_ids).cuda()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(classifier.parameters(), lr=0.001)
        classifier.train()

        # load interpreter
        interpreter = MultiFca()
        interpreter = nn.DataParallel(interpreter, device_ids=device_ids).cuda()
        optimizer_f = optim.Adam(interpreter.parameters(), lr=0.001)
        interpreter.train()

        # training for the label generator
        iteration = 0
        break_count = 0
        best_loss = 10000000
        stop_sign = 0
        for epoch in range(100):
            for i, (x_inter, y_inter) in enumerate(train_loader_inter):
                optimizer.zero_grad()
                optimizer_inter.zero_grad()
                x_inter = interpreter(x_inter)
                x_inter = x_inter.cpu().detach().numpy().astype(np.float16)
                y_inter = y_inter.cpu().detach().numpy().astype(np.float16)
                x_shape = x_inter.shape
                x_inter = x_inter.transpose(0, 2, 3, 1).reshape(-1, x_shape[1])
                y_inter = y_inter.reshape(-1)
                train_data = trainData(torch.FloatTensor(x_inter), torch.FloatTensor(y_inter))
                train_loader = DataLoader(dataset=train_data, batch_size=64, shuffle=True)

                for X_batch, y_batch in train_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    y_batch = y_batch.type(torch.long)
                    y_pred = classifier(X_batch)
                    loss = criterion(y_pred, y_batch)
                    acc = multi_acc(y_pred, y_batch)

                    loss.backward()
                    optimizer_f.step()
                    optimizer.step()

                    iteration += 1
                    if iteration % 1000 == 0:
                        print('Epoch : ', str(epoch), 'image : ', i, 'iteration', iteration, 'loss', loss.item(), 'acc', acc)
                        gc.collect()

                    if epoch > 3:
                        if loss.item() < best_loss:
                            best_loss = loss.item()
                            break_count = 0
                        else:
                            break_count += 1

                        if break_count > 50:
                            stop_sign = 1
                            break
                    if stop_sign == 1:
                        break

            if stop_sign == 1:
                print("*************** Break, Total iters,", iteration, ", at epoch", str(epoch), "***************")
                break

        gc.collect()
        cls_model_path = os.path.join(args['exp_dir'], 'model_cls' + '.pth')
        inter_model_path = os.path.join(args['exp_dir'], 'model_inter' + '.pth')
        print('save to:', cls_model_path, inter_model_path)
        torch.save({'model_state_dict': classifier.state_dict()}, cls_model_path)
        torch.save({'model_state_dict': inter.state_dict()}, inter_model_path)
        gc.collect()
        torch.cuda.empty_cache()    # clear cache memory on GPU


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--exp', type=str)
    parser.add_argument('--exp_dir', type=str,  default="")
    parser.add_argument('--generate_data', type=bool, default=False)
    parser.add_argument('--save_vis', type=bool, default=False)
    parser.add_argument('--start_step', type=int, default=0)
    parser.add_argument('--resume', type=str,  default="")
    parser.add_argument('--num_sample', type=int,  default=1000)

    args = parser.parse_args()

    opts = json.load(open(args.exp, 'r'))
    print("Opt", opts)

    if args.exp_dir != "":
        opts['exp_dir'] = args.exp_dir

    path = opts['exp_dir']
    if os.path.exists(path):
        pass
    else:
        os.system('mkdir -p %s' % (path))
        print('Experiment folder created at: %s' % (path))

    os.system('cp %s %s' % (args.exp, opts['exp_dir']))

    if args.generate_data:
        generate_data(opts, args.resume, args.num_sample, vis=args.save_vis, start_step=args.start_step)
    else:
        main(opts)