import os
import copy

from torch.utils.data import Dataset

from common import _load_paths, _load_image, _filename_separator
from torch.utils.data import Dataset
import cv2
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
import torch
import random
import librosa
import librosa.display
from util import Equirec2Cube
from mpl_toolkits.axes_grid1 import make_axes_locatable
import glob, ipdb, pickle

class Dataload(Dataset):
    def __init__(self, txt_path, height=256, width=512,disable_color_augmentation=False,
                 disable_LR_filp_augmentation=False, disable_yaw_rotation_augmentation=False, test = False, is_training=False, transform = None, target_transform = None):

        self.max_depth_meters = 16.0
        self.w = width
        self.h = height

        self.is_training = is_training

        self.color_augmentation = not disable_color_augmentation
        self.LR_filp_augmentation = not disable_LR_filp_augmentation
        self.yaw_rotation_augmentation = not disable_yaw_rotation_augmentation

        if self.color_augmentation:
            try:
                self.brightness = (0.8, 1.2)
                self.contrast = (0.8, 1.2)
                self.saturation = (0.8, 1.2)
                self.hue = (-0.1, 0.1)
                self.color_aug = transforms.ColorJitter.get_params(
                    self.brightness, self.contrast, self.saturation, self.hue)
            except TypeError:
                self.brightness = 0.2
                self.contrast = 0.2
                self.saturation = 0.2
                self.hue = 0.1
                self.color_aug = transforms.ColorJitter.get_params(
                    self.brightness, self.contrast, self.saturation, self.hue)

        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self.test = test
        fh = open(txt_path, 'r')
        imgs = []
        for line in fh:
            line = line.rstrip()
            words = line.split()
            imgs.append((words[0], words[1]))#imgs.append((words[0], words[3]))
            self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        #print(fn)
        inputs = {}
        # with open('rgbpath.txt', "a") as f:
        #     f.write(str(fn) + '\n')
        #     f.close()
        # img = cv2.imread(fn)
        # img = cv2.resize(img, (512,256), interpolation=cv2.INTER_CUBIC)
        rgb = cv2.imread('../data/'+fn)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, dsize=(512, 256))


        
        # depth = cv2.imread(label, cv2.IMREAD_ANYDEPTH)
        # depth = cv2.resize(depth,(512,256),interpolation=cv2.INTER_CUBIC)

        gt_depth = cv2.imread('../data/'+label, cv2.IMREAD_ANYDEPTH)
        gt_depth = cv2.resize(gt_depth, dsize=(512, 256), interpolation=cv2.INTER_CUBIC)
        gt_depth = gt_depth.astype(np.float)/512#gt_depth = gt_depth.astype(np.float)
        gt_depth[gt_depth > self.max_depth_meters + 1] = self.max_depth_meters + 1

        if self.is_training and self.yaw_rotation_augmentation:
            # random yaw rotation
            roll_idx = random.randint(0, self.w//4)
        else:
            roll_idx = 0
            
            
        rgb = np.roll(rgb, roll_idx, 1)
        gt_depth = np.roll(gt_depth, roll_idx, 1)

        if self.is_training and self.LR_filp_augmentation and random.random() > 0.5:
            rgb = cv2.flip(rgb, 1)
            gt_depth = cv2.flip(gt_depth, 1)

        if self.is_training and self.color_augmentation and random.random() > 0.5:
            aug_rgb = np.asarray(self.color_aug(transforms.ToPILImage()(rgb)))
        else:
            aug_rgb = rgb

        rgb = self.to_tensor(rgb.copy())
        aug_rgb = self.to_tensor(aug_rgb.copy())

        inputs["rgb"] = rgb
        inputs["normalized_rgb"] = self.normalize(aug_rgb)

        inputs["gt_depth"] = torch.from_numpy(np.expand_dims(gt_depth, axis=0))
        mask = ((inputs["gt_depth"] > 0) & (inputs["gt_depth"] <= self.max_depth_meters)
                              & ~torch.isnan(inputs["gt_depth"]))
        mask1 = (rgb[0] != 0)
        mask1 = (mask1)
        inputs["val_mask"] = mask*mask1
        return inputs

    def __len__(self):
        return len(self.imgs)



F_EXTENSIONS = [
    ".jpg", ".JPG", ".jpeg", ".JPEG", ".npy",
    ".png", ".PNG", ".ppm", ".PPM", ".bmp", ".BMP", ".tiff", ".wav", ".WAV", ".aif", ".aiff", ".AIF", ".AIFF"
]


SCENE_SPLITS = {
    'train': ['sT4fr6TAbpF', 'E9uDoFAP3SH', 'VzqfbhrpDEA', 'kEZ7cmS4wCh', '29hnd4uzFmX', 'ac26ZMwG7aT',
              'i5noydFURQK', 's8pcmisQ38h', 'rPc6DW4iMge', 'EDJbREhghzL', 'mJXqzFtmKg4', 'B6ByNegPMKs',
              'JeFG25nYj2p', '82sE5b5pLXE', 'D7N2EKCX4Sj', '7y3sRwLe3Va', 'HxpKQynjfin', '5LpN3gDmAk7',
              'gTV8FGcVJC9', 'ur6pFq6Qu1A', 'qoiz87JEwZ2', 'PuKPg4mmafe', 'VLzqgDo317F', 'aayBHfsNo7d',
              'JmbYfDe2QKZ', 'XcA2TqTSSAj', '8WUmhLawc2A', 'sKLMLpTHeUy', 'r47D5H71a5s', 'Uxmj2M2itWa',
              'Pm6F8kyY3z2', 'p5wJjkQkbXX', '759xd9YjKW5', 'JF19kD82Mey', 'V2XKFyX4ASd', '1LXtFkjw3qL',
              '17DRP5sb8fy', '5q7pvUzZiYa', 'VVfe2KiqLaN', 'Vvot9Ly1tCj', 'ULsKaCPVFJR', 'D7G3Y4RVNrH',
              'uNb9QFRL6hY', 'ZMojNkEp431', '2n8kARJN3HM', 'vyrNrziPKCB', 'e9zR4mvMWw7', 'r1Q1Z4BcV1o',
              'PX4nDJXEHrG', 'YmJkqBEsHnH', 'b8cTxDM8gDG', 'GdvgFV5R1Z5', 'pRbA3pwrgk9', 'jh4fc5c5qoQ',
              '1pXnuDYAj8r', 'S9hNv5qa7GM', 'VFuaQ6m2Qom', 'cV4RVeZvu5T', 'SN83YJsR3w2'],
    'val': ['x8F5xyUWy9e', 'QUCTc6BB5sX', 'EU6Fwq7SyZv', '2azQ1b91cZZ', 'Z6MFQCViBuw', 'pLe4wQe7qrG',
            'oLBMNvg9in8', 'X7HyMhZNoso', 'zsNo4HB9uLZ', 'TbHJrupSAjP', '8194nk5LbLH'],
    'test': ['pa4otMbVnkk', 'yqstnuAEVhm', '5ZKStnWn8Zo', 'Vt2qJdWjCF2', 'wc2JMjhGNzB', 'WYY7iVyf5p8',
             'fzynW3qQPVF', 'UwV83HsGsw3', 'q9vSo1VnCiC', 'ARNzJeq3xxb', 'rqfALeAoiTq', 'gYvKGZ5eRqb',
             'YFuZgdQ5vWj', 'jtcxE69GiFV', 'gxdoqLR6rwA'],
}
SCENE_SPLITS['train_distractor'] = SCENE_SPLITS['train']
SCENE_SPLITS['val_distractor'] = SCENE_SPLITS['val']
SCENE_SPLITS['test_distractor'] = SCENE_SPLITS['test']

def is_image_audio_file(filename):
    return any(filename.endswith(extension) for extension in F_EXTENSIONS)

def make_dataset(dir, mode):
    images = []
    depth = []
    waves = []
    assert os.path.isdir(dir), "%s is not a valid directory." % dir

    dir_rgb = os.path.join(dir,'{}_A'.format(mode))
    dir_wave = os.path.join(dir,'{}_B'.format(mode))
    dir_depth = os.path.join(dir,'{}_C'.format(mode))
    
    for root, _, fnames in sorted(os.walk(dir_rgb)):
        for fname in fnames:
            if is_image_audio_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
             
    for root, _, fnames in sorted(os.walk(dir_depth)):
        for fname in fnames:
            if is_image_audio_file(fname):
                path = os.path.join(root, fname)
                depth.append(path)
                
    for root, _, fnames in sorted(os.walk(dir_wave)):
        for fname in fnames:
            if is_image_audio_file(fname):
                path = os.path.join(root, fname)
                waves.append(path)

    return images, depth, waves




def get_transform(convert =  False, resize = True, resolution = 128):
    transform_list = []

    if convert:
        # need to convert to tensor if not already the case to apply transform
        transform_list += [transforms.ToTensor()]

    if resize:
        transform_list.append(transforms.Resize((resolution,resolution)))

    return transforms.Compose(transform_list)

"""
#for original mp3d soundspace
HOP_LENGTH = 32//4 #1024//16 #8 #64//4
N_FFT = 512 #1024 #511 #512//2 #512
WIN_LENGTH = 32 #64 #1024//2 #16
"""

"""
# from VisualEchos or beyond image to depth
HOP_LENGTH = 16 #1024//16 #8 #64//4
N_FFT = 512 #1024 #511 #512//2 #512
WIN_LENGTH = 64 #64 #1024//2 #16
"""

min_list = []
max_list = []
# -18.42068099975586 minavg
# 0.8 avg max
def generate_spectrogram_mp3d(audioL, audioR, resize):

    #for original mp3d soundspace
    HOP_LENGTH = 32//4 #1024//16 #8 #64//4
    N_FFT = 512 #1024 #511 #512//2 #512
    WIN_LENGTH = 32 #64 #1024//2 #16
    winl=WIN_LENGTH

    #channel_1_spec = librosa.stft(audioL)#, n_fft=NFFT, win_length=winl)
    #channel_2_spec = librosa.stft(audioR)#, n_fft=NFFT, win_length=winl)
    #channel_1_spec = librosa.feature.melspectrogram(audioL, n_fft=512//2, win_length=winl)
    #channel_2_spec = librosa.feature.melspectrogram(audioR, n_fft=512//2, win_length=winl)
    #audio_chip = librosa.chirp(sr= 44100, fmin=0, fmax=44100//2, duration=0.003, linear=True)
    audio_chip = librosa.chirp(sr= 16000, fmin=0, fmax=16000//2, duration=0.003, linear=True)
    
    audioR = np.convolve(audioR,audio_chip)
    audioL = np.convolve(audioL,audio_chip)

    channel_1_spec = librosa.stft(audioL, n_fft=N_FFT, win_length=winl, hop_length = HOP_LENGTH)
    channel_2_spec = librosa.stft(audioR, n_fft=N_FFT, win_length=winl, hop_length = HOP_LENGTH)
    #channel_1_spec = np.log(np.abs(librosa.stft(audioL, n_fft=N_FFT, win_length=winl, hop_length = HOP_LENGTH))+ 1e-8)
    #channel_2_spec = np.log(np.abs(librosa.stft(audioR, n_fft=N_FFT, win_length=winl, hop_length = HOP_LENGTH))+ 1e-8)

    #min_list.append((channel_1_spec.min()+channel_2_spec.min())/2.)
    #max_list.append((channel_1_spec.max()+channel_2_spec.max())/2.)
    #print('avg min:{}'.format(np.average(np.array(min_list))))
    #print('avg max:{}'.format(np.average(np.array(max_list))))

    #channel_1_spec = (((channel_1_spec - channel_1_spec.min())/(channel_1_spec.max() - channel_1_spec.min()) * 2) - 1)
    #channel_2_spec = (((channel_2_spec - channel_2_spec.min())/(channel_2_spec.max() - channel_2_spec.min()) * 2) - 1)
    spec_left = np.expand_dims(np.abs(channel_1_spec), axis=0)
    spec_right = np.expand_dims(np.abs(channel_2_spec), axis=0)

    #spectro_two_channel = np.concatenate((np.log(spec_left**2 + 1), np.log(spec_right**2 + 1)), axis=0)
    spectro_two_channel = np.concatenate((spec_left, spec_right), axis=0)

    #spectro_two_channel = np.log(spectro_two_channel + 1e-8)
    #spectro_two_channel = (spectro_two_channel - spectro_two_channel.min())/(spectro_two_channel.max() - spectro_two_channel.min())
    #print(spectro_two_channel.shape)
    PLOT = False
    if PLOT:
        plt.figure(figsize=(32,16))
        fig, ax = plt.subplots(1,2)                                               
        ax[0].set_title('Power spectrogram gt left')
        img_left = librosa.display.specshow(librosa.amplitude_to_db(spectro_two_channel[0,:,:]), 
                                    y_axis='linear', x_axis='ms', ax=ax[0], sr = 16000, n_fft = N_FFT, win_length = winl, hop_length = HOP_LENGTH)
        fig.colorbar(img_left, ax=ax[0], format="%+2.0f dB")

        ax[1].set_title('Power spectrogram gt right')
        img_right = librosa.display.specshow(librosa.amplitude_to_db(spectro_two_channel[1,:,:]), 
                                    y_axis='linear', x_axis='ms', ax=ax[1], sr = 16000, n_fft = N_FFT, win_length = winl, hop_length = HOP_LENGTH)
        fig.colorbar(img_right, ax=ax[1], format="%+2.0f dB")
        
        plt.savefig( './img/IR_debug_new.jpg')
    
        
        nb_display = 1
        spec_to_display = spectro_two_channel[0,:,:] #np.exp((spectro_two_channel[0,:,:] + 1)*0.5*(18.42+0.8) - 18.42) - 1e-8
        figure, axes = plt.subplots(1, nb_display, figsize = (16, 16))
        spec = axes.imshow(spec_to_display, origin = 'lower', cmap = 'magma')
        #spec = axes[0, i].imshow(batch_pred_tensor[i,...].data.cpu().float().numpy(), origin = 'lower', cmap = 'magma')
        divider = make_axes_locatable(axes)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(spec, cax=cax)
        axes.set_title('gt_spec_ch_' + str(0), size = 8)
        plt.tight_layout()
        plt.savefig("./img" + '/spec.png')
        
        print("shape:",spectro_two_channel.shape)
    


    spec_transform = get_transform(convert =  True, resize = resize, resolution = 256)
    #new_spectro_two_channel = torch.zeros((2,256,256))
    #new_spectro_two_channel = torch.zeros((2,513,656))
    #new_spectro_two_channel = torch.zeros((2,256,164))
    #new_spectro_two_channel = torch.zeros((2,257,312))
    new_spectro_two_channel = torch.zeros((2,257,226))
    new_spectro_two_channel[0] = spec_transform(spectro_two_channel[0])
    new_spectro_two_channel[1] = spec_transform(spectro_two_channel[1])

    return new_spectro_two_channel #.permute((1,2,0)) #shape: (2, 256, 43)

def generate_spectrogram_replica(audioL, audioR, resize):

    # from VisualEchos or beyond image to depth
    HOP_LENGTH = 16 #1024//16 #8 #64//4
    N_FFT = 512 #1024 #511 #512//2 #512
    WIN_LENGTH = 64 #64 #1024//2 #16
    winl=WIN_LENGTH

    #channel_1_spec = librosa.stft(audioL)#, n_fft=NFFT, win_length=winl)
    #channel_2_spec = librosa.stft(audioR)#, n_fft=NFFT, win_length=winl)
    #channel_1_spec = librosa.feature.melspectrogram(audioL, n_fft=512//2, win_length=winl)
    #channel_2_spec = librosa.feature.melspectrogram(audioR, n_fft=512//2, win_length=winl)
    audio_chip = librosa.chirp(sr= 44100, fmin=0, fmax=44100//2, duration=0.003, linear=True)
    #audio_chip = librosa.chirp(sr= 16000, fmin=0, fmax=16000//2, duration=0.003, linear=True)
    
    audioR = np.convolve(audioR,audio_chip)
    audioL = np.convolve(audioL,audio_chip)

    channel_1_spec = librosa.stft(audioL, n_fft=N_FFT, win_length=winl, hop_length = HOP_LENGTH)
    channel_2_spec = librosa.stft(audioR, n_fft=N_FFT, win_length=winl, hop_length = HOP_LENGTH)
    #channel_1_spec = np.log(np.abs(librosa.stft(audioL, n_fft=N_FFT, win_length=winl, hop_length = HOP_LENGTH))+ 1e-8)
    #channel_2_spec = np.log(np.abs(librosa.stft(audioR, n_fft=N_FFT, win_length=winl, hop_length = HOP_LENGTH))+ 1e-8)

    #min_list.append((channel_1_spec.min()+channel_2_spec.min())/2.)
    #max_list.append((channel_1_spec.max()+channel_2_spec.max())/2.)
    #print('avg min:{}'.format(np.average(np.array(min_list))))
    #print('avg max:{}'.format(np.average(np.array(max_list))))

    #channel_1_spec = (((channel_1_spec - channel_1_spec.min())/(channel_1_spec.max() - channel_1_spec.min()) * 2) - 1)
    #channel_2_spec = (((channel_2_spec - channel_2_spec.min())/(channel_2_spec.max() - channel_2_spec.min()) * 2) - 1)
    spec_left = np.expand_dims(np.abs(channel_1_spec), axis=0)
    spec_right = np.expand_dims(np.abs(channel_2_spec), axis=0)

    #spectro_two_channel = np.concatenate((np.log(spec_left**2 + 1), np.log(spec_right**2 + 1)), axis=0)
    spectro_two_channel = np.concatenate((spec_left, spec_right), axis=0)

    #spectro_two_channel = np.log(spectro_two_channel + 1e-8)
    #spectro_two_channel = (spectro_two_channel - spectro_two_channel.min())/(spectro_two_channel.max() - spectro_two_channel.min())
    #print(spectro_two_channel.shape)
    PLOT = False
    if PLOT:
        plt.figure(figsize=(32,16))
        fig, ax = plt.subplots(1,2)                                               
        ax[0].set_title('Power spectrogram gt left')
        img_left = librosa.display.specshow(librosa.amplitude_to_db(spectro_two_channel[0,:,:]), 
                                    y_axis='linear', x_axis='ms', ax=ax[0], sr = 44100, n_fft = N_FFT, win_length = winl, hop_length = HOP_LENGTH)
        fig.colorbar(img_left, ax=ax[0], format="%+2.0f dB")
        

        ax[1].set_title('Power spectrogram gt right')
        img_right = librosa.display.specshow(librosa.amplitude_to_db(spectro_two_channel[1,:,:]), 
                                    y_axis='linear', x_axis='ms', ax=ax[1], sr = 44100, n_fft = N_FFT, win_length = winl, hop_length = HOP_LENGTH)
        fig.colorbar(img_right, ax=ax[1], format="%+2.0f dB")
        
        plt.savefig( './img/IR_debug_new.jpg')
    
        
        nb_display = 1
        spec_to_display = spectro_two_channel[0,:,:] #np.exp((spectro_two_channel[0,:,:] + 1)*0.5*(18.42+0.8) - 18.42) - 1e-8
        figure, axes = plt.subplots(1, nb_display, figsize = (16, 16))
        spec = axes.imshow(spec_to_display, origin = 'lower', cmap = 'magma')
        #spec = axes[0, i].imshow(batch_pred_tensor[i,...].data.cpu().float().numpy(), origin = 'lower', cmap = 'magma')
        divider = make_axes_locatable(axes)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(spec, cax=cax)
        axes.set_title('gt_spec_ch_' + str(0), size = 8)
        plt.tight_layout()
        plt.savefig("./img" + '/spec.png')
        
        print("shape:",spectro_two_channel.shape)
    


    spec_transform = get_transform(convert =  True, resize = resize, resolution = 256)
    #new_spectro_two_channel = torch.zeros((2,256,256))
    #new_spectro_two_channel = torch.zeros((2,513,656))
    #new_spectro_two_channel = torch.zeros((2,256,164))
    new_spectro_two_channel = torch.zeros((2,257,312))
    #new_spectro_two_channel = torch.zeros((2,257,226))
    new_spectro_two_channel[0] = spec_transform(spectro_two_channel[0])
    new_spectro_two_channel[1] = spec_transform(spectro_two_channel[1])

    return new_spectro_two_channel #.permute((1,2,0)) #shape: (2, 256, 43)


class Dataload_RealEquirectangular_PanoCacheObs(Dataset):
    def __init__(self, data_path, mode = "train", height=256, width=512 ,disable_color_augmentation=False,
                    disable_LR_filp_augmentation=False, disable_yaw_rotation_augmentation=False, test = False, is_training=False, transform = None, target_transform = None, dataset_use_ratio = 1., dataset = "mp3d", model = "PanoFormer"):

        self.max_depth_meters = 16.0
        self.w = width
        self.h = height
        
        print('resolution='+ str((self.w, self.h)) + 'used')

        self.is_training = is_training

        self.color_augmentation = not disable_color_augmentation
        self.LR_filp_augmentation = not disable_LR_filp_augmentation
        self.yaw_rotation_augmentation = not disable_yaw_rotation_augmentation
        
        self.mode = mode
        print("mode = {}".format(self.mode))
        
        self.dataset_use_ratio = dataset_use_ratio
        print("self.dataset_use_ratio = {}".format(self.dataset_use_ratio))
        
        self.model = model
        if self.model == "Unifuse":
            self.e2c = Equirec2Cube(self.h, self.w, self.h // 2)

        if self.color_augmentation:
            try:
                self.brightness = (0.8, 1.2)
                self.contrast = (0.8, 1.2)
                self.saturation = (0.8, 1.2)
                self.hue = (-0.1, 0.1)
                self.color_aug = transforms.ColorJitter.get_params(
                    self.brightness, self.contrast, self.saturation, self.hue)
            except TypeError:
                self.brightness = 0.2
                self.contrast = 0.2
                self.saturation = 0.2
                self.hue = 0.1
                self.color_aug = transforms.ColorJitter.get_params(
                    self.brightness, self.contrast, self.saturation, self.hue)

        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self.test = test
        
        self.rgb_paths, self.depth_paths, self.wave_paths = (make_dataset(data_path, self.mode))
        self.rgb_paths = sorted(self.rgb_paths)
        self.depth_paths = sorted(self.depth_paths)
        self.wave_paths = sorted(self.wave_paths)
        
        assert(len(self.rgb_paths) == len(self.depth_paths) == len(self.wave_paths))
        
        # sampling
        use_dataset_ratio = self.dataset_use_ratio
        print("dataset ratio used:{} of the whole datasets".format(use_dataset_ratio))
        NUM_dataset_all = len(self.rgb_paths)
        NUM_dataset = int(NUM_dataset_all*use_dataset_ratio)
        print("dataset num used:{}/{} for datasets{}".format(NUM_dataset, NUM_dataset_all, self.mode))
        dataset_sampled_index = random.choices(np.array(range(NUM_dataset_all)), k = NUM_dataset)
        self.rgb_paths = [self.rgb_paths[i] for i in dataset_sampled_index]
        self.depth_paths = [self.depth_paths[i] for i in dataset_sampled_index]
        self.wave_paths = [self.wave_paths[i] for i in dataset_sampled_index]
        
        self.transform = transform
        self.target_transform = target_transform
        self.dataset = dataset
        print("dataset {} used.".format(self.dataset))
        

        current_work_dir = os.path.dirname(__file__)   
        path_check_dir = current_work_dir + '/random_path_check_files'
        if not os.path.exists(path_check_dir):
            os.makedirs(path_check_dir)
            print("created path_check_dir : {}".format(path_check_dir))
        path_check_file = path_check_dir + '/selected_paths_{}_{}_ratio{}.pickle'.format(self.dataset ,self.mode,use_dataset_ratio)
        path_check_content = {"rgb_paths": self.rgb_paths,
                              "depth_paths": self.depth_paths,
                              "wave_paths": self.wave_paths}
        if not os.path.exists(path_check_file):
            print("path_check_file : {} not yet existed, creating pickle file.".format(path_check_file))
            with open(path_check_file, 'wb') as handle:
                pickle.dump(path_check_content, handle)
        else:
            print("path_check_file : {} already existed, checking pickle file.".format(path_check_file))
            with open(path_check_file, 'rb') as handle:
                path_check_content = pickle.load(handle)
                assert (path_check_content["rgb_paths"] == self.rgb_paths
                and path_check_content["depth_paths"] == self.depth_paths
                and path_check_content["wave_paths"] == self.wave_paths)
                print("path identity check passed.")

            

    def get_PanoCacheObs_rgb(self,rgb_path):

        rgb_ = cv2.imread(rgb_path)
        rgb_ = cv2.cvtColor(rgb_, cv2.COLOR_BGR2RGB)

        rgb_angle360 =  rgb_/255.
        
        save_for_debug = False
        if save_for_debug:
            im_to_save = rgb_angle360*255.
            print("saving image using, remember to shut down.")                  
            cv2.imwrite("./your_file.jpeg", im_to_save)
            
        return  rgb_angle360

    def get_PanoCacheObs_depth(self,rgb_path):
        rgb_ = np.load(rgb_path)
        rgb_angle360 =  rgb_ #np.reshape(np.concatenate(rgb_list_angle360, axis=1), (128*2,128*2,3))
        
        return  rgb_angle360

    def get_spec(self,wave_path):
        
        #get the spectrogram of both channel

        if self.dataset == "mp3d":
            audio, audio_rate = librosa.load(wave_path, sr=16000, mono=False, duration=0.11)
            audio_spec_both = generate_spectrogram_mp3d(audio[0,:], audio[1,:], resize = False)
        elif self.dataset == "replica":
            audio, audio_rate = librosa.load(wave_path, sr=44100, mono=False, duration=0.11)
            audio_spec_both = generate_spectrogram_replica(audio[0,:], audio[1,:], resize = False)

        return  audio_spec_both

    def __getitem__(self, index):
        rgb_path, depth_path, wave_path = self.rgb_paths[index], self.depth_paths[index], self.wave_paths[index]
        #print(fn)

        inputs = {}
        rgb = (self.get_PanoCacheObs_rgb(rgb_path)).astype(np.float32)
        #ipdb.set_trace()
        rgb = cv2.resize(rgb, dsize=(self.w, self.h), interpolation=cv2.INTER_CUBIC)

        gt_depth = (self.get_PanoCacheObs_depth(depth_path)).astype(np.float32)
        gt_depth = cv2.resize(gt_depth, dsize=(self.w, self.h), interpolation=cv2.INTER_NEAREST)
        gt_depth = gt_depth.astype(np.float32)#gt_depth = gt_depth.astype(np.float)

        gt_depth[gt_depth > self.max_depth_meters] = self.max_depth_meters
        
        if self.is_training and self.yaw_rotation_augmentation:
            # random yaw rotation
            roll_idx = random.randint(0, self.w//4)
        else:
            roll_idx = int(self.w//8) #put center fro echo in center
            #roll_idx = 0
            

        if self.is_training and self.LR_filp_augmentation and random.random() > 0.5:
            rgb = cv2.flip(rgb, 1)
            gt_depth = cv2.flip(gt_depth, 1)

        if self.is_training and self.color_augmentation and random.random() > 0.5:
            aug_rgb = np.asarray(self.color_aug((transforms.ToPILImage()((rgb*255.).astype(np.uint8)))))/255.
        else:
            aug_rgb = rgb


        #cube_rgb, cube_gt_depth = self.e2c.run(rgb, gt_depth[..., np.newaxis])
        if self.model == "Unifuse":
            cube_rgb = self.e2c.run(rgb)
            cube_aug_rgb = self.e2c.run(aug_rgb)
            cube_rgb = self.to_tensor(cube_rgb.copy())
            cube_aug_rgb = self.to_tensor(cube_aug_rgb.copy())

        rgb = self.to_tensor(rgb.copy()).to(torch.float32)
        aug_rgb = self.to_tensor(aug_rgb.copy()).to(torch.float32)
        spec_binaural = self.get_spec(wave_path)

        inputs["rgb"] = rgb
        inputs["normalized_rgb"] = self.normalize(aug_rgb)

        if self.model == "Unifuse":
            inputs["cube_rgb"] = cube_rgb
            inputs["normalized_cube_rgb"] = self.normalize(cube_aug_rgb)

        inputs["gt_depth"] = torch.from_numpy(np.expand_dims(gt_depth, axis=0)).to(torch.float32)
        inputs["val_mask"] = ((inputs["gt_depth"] > 0) & (inputs["gt_depth"] <= self.max_depth_meters )
                                & ~torch.isnan(inputs["gt_depth"]))

        inputs['spec_binaural'] = spec_binaural
        inputs['index'] = index

        return inputs


    def __len__(self):
        return len(self.rgb_paths)





if __name__ == "__main__":
    from tqdm import tqdm
    import ipdb
    

    dataset = Dataload_RealEquirectangular_PanoCacheObs(data_path="./dataset_realEquirec_mp3d_organized", mode="train",disable_color_augmentation = True, disable_LR_filp_augmentation = True,
                                     disable_yaw_rotation_augmentation = True, is_training=True, dataset_use_ratio = 0.1, dataset = "mp3d")

    for item in tqdm(dataset):
        print(item['gt_depth'].max())
        ipdb.set_trace()
        #print(1)
    