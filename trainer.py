from __future__ import absolute_import, division, print_function
import os

import numpy as np
import time
import json
import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter
torch.manual_seed(100)
torch.cuda.manual_seed(100)
from metrics import compute_depth_metrics, Evaluator, EarlyStopperAcc
from losses import BerhuLoss
import loss_gradient as loss_g
from network.model import Panoformer as PanoBiT
from network.networks_unifuse import UniFuse
from network.networks_bifuse import MyModel as Bifuse
from dataload import Dataload_RealEquirectangular_PanoCacheObs
import random
import pandas as pd
import ipdb

import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
import torch.functional as F
import requests
import torchvision
from PIL import Image

def gradient(x):
    gradient_model = loss_g.Gradient_Net()
    g_x, g_y = gradient_model(x)
    return g_x, g_y


class Trainer:
    def __init__(self, settings):
        self.settings = settings
        
        def setup_seed(seed):
            np.random.seed(seed)
            random.seed(seed)
            torch.manual_seed(seed)  # cpu
            torch.cuda.manual_seed_all(seed)  
            torch.backends.cudnn.deterministic = True  
            torch.backends.cudnn.benchmark = False  
            print("random seed set: {}".format(seed))

        setup_seed(seed = self.settings.random_seed)

        
        self.device = torch.device("cuda" if len(self.settings.gpu_devices) else "cpu")
        self.gpu_devices = ','.join([str(id) for id in settings.gpu_devices])
        os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu_devices
        print("self.gpu_devices: ",self.gpu_devices)
        

        self.log_path = os.path.join(self.settings.log_dir, self.settings.model_name)
        print("dataset {} used.\n".format(self.settings.dataset))
        
        self.MAX_DEPTH = 16
        print("self.MAX_DEPTH defined = {}".format(self.MAX_DEPTH))
        data_loader = Dataload_RealEquirectangular_PanoCacheObs

        train_dataset = data_loader(data_path="./prepare_datasets/dataset_realEquirec_{}_organized".format(self.settings.dataset), mode="train",disable_color_augmentation = self.settings.disable_color_augmentation, disable_LR_filp_augmentation = self.settings.disable_LR_filp_augmentation,
                                        disable_yaw_rotation_augmentation = self.settings.disable_yaw_rotation_augmentation, is_training=True, dataset_use_ratio = self.settings.dataset_use_ratio, dataset = self.settings.dataset, model=self.settings.model)
        
        SHUFFLE = True
        DROP_LAST = True
            
        self.train_loader = DataLoader(train_dataset, self.settings.batch_size, SHUFFLE,
                                       num_workers=self.settings.num_workers, pin_memory=True, drop_last=DROP_LAST)
        num_train_samples = len(train_dataset)
        self.num_total_steps = num_train_samples // self.settings.batch_size * self.settings.num_epochs


        val_dataset = data_loader(data_path="./prepare_datasets/dataset_realEquirec_{}_organized".format(self.settings.dataset), mode="val",disable_color_augmentation = self.settings.disable_color_augmentation, disable_LR_filp_augmentation = self.settings.disable_LR_filp_augmentation,
                                        disable_yaw_rotation_augmentation = self.settings.disable_yaw_rotation_augmentation, is_training=False, dataset_use_ratio = self.settings.dataset_use_ratio, dataset = self.settings.dataset, model=self.settings.model)
       

        self.val_loader = DataLoader(val_dataset, self.settings.batch_size, False,
                                     num_workers=self.settings.num_workers, pin_memory=True, drop_last=DROP_LAST)


        test_dataset = data_loader(data_path="./prepare_datasets/dataset_realEquirec_{}_organized".format(self.settings.dataset), mode="test",disable_color_augmentation = self.settings.disable_color_augmentation, disable_LR_filp_augmentation = self.settings.disable_LR_filp_augmentation,
                                     disable_yaw_rotation_augmentation = self.settings.disable_yaw_rotation_augmentation, is_training=False, dataset_use_ratio = self.settings.dataset_use_ratio, dataset = self.settings.dataset, model=self.settings.model)
 

        self.test_loader = DataLoader(test_dataset, 1, False,
                                     num_workers=self.settings.num_workers, pin_memory=True, drop_last=DROP_LAST)
        
        self.model_mode = self.settings.model_mode
        print("self.model_mode = {}".format(self.model_mode))

        if self.settings.model == "Unifuse": 
            print("Unifuse used.\n")
            self.model =  UniFuse(model_mode = self.model_mode)

        elif self.settings.model == "PanoFormer":    
            print("PanoFormer used.\n")              
            self.model = PanoBiT(model_mode = self.model_mode )
            
        elif self.settings.model == "Bifuse": 
            print("Bifuse used.\n")  
            self.model = Bifuse(
                layers=18,
                decoder="upproj",
                output_size=None,
                in_channels=3,
                pretrained=True, 
                model_mode = self.model_mode 
                )
        
        self.model.cuda()
       

        self.parameters_to_train = list(self.model.parameters())

        if self.settings.optimiser == "Adam":
            print("Adam optimiser used\n")
            self.optimizer = optim.Adam(self.parameters_to_train, self.settings.learning_rate)
        elif self.settings.optimiser == "SGD":
            self.optimizer = torch.optim.SGD(self.parameters_to_train, lr=self.settings.learning_rate, momentum=0.9, nesterov = True)
            print("SGD with nesterov optimiser used\n")
        else:
            raise NotImplementedError

        if self.settings.load_weights_dir is not None:
            self.load_model()

        print("Training model named:\n ", self.settings.model_name)
        print("Models and tensorboard events files are saved to:\n", self.settings.log_dir)
        print("Training is using:\n ", self.device)

        self.compute_loss = BerhuLoss()
        self.evaluator = Evaluator()

        self.writers = {}
        
        if self.settings.mode != "test":
            for mode in ["train", "val"]:
                self.writers[mode] = SummaryWriter(os.path.join(self.log_path, self.settings.exp_name ,mode))
            self.save_settings()
        
        self.early_stopper = EarlyStopperAcc(patience=20, min_delta=0.001)
        self.save_pickle = self.settings.save_pickle

    def train(self):
        """Run the entire training pipeline
        """
        self.epoch = 0
        self.step = 0
        self.start_time = time.time()
        self.best_acca1 = 0

        for self.epoch in range(self.settings.num_epochs):
            self.train_one_epoch()
            losses = self.validate()
            
            if losses["acc/a1"] > self.best_acca1:
                self.best_acca1 = losses["acc/a1"]
                print("new best acc_a1:{}, model saved.\n".format(self.best_acca1))
                self.save_model()
            
            if self.early_stopper.early_stop(losses["acc/a1"]): 
                print('early stopped')            
                break

                
    def train_one_epoch(self):
        """Run a single epoch of training
        """
        self.model.train()

        pbar = tqdm.tqdm(self.train_loader)
        pbar.set_description("Training Epoch_{}".format(self.epoch))

        for batch_idx, inputs in enumerate(pbar):

            outputs, losses, runtime = self.process_batch(inputs)

            self.optimizer.zero_grad()
            losses["loss"].backward()
            self.optimizer.step()

            # log less frequently after the first 1000 steps to save time & disk space
            early_phase = batch_idx % self.settings.log_frequency == 0 and self.step < 1000
            late_phase = self.step % 1000 == 0

            if early_phase or late_phase:

                pred_depth = outputs["pred_depth"].detach()
                gt_depth = inputs["gt_depth"]
                mask = inputs["val_mask"]
                
                pred_depth = outputs["pred_depth"].detach() * mask
                gt_depth = inputs["gt_depth"] * mask
                
                depth_errors = compute_depth_metrics(gt_depth, pred_depth, mask)
                for i, key in enumerate(self.evaluator.metrics.keys()):
                    losses[key] = np.array(depth_errors[i].cpu())

                self.log("train", inputs, outputs, losses)

            self.step += 1

    def process_batch(self, inputs):
        for key, ipt in inputs.items():
            if key not in ["rgb"]:
                inputs[key] = ipt.cuda()

        losses = {}

        equi_inputs = inputs["normalized_rgb"].to(torch.float32).cuda()# * inputs["val_mask"]
        audio_inputs = inputs['spec_binaural'].cuda()

        start = time.time()
        
        if self.settings.model == "Unifuse": 
            cube_inputs = inputs["normalized_cube_rgb"].cuda()
            outputs = self.model(equi_inputs, cube_inputs, audio_inputs)
        elif self.settings.model == "PanoFormer" or self.settings.model == "Bifuse":   
            outputs = self.model(equi_inputs, audio_inputs)
            
        end = time.time()
        run_time = end - start

        gt = inputs["gt_depth"] * inputs["val_mask"]
        pred = outputs["pred_depth"] * inputs["val_mask"]
        outputs["pred_depth"] = outputs["pred_depth"] * inputs["val_mask"]

        G_x, G_y = gradient(gt.float())
        p_x, p_y = gradient(pred)

        losses["loss"] = self.compute_loss(inputs["gt_depth"].float
                                            () * inputs["val_mask"], outputs["pred_depth"]) +\
                         self.compute_loss(G_x, p_x) +\
                         self.compute_loss(G_y, p_y)

        return outputs, losses, run_time

    def validate(self):
        """Validate the model on the validation set
        """
        self.model.eval()

        self.evaluator.reset_eval_metrics()

        pbar = tqdm.tqdm(self.val_loader)
        pbar.set_description("Validating Epoch_{}".format(self.epoch))

        with torch.no_grad():
            for batch_idx, inputs in enumerate(pbar):
                outputs, losses, _ = self.process_batch(inputs)
                pred_depth = outputs["pred_depth"].detach() * inputs["val_mask"]
                gt_depth = inputs["gt_depth"] * inputs["val_mask"]
                
                self.evaluator.compute_eval_metrics(gt_depth, pred_depth)

        self.evaluator.print()

        for i, key in enumerate(self.evaluator.metrics.keys()):
            losses[key] = np.array(self.evaluator.metrics[key].avg.cpu())
        self.log("val", inputs, outputs, losses)
        del inputs, outputs
        return losses

    def test(self):
        """test the model on the test set
        """
 
        assert self.settings.load_weights_dir is not None
        print("loaded from {}".format(self.settings.load_weights_dir))
        
        self.model.eval()

        self.evaluator.reset_eval_metrics()

        pbar = tqdm.tqdm(self.test_loader)
        pbar.set_description("Testing")
        input_list = []
        pred_list = []
        gt_list = []
        losses_list = []
        batch_idx_list = []
        depth_pred_list = []
        run_time_list = []
        
        
        with torch.no_grad():
            for batch_idx, inputs in enumerate(pbar):
                outputs, losses, run_time = self.process_batch(inputs)
                pred_depth = outputs["pred_depth"].detach() * inputs["val_mask"]
                gt_depth = inputs["gt_depth"] * inputs["val_mask"]

                for key in inputs.keys():
                    inputs[key] = inputs[key].detach().cpu().numpy()
                
                if self.save_pickle:
                    input_list.append(inputs)
                    pred_list.append(pred_depth.detach().cpu().numpy())
                    
                    gt_list.append(gt_depth.detach().cpu().numpy())
                    for key in losses.keys():
                        losses[key] = losses[key].detach().cpu().numpy()
                    losses_list.append(losses)
                    batch_idx_list.append(batch_idx)
                    depth_pred_list.append({int(inputs['index']):pred_depth.detach().cpu().numpy()})
                    run_time_list.append(run_time)
                
                return_dict = self.evaluator.compute_eval_metrics(gt_depth, pred_depth)

        self.evaluator.print()

        d = {'input_list':input_list, 'pred_list': pred_list, "gt_list": gt_list,  "batch_idx_list": batch_idx_list, "losses_list":losses_list}
        stats_df = pd.DataFrame(data=d)
        if self.save_pickle:
            print("self.save_pickle = {}, saving pickle file".format(self.save_pickle))
            stats_df.to_pickle("./tmp/tmp_test_results_{}.pkl".format( self.settings.exp_name))
        
        for i, key in enumerate(self.evaluator.metrics.keys()):
            losses[key] = np.array(self.evaluator.metrics[key].avg.cpu())
        #self.log("val", inputs, outputs, losses)

        del inputs, outputs, losses
   
    
    def log(self, mode, inputs, outputs, losses):
        """Write an event to the tensorboard events file
        """
        outputs["pred_depth"] = outputs["pred_depth"] * inputs["val_mask"]
        inputs["gt_depth"] = inputs["gt_depth"] * inputs["val_mask"]
        writer = self.writers[mode]
        for l, v in losses.items():
            writer.add_scalar("{}".format(l), v, self.step)

        
    def save_settings(self):
        """Save settings to disk so we know what we ran this experiment with
        """
        models_dir = os.path.join(self.log_path, self.settings.exp_name ,"models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        to_save = self.settings.__dict__.copy()

        with open(os.path.join(models_dir, 'settings.json'), 'w') as f:
            json.dump(to_save, f, indent=2)

    def save_model(self):
        """Save model weights to disk
        """
        #save_folder = os.path.join(self.log_path, self.settings.exp_name, "models", "weights_{}".format(self.epoch))
        save_folder = os.path.join(self.log_path, self.settings.exp_name, "models", "best")
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        save_path = os.path.join(save_folder, "{}.pth".format("model"))
        to_save = self.model.state_dict()
        torch.save(to_save, save_path)

        save_path = os.path.join(save_folder, "{}.pth".format("adam"))
        torch.save(self.optimizer.state_dict(), save_path)
    

    def load_model(self):
        """Load model from disk
        """
        self.settings.load_weights_dir = os.path.expanduser(self.settings.load_weights_dir)

        assert os.path.isdir(self.settings.load_weights_dir), \
            "Cannot find folder {}".format(self.settings.load_weights_dir)
        print("loading model from folder {}".format(self.settings.load_weights_dir))

        path = os.path.join(self.settings.load_weights_dir, "{}.pth".format("model"))
        model_dict = self.model.state_dict()
        pretrained_dict = torch.load(path)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.model.load_state_dict(model_dict)

        # loading adam state
        optimizer_load_path = os.path.join(self.settings.load_weights_dir, "adam.pth")
        if os.path.isfile(optimizer_load_path):
            print("Loading Adam weights")
            optimizer_dict = torch.load(optimizer_load_path)
            self.optimizer.load_state_dict(optimizer_dict)
        else:
            print("Cannot find Adam weights so Adam is randomly initialized")


