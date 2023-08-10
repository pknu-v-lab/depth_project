# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import numpy as np
import time

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
torch.backends.cudnn.enabled=False
import json

from utils import *
from kitti_utils import *
from layers import *

import datasets
import networks
from IPython import embed


class Trainer:
    def __init__(self, options):
        self.opt = options
        self.log_path = os.path.join(self.opt.log_dir, self.opt.model_name)

        # checking height and width are multiples of 32
        assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"

        self.models = {}
        self.parameters_to_train = []
        self.parameters_blur_train = []

        self.device = torch.device("cpu" if self.opt.no_cuda else "cuda")

        self.num_scales = len(self.opt.scales)
        self.num_input_frames = len(self.opt.frame_ids)
        self.num_pose_frames = 2 if self.opt.pose_model_input == "pairs" else self.num_input_frames

        assert self.opt.frame_ids[0] == 0, "frame_ids must start with 0"

        self.use_pose_net = not (self.opt.use_stereo and self.opt.frame_ids == [0])

        if self.opt.use_stereo:
            self.opt.frame_ids.append("s")

<<<<<<< HEAD
        
        ###################### sharp network ######################
        
        self.models["encoder"] = networks.ResnetEncoder(
            self.opt.num_layers, self.opt.weights_init == "pretrained")
        
=======
        # blur network 선언
        self.models["blur_encoder"] = networks.ResnetEncoder(
            self.opt.num_layers, self.opt.weights_init == "pretrained")
        self.models["blur_encoder"].to(self.device)
        self.parameters_blur_train += list(self.models["blur_encoder"].parameters())
        self.models["blur_depth"] = networks.DepthDecoder(
            self.models["blur_encoder"].num_ch_enc, self.opt.scales)
        self.models["blur_depth"].to(self.device)
        self.parameters_blur_train += list(self.models["blur_depth"].parameters())
        
        # shar network 선언
        self.models["encoder"] = networks.ResnetEncoder(
            self.opt.num_layers, self.opt.weights_init == "pretrained")
        self.models["encoder"].to(self.device)
        self.parameters_to_train += list(self.models["encoder"].parameters())
>>>>>>> 89f0717980b33cf40d74f2e6d192ee8e7f485e5c
        self.models["depth"] = networks.DepthDecoder(
            self.models["encoder"].num_ch_enc, self.opt.scales)
        
    
        
        
        self.models["encoder"].to(self.device)
        self.parameters_to_train += list(self.models["encoder"].parameters())
        self.models["depth"].to(self.device)
        self.parameters_to_train += list(self.models["depth"].parameters())

        if self.use_pose_net:
<<<<<<< HEAD
             
            self.models["pose_encoder"] = networks.ResnetEncoder(
                self.opt.num_layers,
                self.opt.weights_init == "pretrained",
                num_input_images=self.num_pose_frames)
             
            self.models["pose_encoder"].to(self.device)
             
            self.parameters_to_train += list(self.models["pose_encoder"].parameters())

            self.models["pose"] = networks.PoseDecoder(
                self.models["pose_encoder"].num_ch_enc,
                num_input_features=1,
                num_frames_to_predict_for=2)
=======
            if self.opt.pose_model_type == "separate_resnet":
                self.models["pose_encoder"] = networks.ResnetEncoder(
                    self.opt.num_layers,
                    self.opt.weights_init == "pretrained",
                    num_input_images=self.num_pose_frames)
                
                self.models["blur_pose_encoder"] = networks.ResnetEncoder(
                    self.opt.num_layers,
                    self.opt.weights_init == "pretrained",
                    num_input_images=self.num_pose_frames)

                self.models["pose_encoder"].to(self.device)
                self.parameters_to_train += list(self.models["pose_encoder"].parameters())
                
                self.models["blur_pose_encoder"].to(self.device)
                self.parameters_blur_train += list(self.models["blur_pose_encoder"].parameters())

                self.models["blur_pose"] = networks.PoseDecoder(
                    self.models["blur_pose_encoder"].num_ch_enc,
                    num_input_features=1,
                    num_frames_to_predict_for=2)
                
                self.models["pose"] = networks.PoseDecoder(
                    self.models["pose_encoder"].num_ch_enc,
                    num_input_features=1,
                    num_frames_to_predict_for=2)

            elif self.opt.pose_model_type == "shared":
                self.models["blur_pose"] = networks.PoseDecoder(
                    self.models["blur_encoder"].num_ch_enc, self.num_pose_frames)
                
                self.models["pose"] = networks.PoseDecoder(
                    self.models["encoder"].num_ch_enc, self.num_pose_frames)

            elif self.opt.pose_model_type == "posecnn":
                self.models["blur_pose"] = networks.PoseCNN(
                    self.num_input_frames if self.opt.pose_model_input == "all" else 2)
                
                self.models["pose"] = networks.PoseCNN(
                    self.num_input_frames if self.opt.pose_model_input == "all" else 2)
>>>>>>> 89f0717980b33cf40d74f2e6d192ee8e7f485e5c

            self.models["blur_pose"].to(self.device)
            self.models["pose"].to(self.device)
<<<<<<< HEAD

=======
            self.parameters_blur_train += list(self.models["blur_pose"].parameters())
>>>>>>> 89f0717980b33cf40d74f2e6d192ee8e7f485e5c
            self.parameters_to_train += list(self.models["pose"].parameters())


        if self.opt.predictive_mask:
            assert self.opt.disable_automasking, \
                "When using predictive_mask, please disable automasking with --disable_automasking"

            # Our implementation of the predictive masking baseline has the the same architecture
            # as our depth decoder. We predict a separate mask for each source frame.
            self.models["blur_predictive_mask"] = networks.DepthDecoder(
                self.models["blur_encoder"].num_ch_enc, self.opt.scales,
                num_output_channels=(len(self.opt.frame_ids) - 1))
            self.models["blur_predictive_mask"].to(self.device)
            self.parameters_blur_train += list(self.models["blur_predictive_mask"].parameters())
            
            self.models["predictive_mask"] = networks.DepthDecoder(
                self.models["encoder"].num_ch_enc, self.opt.scales,
                num_output_channels=(len(self.opt.frame_ids) - 1))
            self.models["predictive_mask"].to(self.device)
            self.parameters_to_train += list(self.models["predictive_mask"].parameters())
            

        self.model_optimizer = optim.Adam(self.parameters_to_train, self.opt.learning_rate)
<<<<<<< HEAD
        
=======
        self.blur_model_optimizer = optim.Adam(self.parameters_blur_train, self.opt.learning_rate)
>>>>>>> 89f0717980b33cf40d74f2e6d192ee8e7f485e5c
        
        self.model_lr_scheduler = optim.lr_scheduler.StepLR(
            self.model_optimizer, self.opt.scheduler_step_size, 0.1)
        self.blur_model_lr_scheduler = optim.lr_scheduler.StepLR(
            self.blur_model_optimizer, self.opt.scheduler_step_size, 0.1)

        if self.opt.load_weights_folder is not None:
            self.load_model()
            

        print("Training model named:\n  ", self.opt.model_name)
        print("Models and tensorboard events files are saved to:\n  ", self.opt.log_dir)
        print("Training is using:\n  ", self.device)

        # data
        datasets_dict = {"kitti": datasets.KITTIRAWDataset,
                         "kitti_odom": datasets.KITTIOdomDataset}
        self.dataset = datasets_dict[self.opt.dataset]

        fpath = os.path.join("C:\\Users\\user\\monodepth2\\monodepth2", "splits", self.opt.split, "{}_files.txt")

        train_filenames = readlines(fpath.format("train"))
        val_filenames = readlines(fpath.format("val"))
        img_ext = '.png' if self.opt.png else '.jpg'

        num_train_samples = len(train_filenames)
        self.num_total_steps = num_train_samples // self.opt.batch_size * self.opt.num_epochs

        train_dataset = self.dataset(
            self.opt.data_path, train_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, 4, is_train=True, img_ext=img_ext)
        self.train_loader = DataLoader(
            train_dataset, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        val_dataset = self.dataset(
            self.opt.data_path, val_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, 4, is_train=False, img_ext=img_ext)
        self.val_loader = DataLoader(
            val_dataset, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        self.val_iter = iter(self.val_loader)

        self.writers = {}

        for mode in ["train", "val"]:
            self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))

        if not self.opt.no_ssim:
            self.ssim = SSIM()
            self.ssim.to(self.device)

        self.backproject_depth = {}
        self.project_3d = {}
        for scale in self.opt.scales:
            h = self.opt.height // (2 ** scale)
            w = self.opt.width // (2 ** scale)

            self.backproject_depth[scale] = BackprojectDepth(self.opt.batch_size, h, w)
            self.backproject_depth[scale].to(self.device)

            self.project_3d[scale] = Project3D(self.opt.batch_size, h, w)
            self.project_3d[scale].to(self.device)

        self.depth_metric_names = [
            "de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]

        print("Using split:\n  ", self.opt.split)
        print("There are {:d} training items and {:d} validation items\n".format(
            len(train_dataset), len(val_dataset)))

        self.save_opts()

    def set_train(self):
        """Convert all models to training mode
        """
        for m in self.models.values():
            m.train()

    def set_eval(self):
        """Convert all models to testing/evaluation mode
        """
        for m in self.models.values():
            m.eval()

    def train(self):
        """Run the entire training pipeline
        """
        self.epoch = 0
        self.step = 0
        self.start_time = time.time()
        for self.epoch in range(self.opt.num_epochs):
            self.run_epoch()
            if (self.epoch + 1) % self.opt.save_frequency == 0:
                self.save_model()

    def run_epoch(self):
        """Run a single epoch of training and validation
        """
        
        self.model_lr_scheduler.step()
        self.blur_model_lr_scheduler.step()

        print("Training")
        self.set_train()

        for batch_idx, inputs in enumerate(self.train_loader):
<<<<<<< HEAD
        
            before_op_time = time.time()
           
    
           
           #### Blur(student) Model 학습 #### 
            outputs, losses = self.process_batch(inputs)
=======
            
            i_type = ['sharp', 'blur']
            
            b_inputs = inputs[1]
            sharp_inputs = inputs[0]
            
            before_op_time = time.time()
           #### Sharp(teacher) Model 학습 ####
            outputs, losses = self.process_batch(sharp_inputs, i_type[0]) 

>>>>>>> 89f0717980b33cf40d74f2e6d192ee8e7f485e5c
            self.model_optimizer.zero_grad()
            final_loss = losses["loss"] + losses["reg_loss"] if losses["reg_loss"] is not None else losses["loss"]
            final_loss.backward()
            self.model_optimizer.step()
<<<<<<< HEAD
=======
           ##################################
           
           #### Blur(student) Model 학습 #### 
            blur_outputs, blur_losses = self.process_batch(b_inputs, i_type[1])
            blur_losses = self.process_batch_blur(outputs, blur_outputs, blur_losses)
            self.blur_model_optimizer.zero_grad()
            blur_losses["loss"].backward()
            self.blur_model_optimizer.step()
>>>>>>> 89f0717980b33cf40d74f2e6d192ee8e7f485e5c
            
           ##################################

            duration = time.time() - before_op_time

            # log less frequently after the first 2000 steps to save time & disk space
            early_phase = batch_idx % self.opt.log_frequency == 0 and self.step < 2000
            late_phase = self.step % 2000 == 0

            if early_phase or late_phase:
<<<<<<< HEAD
                if losses["reg_loss"] is not None:
                    self.log_time(batch_idx, duration, losses["loss"].cpu().data, losses["reg_loss"].item())
                else:
                    self.log_time(batch_idx, duration, losses["loss"].cpu().data)

                if "depth_gt" in inputs[0]:
                    self.compute_depth_losses(inputs[0], outputs, losses)

                self.log("train", inputs[0], outputs, losses)
=======
                self.log_time(batch_idx, duration, blur_losses["loss"].cpu().data)

                if "depth_gt" in inputs[1]:
                    self.compute_depth_losses(inputs[1], blur_outputs, blur_losses)

                self.log("train", inputs[1], blur_outputs, blur_losses)
>>>>>>> 89f0717980b33cf40d74f2e6d192ee8e7f485e5c
                self.val()

            self.step += 1

<<<<<<< HEAD
    def process_batch_blur(self, temp, b_outputs, losses):
        
        regression_loss = self.regress_loss(temp, b_outputs)
        losses["reg_loss"] = regression_loss
=======
    def process_batch_blur(self, outputs, blur_outputs, losses):
        
        # for key, item in outputs.items():
            # outputs[key].detach()
        # regression_loss = self.regress_loss(outputs,blur_outputs)
>>>>>>> 89f0717980b33cf40d74f2e6d192ee8e7f485e5c
        # losses["loss"] += regression_loss
        
        return losses 
    
<<<<<<< HEAD
    def process_batch(self, inputs):
        """Pass a minibatch through the network and generate images and losses
        """
        if type(inputs) == list:
            b_inputs = inputs[1]
            s_inputs = inputs[0]    
        
        else:
            b_inputs = inputs
        
        for key, ipt in b_inputs.items():
                b_inputs[key] = ipt.to(self.device)
   
        # Otherwise, we only feed the image with frame_id 0 through the depth encoder
        
        b_features = self.models["encoder"](b_inputs["color_aug", 0, 0])
        b_outputs = self.models["depth"](b_features)

        if self.opt.predictive_mask:
                b_outputs["predictive_mask"] = self.models["predictive_mask"](b_features)
                    
        b_outputs.update(self.predict_poses(b_inputs, b_features))

=======
    def process_batch(self, inputs, type):
        """Pass a minibatch through the network and generate images and losses
        """
        
        
        for key, ipt in inputs.items():
            inputs[key] = ipt.to(self.device)

        if self.opt.pose_model_type == "shared":
            # If we are using a shared encoder for both depth and pose (as advocated
            # in monodepthv1), then all images are fed separately through the depth encoder.
            
            if type == "blur":
                all_color_aug = torch.cat([inputs[("color_aug", i, 0)] for i in self.opt.frame_ids])
                all_features = self.models["blur_encoder"](all_color_aug)
                all_features = [torch.split(f, self.opt.batch_size) for f in all_features]

                features = {}
                for i, k in enumerate(self.opt.frame_ids):
                    features[k] = [f[i] for f in all_features]

                outputs = self.models["blur_depth"](features[0])
                
            else:
                all_color_aug = torch.cat([inputs[("color_aug", i, 0)] for i in self.opt.frame_ids])
                all_features = self.models["encoder"](all_color_aug)
                all_features = [torch.split(f, self.opt.batch_size) for f in all_features]

                features = {}
                for i, k in enumerate(self.opt.frame_ids):
                    features[k] = [f[i] for f in all_features]

                outputs = self.models["depth"](features[0])
                
                
        else:
            # Otherwise, we only feed the image with frame_id 0 through the depth encoder
            if type == "blur":
                features = self.models["blur_encoder"](inputs["color_aug", 0, 0])
                outputs = self.models["blur_depth"](features)
            
            else:  
                features = self.models["encoder"](inputs["color_aug", 0, 0])
                outputs = self.models["depth"](features)

        if self.opt.predictive_mask:
            if type == "blur":
                outputs["predictive_mask"] = self.models["predictive_mask"](features)
                
            else:
                outputs["blur_predictive_mask"] = self.models["blur_predictive_mask"](features)

        if self.use_pose_net:
            outputs.update(self.predict_poses(inputs, features, type))
>>>>>>> 89f0717980b33cf40d74f2e6d192ee8e7f485e5c

        self.generate_images_pred(b_inputs, b_outputs)
        losses = self.compute_losses(b_inputs, b_outputs)
        
        if type(inputs) == list:
            for key, ipt in s_inputs.items():
                s_inputs[key] = ipt.to(self.device)   
            s_features = self.models["encoder"](s_inputs["color_aug", 0, 0])
            s_outputs = self.models["depth"](s_features)
            
            if self.opt.predictive_mask:
                s_outputs["predictive_mask"] = self.models["predictive_mask"](s_features)
            
            s_outputs.update(self.predict_poses(s_inputs, s_features))
            
            self.generate_images_pred(s_inputs, s_outputs)
            s_losses = self.compute_losses(s_inputs, s_outputs)
            
            for key, value in s_losses.items():
                losses[key] += value

            temp = {}
            for key, item in s_outputs.items():
                temp[key] = s_outputs[key].detach()
        
            losses = self.process_batch_blur(temp, b_outputs, losses)
            

        return b_outputs, losses

    def predict_poses(self, inputs, features, type):
        """Predict poses between input frames for monocular sequences.
        """
        outputs = {}
        if self.num_pose_frames == 2:
            # In this setting, we compute the pose to each source frame via a
            # separate forward pass through the pose network.

            
            
            pose_feats = {f_i: inputs["color_aug", f_i, 0] for f_i in self.opt.frame_ids}

            for f_i in self.opt.frame_ids[1:]:
                if f_i != "s":
                    # To maintain ordering we always pass frames in temporal order
                    if f_i < 0:
                        pose_inputs = [pose_feats[f_i], pose_feats[0]]
                    else:
                        pose_inputs = [pose_feats[0], pose_feats[f_i]]

<<<<<<< HEAD
                    
                        
                    pose_inputs = [self.models["pose_encoder"](torch.cat(pose_inputs, 1))]
                        
                    axisangle, translation = self.models["pose"](pose_inputs)
=======
                    if self.opt.pose_model_type == "separate_resnet":
                        if type == "blur":
                            pose_inputs = [self.models["blur_pose_encoder"](torch.cat(pose_inputs, 1))]
                        else:
                            pose_inputs = [self.models["pose_encoder"](torch.cat(pose_inputs, 1))]
                    elif self.opt.pose_model_type == "posecnn":
                        pose_inputs = torch.cat(pose_inputs, 1)
                    
                    if type == "blur":    
                        axisangle, translation = self.models["blur_pose"](pose_inputs)
                    else:
                        axisangle, translation = self.models["pose"](pose_inputs)
>>>>>>> 89f0717980b33cf40d74f2e6d192ee8e7f485e5c
                        
                    outputs[("axisangle", 0, f_i)] = axisangle
                    outputs[("translation", 0, f_i)] = translation

                    # Invert the matrix if the frame id is negative
                    outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0], invert=(f_i < 0))

        else:
            # Here we input all frames to the pose net (and predict all poses) together
            if self.opt.pose_model_type in ["separate_resnet", "posecnn"]:
                pose_inputs = torch.cat(
                    [inputs[("color_aug", i, 0)] for i in self.opt.frame_ids if i != "s"], 1)

                if self.opt.pose_model_type == "separate_resnet":
<<<<<<< HEAD
                    
                 
                    pose_inputs = [self.models["pose_encoder"](pose_inputs)]
=======
                    if type == "blur":
                        pose_inputs = [self.models["blur_pose_encoder"](pose_inputs)]
                    else:
                        pose_inputs = [self.models["pose_encoder"](pose_inputs)]
>>>>>>> 89f0717980b33cf40d74f2e6d192ee8e7f485e5c

            elif self.opt.pose_model_type == "shared":
                pose_inputs = [features[i] for i in self.opt.frame_ids if i != "s"]
                
<<<<<<< HEAD
            
            axisangle, translation = self.models["pose"](pose_inputs)
=======
            if type == "blur":
                axisangle, translation = self.models["blur_pose"](pose_inputs)
            
            else:
                axisangle, translation = self.models["pose"](pose_inputs)
>>>>>>> 89f0717980b33cf40d74f2e6d192ee8e7f485e5c

            for i, f_i in enumerate(self.opt.frame_ids[1:]):
                if f_i != "s":
                    outputs[("axisangle", 0, f_i)] = axisangle
                    outputs[("translation", 0, f_i)] = translation
                    outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                        axisangle[:, i], translation[:, i])

        return outputs

    def val(self):
        """Validate the model on a single minibatch
        """
<<<<<<< HEAD
        
        self.set_eval()
        try:
            inputs = self.val_iter.next()
            s_inputs, b_inputs = inputs[0], inputs[1]
        except StopIteration:
            self.val_iter = iter(self.val_loader)
            inputs = self.val_iter.next()
            s_inputs, b_inputs = inputs[0], inputs[1]
            
        with torch.no_grad():
            outputs, losses = self.process_batch(b_inputs)
=======
        i_type = ['sharp', 'blur']
        self.set_eval()
        try:
            inputs, b_inputs = self.val_iter.next()
        except StopIteration:
            self.val_iter = iter(self.val_loader)
            inputs, b_inputs = self.val_iter.next()

        with torch.no_grad():
            outputs, losses = self.process_batch(inputs,i_type[0])
>>>>>>> 89f0717980b33cf40d74f2e6d192ee8e7f485e5c

            if "depth_gt" in s_inputs:
                self.compute_depth_losses(s_inputs, outputs, losses)

<<<<<<< HEAD
            self.log("val", s_inputs, outputs, losses)
            
    
            del inputs, outputs, losses, b_inputs, s_inputs
=======
            self.log("val", inputs, outputs, losses)
            
        
        with torch.no_grad():
            blur_outputs, blur_losses = self.process_batch(b_inputs, i_type[1])
            blur_losses = self.process_batch_blur(self, outputs, blur_losses, blur_losses)

            if "depth_gt" in inputs:
                self.compute_depth_losses(b_inputs, blur_outputs, blur_losses)

            self.log("val", b_inputs, blur_outputs, blur_losses)
            del inputs, outputs, losses, b_inputs, blur_outputs, blur_losses
>>>>>>> 89f0717980b33cf40d74f2e6d192ee8e7f485e5c

        self.set_train()

    def generate_images_pred(self, inputs, outputs):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        for scale in self.opt.scales:
            disp = outputs[("disp", scale)]
            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                disp = F.interpolate(
                    disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
                source_scale = 0

            _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)

            outputs[("depth", 0, scale)] = depth

            for i, frame_id in enumerate(self.opt.frame_ids[1:]):

                if frame_id == "s":
                    T = inputs["stereo_T"]
                else:
                    T = outputs[("cam_T_cam", 0, frame_id)]

                # from the authors of https://arxiv.org/abs/1712.00175
                if self.opt.pose_model_type == "posecnn":

                    axisangle = outputs[("axisangle", 0, frame_id)]
                    translation = outputs[("translation", 0, frame_id)]

                    inv_depth = 1 / depth
                    mean_inv_depth = inv_depth.mean(3, True).mean(2, True)

                    T = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0] * mean_inv_depth[:, 0], frame_id < 0)

                cam_points = self.backproject_depth[source_scale](
                    depth, inputs[("inv_K", source_scale)])
                pix_coords = self.project_3d[source_scale](
                    cam_points, inputs[("K", source_scale)], T)

                outputs[("sample", frame_id, scale)] = pix_coords

                outputs[("color", frame_id, scale)] = F.grid_sample(
                    inputs[("color", frame_id, source_scale)],
                    outputs[("sample", frame_id, scale)],
                    padding_mode="border")

                if not self.opt.disable_automasking:
                    outputs[("color_identity", frame_id, scale)] = \
                        inputs[("color", frame_id, source_scale)]
<<<<<<< HEAD
    
    # feature distillation                    
    def feature_loss(self, s_outputs, b_outputs):
        l1_loss = 0
        for i in range(4, -1, -1):
            l1_loss += F.l1_loss(b_outputs[("upconv", i, 1)], s_outputs[("upconv", i, 1)])
            # abs_diff += torch.abs(s_outputs[("upconv", i, 1)]  - b_outputs[("upconv", i, 1)]).mean(1, True)
            # l1_loss += abs_diff.mean()

        return l1_loss
    
    def regress_loss(self, s_outputs, b_outputs):
        feature_loss = self.feature_loss(s_outputs, b_outputs)
        disp_loss = F.l1_loss(b_outputs["disp", 0], s_outputs["disp", 0])
        loss = (feature_loss + disp_loss) * self.opt.feature_loss_coefficient
        
        return loss
=======
                        
    # def regress_loss(self, outputs_t, outputs):
    #     losses ={}
    #     abs_diff = torch.abs(outputs[("disp",0)] - outputs_t[("disp",0)])
    #     uncerted_l1_loss = ( abs_diff / outputs[("uncert",0)] + torch.log(outputs[("uncert",0)])).mean()
    #     return uncerted_l1_loss
>>>>>>> 89f0717980b33cf40d74f2e6d192ee8e7f485e5c

    def compute_reprojection_loss(self, pred, target):
        """Computes reprojection loss between a batch of predicted and target images
        """
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)

        if self.opt.no_ssim:
            reprojection_loss = l1_loss
        else:
            ssim_loss = self.ssim(pred, target).mean(1, True)
            reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

        return reprojection_loss

    def compute_losses(self, inputs, outputs):
        """Compute the reprojection and smoothness losses for a minibatch
        """
        losses = {}
        total_loss = 0

        for scale in self.opt.scales:
            loss = 0
            reprojection_losses = []

            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                source_scale = 0

            disp = outputs[("disp", scale)]
            color = inputs[("color", 0, scale)]
            target = inputs[("color", 0, source_scale)]

            for frame_id in self.opt.frame_ids[1:]:
                pred = outputs[("color", frame_id, scale)]
                reprojection_losses.append(self.compute_reprojection_loss(pred, target))

            reprojection_losses = torch.cat(reprojection_losses, 1)

            if not self.opt.disable_automasking:
                identity_reprojection_losses = []
                for frame_id in self.opt.frame_ids[1:]:
                    pred = inputs[("color", frame_id, source_scale)]
                    identity_reprojection_losses.append(
                        self.compute_reprojection_loss(pred, target))

                identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)

                if self.opt.avg_reprojection:
                    identity_reprojection_loss = identity_reprojection_losses.mean(1, keepdim=True)
                else:
                    # save both images, and do min all at once below
                    identity_reprojection_loss = identity_reprojection_losses

            elif self.opt.predictive_mask:
                # use the predicted mask
                mask = outputs["predictive_mask"]["disp", scale]
                if not self.opt.v1_multiscale:
                    mask = F.interpolate(
                        mask, [self.opt.height, self.opt.width],
                        mode="bilinear", align_corners=False)

                reprojection_losses *= mask

                # add a loss pushing mask to 1 (using nn.BCELoss for stability)
                weighting_loss = 0.2 * nn.BCELoss()(mask, torch.ones(mask.shape).cuda())
                loss += weighting_loss.mean()

            if self.opt.avg_reprojection:
                reprojection_loss = reprojection_losses.mean(1, keepdim=True)
            else:
                reprojection_loss = reprojection_losses

            if not self.opt.disable_automasking:
                # add random numbers to break ties
                identity_reprojection_loss += torch.randn(
                    identity_reprojection_loss.shape, device=self.device) * 0.00001

                combined = torch.cat((identity_reprojection_loss, reprojection_loss), dim=1)
            else:
                combined = reprojection_loss

            if combined.shape[1] == 1:
                to_optimise = combined
            else:
                to_optimise, idxs = torch.min(combined, dim=1)

            if not self.opt.disable_automasking:
                outputs["identity_selection/{}".format(scale)] = (
                    idxs > identity_reprojection_loss.shape[1] - 1).float()

            loss += to_optimise.mean()

            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)
            smooth_loss = get_smooth_loss(norm_disp, color)

            loss += self.opt.disparity_smoothness * smooth_loss / (2 ** scale)
            total_loss += loss
            losses["loss/{}".format(scale)] = loss

        total_loss /= self.num_scales
        losses["loss"] = total_loss
        return losses

    def compute_depth_losses(self, inputs, outputs, losses):
        """Compute depth metrics, to allow monitoring during training

        This isn't particularly accurate as it averages over the entire batch,
        so is only used to give an indication of validation performance
        """
        depth_pred = outputs[("depth", 0, 0)]
        depth_pred = torch.clamp(F.interpolate(
            depth_pred, [375, 1242], mode="bilinear", align_corners=False), 1e-3, 80)
        depth_pred = depth_pred.detach()

        depth_gt = inputs["depth_gt"].to(self.device)
        mask = depth_gt > 0

        # garg/eigen crop
        crop_mask = torch.zeros_like(mask)
        crop_mask[:, :, 153:371, 44:1197] = 1
        mask = mask * crop_mask

        depth_gt = depth_gt[mask]
        depth_pred = depth_pred[mask]
        depth_pred *= torch.median(depth_gt) / torch.median(depth_pred)

        depth_pred = torch.clamp(depth_pred, min=1e-3, max=80)

        depth_errors = compute_depth_errors(depth_gt, depth_pred)

        for i, metric in enumerate(self.depth_metric_names):
            losses[metric] = np.array(depth_errors[i].cpu())

    def log_time(self, batch_idx, duration, loss, reg_loss=None):
        """Print a logging statement to the terminal
        """
        samples_per_sec = self.opt.batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (
            self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
        if reg_loss is None:
            print_string = "epoch {:>3} | batch {:>6} | examples/s: {:5.1f}" + \
                " | loss: {:.5f} | time elapsed: {} | time left: {}"
            print(print_string.format(self.epoch, batch_idx, samples_per_sec, loss, 
                                sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)))
        else:
            print_string = "epoch {:>3} | batch {:>6} | examples/s: {:5.1f}" + \
                " | loss: {:.5f} | reg_loss: {:.5f} | time elapsed: {} | time left: {}"
            print(print_string.format(self.epoch, batch_idx, samples_per_sec, loss, reg_loss,
                                  sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)))

    def log(self, mode, inputs, outputs, losses):
        """Write an event to the tensorboard events file
        """
        writer = self.writers[mode]
        for l, v in losses.items():
            writer.add_scalar("{}".format(l), v, self.step)

        for j in range(min(4, self.opt.batch_size)):  # write a maxmimum of four images
            for s in self.opt.scales:
                for frame_id in self.opt.frame_ids:
                    writer.add_image(
                        "color_{}_{}/{}".format(frame_id, s, j),
                        inputs[("color", frame_id, s)][j].data, self.step)
                    if s == 0 and frame_id != 0:
                        writer.add_image(
                            "color_pred_{}_{}/{}".format(frame_id, s, j),
                            outputs[("color", frame_id, s)][j].data, self.step)

                writer.add_image(
                    "disp_{}/{}".format(s, j),
                    normalize_image(outputs[("disp", s)][j]), self.step)

                if self.opt.predictive_mask:
                    for f_idx, frame_id in enumerate(self.opt.frame_ids[1:]):
                        writer.add_image(
                            "predictive_mask_{}_{}/{}".format(frame_id, s, j),
                            outputs["predictive_mask"][("disp", s)][j, f_idx][None, ...],
                            self.step)

                elif not self.opt.disable_automasking:
                    writer.add_image(
                        "automask_{}/{}".format(s, j),
                        outputs["identity_selection/{}".format(s)][j][None, ...], self.step)

    def save_opts(self):
        """Save options to disk so we know what we ran this experiment with
        """
        models_dir = os.path.join(self.log_path, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        to_save = self.opt.__dict__.copy()

        with open(os.path.join(models_dir, 'opt.json'), 'w') as f:
            json.dump(to_save, f, indent=2)

    def save_model(self):
        """Save model weights to disk
        """
        save_folder = os.path.join(self.log_path, "models", "weights_{}".format(self.epoch))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        for model_name, model in self.models.items():
            save_path = os.path.join(save_folder, "{}.pth".format(model_name))
            to_save = model.state_dict()
            if model_name == 'encoder':
                # save the sizes - these are needed at prediction time
                to_save['height'] = self.opt.height
                to_save['width'] = self.opt.width
                to_save['use_stereo'] = self.opt.use_stereo
            torch.save(to_save, save_path)
        
        #blur model 파라미터 저장
        save_path = os.path.join(save_folder, "{}.pth".format("adam"))
        torch.save(self.model_optimizer.state_dict(), save_path)
        
<<<<<<< HEAD
=======
        #blur model 파라미터 저장
        save_path_blur = os.path.join(save_folder, "{}.pth".format("adam_blur"))
        torch.save(self.blur_model_optimizer.state_dict(), save_path_blur)
        
>>>>>>> 89f0717980b33cf40d74f2e6d192ee8e7f485e5c

    def load_model(self):
        """Load model(s) from disk
        """
        self.opt.load_weights_folder = os.path.expanduser(self.opt.load_weights_folder)

        assert os.path.isdir(self.opt.load_weights_folder), \
            "Cannot find folder {}".format(self.opt.load_weights_folder)
        print("loading model from folder {}".format(self.opt.load_weights_folder))

        for n in self.opt.models_to_load:
            print("Loading {} weights...".format(n))
            path = os.path.join(self.opt.load_weights_folder, "{}.pth".format(n))
            model_dict = self.models[n].state_dict()
            pretrained_dict = torch.load(path)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.models[n].load_state_dict(model_dict)

        # loading adam state
        optimizer_load_path = os.path.join(self.opt.load_weights_folder, "adam.pth")
        if os.path.isfile(optimizer_load_path):
            print("Loading Adam weights")
            optimizer_dict = torch.load(optimizer_load_path)
            self.model_optimizer.load_state_dict(optimizer_dict)
<<<<<<< HEAD
            self.model_optimizer.load_state_dict()
        else:
            print("Cannot find Adam weights so Adam is randomly initialized")
            
        
=======
            self.blur_model_optimizer.load_state_dict()
        else:
            print("Cannot find Adam weights so Adam is randomly initialized")
            
        optimizer_load_path_blur = os.path.join(self.opt.load_weights_folder, "adam_blur.pth")
        if os.path.isfile(optimizer_load_path_blur):
            print("Loading Blur Model Adam weights")
            optimizer_dict_blur = torch.load(optimizer_load_path_blur)
            self.blur_model_optimizer.load_state_dict(optimizer_dict_blur)
        else:
            print("Cannot find Blur Model Adam weights so Adam is randomly initialized")
>>>>>>> 89f0717980b33cf40d74f2e6d192ee8e7f485e5c
