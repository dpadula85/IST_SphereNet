#!/usr/bin/env python

import os
import sys
import torch
import numpy as np
from datasets import Geoms
from dig.threedgraph.method import SphereNet
from dig.threedgraph.evaluation import ThreeDEvaluator
from dig.threedgraph.method import run

if __name__ == '__main__':

    dataset = Geoms()
    split_idx = torch.load("splits.pt")
    train_dataset, valid_dataset, test_dataset = dataset[split_idx['train']], dataset[split_idx['valid']], dataset[split_idx['test']]

    # Define model, loss, and evaluation
    model = SphereNet(energy_and_force=False, cutoff=5.0, num_layers=4,
                      hidden_channels=128, out_channels=1, int_emb_size=64,
                      basis_emb_size_dist=8, basis_emb_size_angle=8, basis_emb_size_torsion=8,
                      out_emb_channels=256, num_spherical=3, num_radial=6, envelope_exponent=5,
                      num_before_skip=1, num_after_skip=2, num_output_layers=3)                 

    loss_func = torch.nn.L1Loss()
    evaluation = ThreeDEvaluator()

    gpu = torch.device('cuda:0')

    # Train and evaluate
    run3d = run()
    run3d.run(gpu, train_dataset, valid_dataset, test_dataset, model, loss_func, evaluation,
              epochs=30, batch_size=16, vt_batch_size=16,
              lr=0.0005, lr_decay_factor=0.5, lr_decay_step_size=15,
              save_dir=f"train_log", log_dir=f"train_log")
