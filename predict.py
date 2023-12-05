#!/usr/bin/env python

import os
import csv
import sys
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from dig.threedgraph.method import SphereNet
from dig.threedgraph.evaluation import ThreeDEvaluator
from dig.threedgraph.method import run

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

import MDAnalysis as mda

from datasets import *


anum = {'H' : 1, 'C' : 6, 'N' : 7, 'O' : 8, 'F' : 9,
        'Si' : 14, 'SI' : 14, 'P' : 15, 'S' : 16 ,'NA' : 11, 'CL' : 17}


def flatten(lst):
    '''
    Recursive function to flatten a nested list.

    Parameters
    ----------
    lst: list.
        Nested list to be flattened.

    Returns
    -------                                             
    flattened: list.
        Flattened list.
    '''

    flattened = sum( ([x] if not isinstance(x, list)
                     else flatten(x) for x in lst), [] )

    return flattened 


def get_sp2(u, alkyl=True, ether=False):
    '''
    Function to crop alkyl side chains in a universe, optionally writing
    a cropped trajectory.

    Parameters
    ----------
    u: object.
        MDAnalysis Universe to be cropped.
    alkyl: bool.
        Whether side chains to crop are purely alkylic.
    ether: bool.
        Whether side chains to crop have an oxygen atom connected to the


    Returns
    -------
    n: object.
        MDAnalysis Universe with cropped side chains.
    '''

    # Get bonds
    try:
        bds = u.bonds.to_indices()
    except:
        u.guess_bonds()
        bds = u.bonds.to_indices()

    # Get connectivity, use -1 as a placeholder for empty valence
    conn = np.ones((len(u.atoms), 4)) * -1
    for bond in bds:
        at1, at2 = bond
        for j in np.arange(conn[at1].shape[0]):
            if conn[at1,j] == -1:
                conn[at1,j] = at2
                break

        for j in np.arange(conn[at2].shape[0]):
            if conn[at2,j] == -1:
                conn[at2,j] = at1
                break

    # Get heavy atoms
    heavy = np.where(u.atoms.types != "H")[0]

    # Get sp3 atoms
    sat = np.where(np.all(conn > -1, axis=1))[0]

    # Alkyls or ether chain
    if alkyl:
        allcheck = sat
    elif ether:
        allcheck = np.concatenate([sat, oxy])
    else:
        allcheck = sat

    # Check all sp3 atoms
    keep = []
    delete = []
    for satat in allcheck:

        # check connectivity
        iconn = conn[satat]

        # filter H out from connected
        iconnheavy = iconn[np.in1d(iconn, heavy)]

        delete.append(satat)
        delete.extend(iconn[~np.in1d(iconn, heavy)])

    # Convert to int arrays
    keep = np.asarray(keep).astype(int)
    delete = np.asarray(delete).astype(int)

    # Get non sp3 atoms
    unsat = ~np.all(conn > -1, axis=1)

    # Set which saturated atoms to keep or delete
    unsat[keep] = True
    unsat[delete] = False
    tokeep = np.where(unsat)[0]

    return tokeep


def predict(batch, checkpoint):

    model = SphereNet(energy_and_force=True, cutoff=5.0, num_layers=4,
                      hidden_channels=128, out_channels=1, int_emb_size=64,
                      basis_emb_size_dist=8, basis_emb_size_angle=8, basis_emb_size_torsion=8,
                      out_emb_channels=256, num_spherical=3, num_radial=6, envelope_exponent=5,
                      num_before_skip=1, num_after_skip=2, num_output_layers=3)

    model.load_state_dict(
            torch.load(checkpoint, map_location=torch.device('cuda:0'))["model_state_dict"]
        )
    model.eval()

    results = []
    grads = []
    for data in batch:
        prediction = model(data)
        results.append(prediction.detach().numpy())

    return results, grads


def predict_from_file(filename, checkpoint):
    """
    args: 
    
    filename (str): path to xyz file
    checkpoint (str): path to checkpoint file

    Description:

    Makes prediction from XYZ file with dimer coordinates
    """

    u = mda.Universe(filename, guess_bonds=True)
    Z = np.asarray(list(map(anum.get, u.atoms.types)))
    sp2idxs = get_sp2(u)
    Z = Z[sp2idxs]
    X = u.atoms.positions[sp2idxs]
    idxs = np.where(Z != 1)[0]
    R_i = torch.tensor(X[idxs], dtype=torch.float32)
    z_i = torch.tensor(Z[idxs], dtype=torch.int64)
    data = Data(pos=R_i, z=z_i,)
    batch = DataLoader([data], batch_size=1)

    return predict(batch,checkpoint=checkpoint)


def pred_data(model, dataset):

    # Get predictions
    batch = DataLoader(dataset, batch_size=8)

    yhat = []
    mols = []
    for data in tqdm(batch):
        mols.append(data.mol)
        data.to(gpu)
        try:
            pred = model(data)
            yhat.append(pred.cpu().detach().numpy())
        except:
            yhat.append(np.zeros((8, 1)))

    yhat = np.concatenate(yhat).reshape(-1)

    df = pd.DataFrame({
                "name" : flatten(mols),
                "ST_ML  / meV" : yhat * 1000,
           })

    return df


if __name__ == '__main__':

    dataset = Geoms()

    split_idx = torch.load("splits.pt")
    train_dataset, valid_dataset, test_dataset = dataset[split_idx['train']], dataset[split_idx['valid']], dataset[split_idx['test']]

    chk = f"train_30.pt"
    gpu = torch.device('cuda:0')

    # Define model
    model = SphereNet(energy_and_force=False, cutoff=5.0, num_layers=4,
                      hidden_channels=128, out_channels=1, int_emb_size=64,
                      basis_emb_size_dist=8, basis_emb_size_angle=8, basis_emb_size_torsion=8,
                      out_emb_channels=256, num_spherical=3, num_radial=6, envelope_exponent=5,
                      num_before_skip=1, num_after_skip=2, num_output_layers=3)                 

    model.load_state_dict(
            torch.load(chk, map_location=gpu)["model_state_dict"]
        )
    model.to(gpu)

    df = pred_data(model, test_dataset) 
    df.to_csv(f"predictions.csv", index=False, quoting=csv.QUOTE_NONNUMERIC)
