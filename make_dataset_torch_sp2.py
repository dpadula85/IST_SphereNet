#!/usr/bin/env python

import os
import sys
import numpy as np
import pandas as pd
import MDAnalysis as mda

anum = {'H' : 1, 'C' : 6, 'N' : 7, 'O' : 8, 'F' : 9,
        'Si' : 14, 'SI': 14, 'P' : 15, 'S' : 16 ,'NA' : 11, 'CL' : 17}


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


def make_fname(data, csvd):

    geom_str_fmt = "%s.xyz"
    geom_fname = geom_str_fmt % tuple(data.values.tolist())
    fname = os.path.join(csvd, geom_fname)

    return fname


def get_geom(fname):

    u = mda.Universe(fname, guess_bonds=True)
    Z = np.asarray(list(map(anum.get, u.atoms.types)))
    sp2idxs = get_sp2(u)
    Z = Z[sp2idxs]
    X = u.atoms.positions[sp2idxs]
    idxs = np.where(Z != 1)[0]
    Z = Z[idxs]
    X = X[idxs]

    return Z, X


if __name__ == '__main__':

    dfs = []
    csv = os.path.join(os.getcwd(), "jorner_data.csv")
    csvd = os.path.dirname(csv)
    df = pd.read_csv(csv)
    fn = lambda x: make_fname(x, csvd)
    df["File"] = df[["id"]].apply(fn, axis=1)
    files = df["File"]
    data = files.apply(get_geom)
    Z, R = zip(*data.values)
    N = np.asarray([ len(i) for i in Z ])
    Z = np.concatenate(Z)
    R = np.concatenate(R)
    Y = df["t1_s1_ref"]
    mol = df["id"]
    files = files.values

    np.savez_compressed(
            "data_torch_sp2.npz",
            N=N,
            Z=Z,
            R=R,
            Y=Y,
            mol=mol,
            files=files
        )
