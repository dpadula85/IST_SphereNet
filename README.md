Description
===========
This repository contains electronic supporting information for the paper
**Molecular Geometry Impact on Deep Learning Predictions of Inverted
Singlet-Triplet Gaps**, submitted to _ChemRxiv_ by L. Barneschi, L. Rotondi
and D. Padula.

As described in the paper, data needed to fully reproduce are our work are
available at:

- [here for DS1](https://github.com/kjelljorner/ppp-invest/);

- [here for DS2](https://github.com/dpadula85/IST_screening)

Files provided contain:

- `jorner_data.csv`: molecules used from DS1, and their reference data;

- `make_dataset_torch_sp2.py`: script to generate an `.npz` object to be
used in conjunction with [PyTorch Dataset](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html)
to be supplied to the model;

- `splits.pt`: indices of molecules used in train, validation, and test sets;

- `train_30.pt`: weights of the SphereNet model used for the paper;

- `test_xyzs`, `xtb`, `gaff`: folders containing geometries  of the test set,
optimised at the level of theory described in the paper;

- `train.py`: script to train the network;

- `predict.py`: script to obtain predictions.

Other predictions, _e.g._ on xTB or GAFF geometries, or from DS2, can be
obtained by generating a dataset and supplying it to the model.
