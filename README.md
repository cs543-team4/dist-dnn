# Distributed DNN Deployment

A simulator for deploying a single neural network model into the distributed environment.


## Instructions

1. Run `python train_and_save.py` to train a full model and save it in HDF5 format.
2. Run `python run_split.py` to generated split models.
3. Run `python inference.py --model_index 1` to run inference server.
4. Run `python inference.py --device` to trigger an inference chain from a device.
