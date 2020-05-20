# Distributed DNN Deployment

A simulator for deploying a single neural network model into the distributed environment.


## Instructions

1. Run `train_and_save.py` to train a full model and save it in HDF5 format.
2. Run `run_split.py` to generated split models.
3. Run `final_inference.py` to wait server to generate final predictions.
4. Run `trigger_inference.py` to start prediction by providing the input.
