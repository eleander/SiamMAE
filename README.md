# SiamMAE

We have replicated the model described in [1]

Due to computation limitations, we were only able to run the MAE Base model taken from [2, 3] with a masking ratio of 50%, 75% and 95% and the Large model with a masking ratio of 75% on a subset of the Kinetics-400 Dataset [4]

## Installation Instructions

1. Download Miniconda from the [official website](https://docs.conda.io/projects/miniconda/en/latest/). Choose the appropriate version for your operating system.  
2. ```conda env create -f env.yml```
3. ```conda activate siammae```

## Project Structure

- `SiamMAE.ipynb`: This Jupyter notebook contains the main code for training and evaluation of the Siamese Masked Autoencoders (SiamMAE) model.

- `checkpoints/`: This directory contains the saved model weights at different stages of training. Each subdirectory represents a different training run.

- `dataset/`: This directory contains the data that the model is trained on.

- `dataset_loader/`: This directory contains scripts for loading and preprocessing the data. The `dataset.py` file defines the transformed Kinetics dataset, and `random_temporal_subsample.py` implements random temporal subsampling.

- `losses/`: This directory contains text files with the training loss values for different training runs.

- `test_losses/`: This directory contains text files with the test loss values for different training runs.

- `mae/`: This directory contains the code for the MAE model taken from [2,3]. The `models_mae.py` file defines the architecture of the model. 

- `ml/`: This directory contains the `SiamMAE.py` file which defines the SiamMAE model, and `utils.py` contains utility functions, such as:
  - `plot_sample(sample, model, device)`: This function takes a sample, a model, and a device as input. It plots the original sample, the model's prediction, and the mask used by the model. This function is used for visualizing the model's predictions.

  - `train_one_epoch(data_loader, optimizer, scheduler, device, model)`: This function trains the model for one epoch. It takes a data loader, an optimizer, a scheduler, a device, and a model as input. It returns a list of the loss values for each batch in the epoch.

  - `train(model, data_loader, epochs=60, prefix="", num_epochs_per_save=10, device=torch.device("cuda"))`: This function trains the model for a specified number of epochs. It takes a model, a data loader, the number of epochs, a prefix for the filenames of the saved losses and checkpoints, the number of epochs between each save of the model's state, and a device as input. It doesn't return anything.

  - `plot_results_from_checkpoint(checkpoint_name, data_loader, device, samples_path="samples.pickle")`: This function loads a model from a checkpoint, evaluates it on a data loader, and plots the results for a set of samples. It takes the name of the checkpoint file, a data loader, a device, and the path to the pickle file containing the samples as input. It doesn't return anything.


and `SiamMAE.py` contains the following classes:

  - `CrossDecoder`: This class is a PyTorch module that implements a cross-attention mechanism. It uses a multi-head attention layer to allow tokens from one sequence (f2) to attend to tokens from another sequence (f1).

  - `SiamMAE`: This class is a PyTorch module that represents a Siamese Masked Autoencoder (SiamMAE). It takes two consecutive frames from a video sequence and learns to predict the next frame. The first frame is passed through the MAE's encoder with no masking, and the next frame is passed through the MAE's encoder with a specified mask ratio. The model then attempts to predict the missing pixels in the next frame.

  - `prepare_model`: This function builds a model and loads the weights from a checkpoint. It takes the directory where the checkpoint file is located and the architecture of the model to be built as input. It returns the prepared model.

  - `download_pretrained_model`: This function downloads a pretrained model from a specified URL and saves it to a specified directory. It takes the URL where the pretrained model can be downloaded and the directory where the downloaded model should be saved as input. It doesn't return anything.

- `samples.pickle`: This file contains preprocessed samples from the Kinetics-400 Dataset [4].

# References
[1] A. Gupta et al., "Siamese Masked Autoencoders," arXiv preprint arXiv:2305.14344, 2023.  
[2] He, Kaiming, et al. "Masked autoencoders are scalable vision learners." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2022.  
[3] https://github.com/facebookresearch/mae/tree/main  
[4] [2] W. Kay et al., "The kinetics human action video dataset," arXiv preprint arXiv:1705.06950, 2017.  