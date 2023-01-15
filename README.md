# CRC-Aided Learned Ensembles of Belief-Propagation Polar Decoders

Python repository for the paper "CRC-Aided Learned Ensembles of Belief-Propagation Polar Decoders".

Please cite our [paper](https://arxiv.org/), if the code is used for publishing research.

# Table of Contents

- [Introduction](#introduction)
- [Folders Structure](#folders-structure)
  * [python_code](#python_code)
    + [config](#config)
    + [codes](#codes)
    + [data_](#data_)
    + [decoders](#decoders)
    + [plotter](#plotter)
    + [trainers](#trainers)
    + [utils](#utils)
  * [data](#data)
  * [results](#results)
  * [globals](#globals)
  * [dir_definitions](#dir_definitions)
- [Execution](#execution)
  * [Environment Installation](#environment-installation)

<small><i><a href='http://ecotrust-canada.github.io/markdown-toc/'>Table of contents generated with markdown-toc</a></i></small>

# Introduction

This repository implements the proposed crc-aided polar ensemble framework.

# Folders Structure

## python_code 

The python simulations of the simplified communication chain: message generation, encoding, channel transmission and decoding via neural based decoding methods.

### config

The config yaml that controls the training/evaluation options:

Decoder params - code_type, code_len, info_len, clipping_val, iteration_num, early_stopping, design_SNR, ensemble_dec_num, ensemble_crc_dist.

Data params - val_batch_size, val_SNR_start, val_SNR_end, val_num_SNR, noise_seed, word_seed, test_errors, crc_order.

NN training hyperparams - run_name, load_weights, lr, optimizer_type, criterion_type, num_of_epochs, validation_epochs, train_minibatch_size, train_SNR_start, train_SNR_end, train_num_SNR.

### codes 

Holds all relevant functions for the encoding/decoding of the polar and CRC codes.

### data_ 

Responsible for creation of the dataset composed of pairs of transmitted codewords, and the respective channel outputs.

### decoders

Includes the WFG decoder and the proposed ensemble decoder.

### plotter

Plotting of the BER versus SNR / FLOPS versus SNR for the paper.

### trainers

The trainers wrapping the respective decoders, WFG trainer and the ensemble trainer. Trains the respective decoders on simulated datasets, and evaluates performance.

### utils

Additional misc functions such as calculation of the FER / FLOPS metrics, and saving/loading of pkl files.

## data

Saves the pt data pairs of transmitted / received pairs for training of the ensemble without the need to re-calculate the CRC remainder each time anew.

## results 

Saves the weights of training under 'weights' subdirectory, the evaluation results under 'plots' subdirectory, and the plotted figures (from plotter) under 'plots' subdirectory. 

## globals 

Cuda/cpu device, as well as the config (following Singleton design pattern) used throughout the code.

## dir_definitions 

Definitions of relative directories.

# Execution

To execute the code, first download and install Git, Anaconda and PyCharm.

Then install the environment, follow the installation setup below. 

At last, open PyCharm in the root directory. You may run either the trainers or one of the plotters.

This code was simulated with GeForce RTX 3060 with CUDA 12. 

## Environment Installation

1. Open git bash and cd to a working directory of you choice.

2. Clone this repository to your local machine.

3. Open Anaconda prompt and navigate to the cloned repository.

4. Create a new environment.

5. Run 'pip install -r requirements.txt'. This builds the appropriate software dependencies.

6. Done!
