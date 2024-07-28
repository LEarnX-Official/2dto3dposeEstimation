# 2dto3dposeEstimation
GraFormer Training Script 
This repository contains a PyTorch implementation for training the GraFormer model on the Human3.6M dataset for 3D human pose estimation. The
script allows for training, evaluation, and resumption of training from checkpoints. 
Prerequisites 
Before you begin, ensure you have met the following requirements: - Python 3.7 or higher - PyTorch 1.7.0 or higher - CUDA 10.1 or higher (for GPU
acceleration) 
Installation 
Clone the repository: sh git clone https://github.com/LEarnX-Official/2dto3dposeEstimation.git cd
2dto3dposeEstimation 
Install the required packages: sh pip install -r requirements.txt 
Usage 
Arguments 
The script accepts several command-line arguments: 
-d, --dataset: Target dataset (default: ‘h36m’)
-k, --keypoints: 2D detections to use (default: ‘gt’)
-a, --actions: Actions to train/test on, separated by comma, or * for all (default: ‘*’)
--evaluate: Checkpoint to evaluate (file name)
-r, --resume: Checkpoint to resume (file name)
-c, --checkpoint: Checkpoint directory (default: ‘checkpoint’)
--snapshot: Save models for every #snapshot epochs (default: 5)
--n_head: Number of attention heads (default: 4)
--dim_model: Dimension of the model (default: 96)
--n_layer: Number of layers (default: 5)
--dropout: Dropout rate (default: 0.25)
-b, --batch_size: Batch size in terms of predicted frames (default: 64)
-e, --epochs: Number of training epochs (default: 10)
--num_workers: Number of workers for data loading (default: 1)
--lr: Initial learning rate (default: 1.0e-3)
--lr_decay: Number of steps of learning rate decay (default: 50000)
--lr_gamma: Gamma of learning rate decay (default: 0.9)
--downsample: Downsample frame rate by factor (default: 1)
--max_norm: Max norm for gradient clipping (default: 1) 
Training 
To train the model, run the following command: sh python main_GraFormer.py -d h36m -k gt -a * -c checkpoint --snapshot 5 -b
64 -e 10 --lr 1.0e-3 --max_norm 1 
Evaluation 
To evaluate the model using a specific checkpoint, run the following command: sh python main_GraFormer.py --evaluate checkpoint
best_model.pth 
Resuming Training 
To resume training from a specific checkpoint, run the following command: sh python main_GraFormer.py --resume checkpoint
latest_model.pth 
Example Command 
Here is an example command to train the model on the Human3.6M dataset using ground truth keypoints: sh python main_GraFormer.py -
dataset h36m --keypoints gt --actions * --checkpoint checkpoint --snapshot 5 --batch_size 64 --epochs 10 --lr
1.0e-3 --max_norm 1 
Additional Information 
Project Structurecommon/: Contains utility scripts for logging, data processing, and loss calculation.
network/: Contains the implementation of the GraFormer model and related components.
data/: Directory where the dataset files are expected to be located. 
Model 
GraFormer is a graph-based transformer model designed for 3D human pose estimation. It utilizes graph convolutional layers and transformer layers to
capture both local and global dependencies in human pose data. 
Dataset 
This script is configured to work with the Human3.6M dataset. Ensure you have the dataset prepared and placed in the data/ directory. 
Checkpoints 
Model checkpoints are saved periodically during training. The best model and the latest model are saved in the specified checkpoint directory. 
Logging 
Training and evaluation logs are saved in the checkpoint directory and can be reviewed to monitor progress and performance. 
