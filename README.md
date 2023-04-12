# biomachina
Code to participate in plantclef2022

## Links

- Competition url: https://www.imageclef.org/PlantCLEF2022
- Competition dataset url: https://lab.plantnet.org/LifeCLEF/PlantCLEF2022
- tensorboard logs: tensorboard --logdir=logs/

## Folder structure

checkpoints: weights of the models are stored in this folder.
logs: tensorboard logging folder.
output: Hydra's output folders for each experiment's logs.

## Linux mount
sudo fdisk -l
sudo mount /dev/nvme0n1p3 /mnt/media/

## Running code in Kabre cluster.

you  can run using the script called `run.sh` it also required the file `load_env.sh`
to load conda environment in the node.
This script does:
1. extract tar files in the node memory.
2. create an environment called `biomachina` with pytorch-lts 1.8 and all dependencies to run this code.
3. run the code.

you can override the following parameters:
- `ds_dir` : expect a folder with 2 tar files: web.tar and trusted.tar
        Default /work/$USER/dataset/plantclef-2022/
- `src_dir` : source code
        Default /home/$USER/plantclef2022/src/
- `run_tmp` : memory mount point to extract tar files.
        Default /dev/shm/plantclef-2022/
- `conda_env` : name of conda environment to install pytorch and dependencies.
        Default biomachina
- `minutes` : for how long this task can run, remember there are limitations on the partition.
        Default 60
- `partition` : 
        Default nukwa-debug