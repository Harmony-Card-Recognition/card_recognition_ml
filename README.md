## NOT A FINISHED PROJECT


# Packages to Install
### NOTE: first activate the venv so that you are installing the packages there and not the host machine 
    DEPRECIATED: pip install scikit-learn opencv-python keras numpy pandas Pillow requests tensorflow matplotlib tensorflow_probability tensorrt tensorflow-probability[tf]
    pip install tensorflow[and-cuda] numpy pandas requests Pillow scikit-learn 


# Virtual Environment 
## Making a Virtual Environment (venv) for python 
### if the .venv is already made, then you don't have to do this again
    python3 -m venv .venv
## Activation
### you must do this before installing or running anything
### you must also be in the scope of the directory with ./.venv
    # windows
    .venv\Scripts\activate
    # unix
    source .venv/bin/activate
## Deactivation
### you must have first activated the .venv
    deactivate

# Testing A Folder of Cards

# GPU usage
https://www.tensorflow.org/install/pip#linux_setup



# other stupid tensorflow and NVIDIA caca and weewee
sudo echo 0 | sudo tee -a /sys/bus/pci/devices/0000:01:00.0/numa_node

# CUDA cuDNN and TensorRT
https://www.youtube.com/watch?v=1Tr1ifuSh6o
## cuDNN
https://developer.nvidia.com/rdp/cudnn-archive
## CUDA 
https://developer.nvidia.com/cuda-12-1-1-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=runfile_local
## TensorRT
https://developer.nvidia.com/nvidia-tensorrt-8x-download
    




# anaCONDA environments
https://medium.com/@dev-charodeyka/tensorflow-conda-nvidia-gpu-on-ubuntu-22-04-3-lts-ad61c1d9ee32

conda activate tf-env-ultimate