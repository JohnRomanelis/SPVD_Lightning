# Efficient and Scalable Point Cloud Generation with Sparse Point-Voxel Diffusion Models


[Paper](https://arxiv.org/abs/2408.06145) | [Project Page](https://johnromanelis.github.io/_spvd/) | [Video](https://youtu.be/Ca51lMpHHms) | [Original Repo](https://github.com/JohnRomanelis/SPVD.git)

<p align="center">
  <img src="assets/SPVD.gif" width="80%"/>
</p>

This repository contains the official implementation for our publication: *"Efficient and Scalable Point Cloud Generation with Sparse Point-Voxel Diffusion Models."*

 The implementation has been restructured using [⚡ PyTorch Lightning ⚡](https://lightning.ai/docs/pytorch/stable/) for improved modularity and scalability.

# News:

- **12/8/2024**: Arxiv submission of the SPVD preprint.
- **12/9/2024**: Release of *SPVD Lightning*, utilizing PyTorch Lightning ⚡

# Installation

### 1. Set Up an Anaconda Environment

We recommend using Anaconda to manage your Python environment.

```bash
conda create --name spvd python=3.9
conda activate spvd
```

### 2. Clone the Repository (Including Submodules)
To clone this repository along with its submodules, use the --recursive flag:

```bash
git clone --recursive https://github.com/JohnRomanelis/SPVD_Lightning.git
```

### 3. Install PyTorch: 
We have tested our code with PyTorch 2.0 and CUDA 11.8. You can install the compatible version using the following command:

```bash
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia
```

### 4. Install Torchsparse
TorchSparse is now included as a submodule in the repository under the dependencies/torchsparse folder, so there is no need to clone it separately. However, you need to install system dependencies and the library itself.

1. Install Google Sparse Hash:

```bash
sudo apt-get install libsparsehash-dev
```

2. Install *torchsparse* from the submodule:

```bash
cd dependencies/torchsparse
pip install -e .
```

### 5. Install Chamfer Distance and Earth Mover Distance

- **Chamfer**:

    1. Navigate to the SPVD/metrics/chamfer_dist directory:

    ```bash
    cd SPVD/metrics/chamfer_dist
    ```

    2. Run:

    ```bash
    python setup.py install --user
    ```

- **EMD**:

    1. Navigate to the SPVD/metrics/PyTorchEMD directory:

    ```bash
    cd SPVD/metrics/PyTorchEMD
    ```

    2. Run:

    ```bash
    python setup.py install
    ```

    3. Run:

    ```bash
    cp ./build/lib.linux-x86_64-cpython-310/emd_cuda.cpython-310-x86_64-linux-gnu.so .
    ```
    
    If an error is raised in this last command, list all directories inside `build` and replace the name of the directory with the one in your system named lib.linux-x86_64-cpython-*.

