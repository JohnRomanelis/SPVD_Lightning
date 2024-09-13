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

### 1. Clone the Repository (Including Submodules)
To clone this repository along with its submodules, use the --recursive flag:

```bash
git clone --recursive https://github.com/JohnRomanelis/SPVD_Lightning.git
```

### 2. Set Up an Anaconda Environment 

We recommend using Anaconda to manage your Python environment.
We also provide an `environment.yml` to set it up. 

By running the following command you will create a conda env called *spvd* containing most of the libraries required to run our code. 
```bash
conda env create -f environment.yml
```

### 3. Install Torchsparse
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

### 4. Install Chamfer Distance and Earth Mover Distance

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

# Data

For generation, we use the same version of ShapeNet as [PointFlow](https://github.com/stevenygd/PointFlow.git). Please refer to their instructions for downloading the dataset.

# Citation

If you find this work useful in your research, please consider citing:

```bibtex
@misc{romanelis2024efficientscalablepointcloud,
      title={Efficient and Scalable Point Cloud Generation with Sparse Point-Voxel Diffusion Models}, 
      author={Ioannis Romanelis and Vlassios Fotis and Athanasios Kalogeras and Christos Alexakos and Konstantinos Moustakas and Adrian Munteanu},
      year={2024},
      eprint={2408.06145},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2408.06145}, 
}