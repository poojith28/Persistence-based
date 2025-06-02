# Persistence-based

## Requirements

This codebase builds upon the TypiClust project, so the software dependencies are identical (with minor modifications). For full context, see the original TypiClust repository:

- **TypiClust GitHub**: [https://github.com/avihu111/TypiClust](https://github.com/avihu111/TypiClust)

Below are the specific packages and versions required:

black==19.3b0

flake8==3.8.4

isort==4.3.21

matplotlib==3.3.4

numpy

opencv-python==4.2.0.34

torch==1.7.1

torchvision==0.8.2

parameterized

setuptools

simplejson

yacs

gudhi>=3.4

scikit-learn>=0.22

pandas>=1.0

joblib


### Setup

Clone the repository

```
   git clone https://github.com/poojith28/Persistence-based.git
   cd Persistence-based
```

Create an environment using

```
conda create --name tdaAL python=3.7
conda activate tdaAL
conda install pytorch torchvision torchaudio cudatoolkit=<CUDA_VERSION> -c pytorch
conda install matplotlib scipy scikit-learn pandas
conda install -c conda-forge faiss-gpu
pip install pyyaml easydict termcolor tqdm simplejson yacs gudhi
```

## Representation Learning
ProbCover experiments rely on representation learning. 
To train CIFAR-10 on simclr please run

```
cd scan
python simclr.py --config_env configs/env.yml --config_exp configs/pretext/simclr_cifar10.yml
cd ..
```
When this finishes, the file `./results/cifar-10/pretext/features_seed1.npy` should exist.




