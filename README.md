# Efficient Neural Architectures, Workload Mappings, and Hardware Layouts for Hyperspectral Imaging
Hyperspectral images (HSI) are becoming increasingly important in various sectors, including remote sensing, medical imaging, and material research. These images contain tens to hundreds of channels, making them high-dimensional and challenging to analyze. In this project, we investigate the energy-accuracy tradeoff of neural network architectures designed for HSI segmentation using 3D convolutions and analyze the effect of architectural configuration on MAC, the number of cycles, and parameters. We compare the results to the neural network using 2D convolutions and conclude that 3D convolution has a better energy-accuracy trade-off, achieving higher accuracy while using less energy. We then investigate hardware designs based on 1D and 2D PE ($1\times16$, $2\times8$, $4\times4$) for performing 3D convolutions with variable filter sizes (3, 5, 7, 9). Our findings have important implications for the development of an efficient neural network for HSI segmentation in various applications, providing an understanding of the efficiency of 3D convolutions compared to 2D convolutions and which hardware design parameters allow for the efficient execution of 3D convolutions. Our findings have significant implications for developing more effective HSI segmentation in a variety of applications.

Authors: Madeline Loui Anderson, Hyewon Jeong, Joanna Kondylis, Ferdi Kossmann

Please refer to the project proposal and final documentation in the repository: [Proposal](reports/proposal.pdf), [Final report](reports/report.pdf)

## Requirements
Our repository contains two part of code base which requires differernt environment to run experiments. Training and testing the 3D/2D Convolutional Neural Network with PyTorch code, you can set up environment with Python 3.9.16 and [PyTorch](http://pytorch.org/) 1.12.1. Code dependency can be found in: [requirements.txt](requirements.txt)

Run code below to install requirements.
```
pip install -r requirements.txt
```

## Use Docker for Evaluating Hardwares
To use docker 
Please pull the docker first to update the container, and then start with `docker-compose up`. 
```
cd <your-git-repo-for-lab3>
export DOCKER_ARCH=<amd64 or arm64>
docker-compose pull
docker-compose up
```

## Hyperspectral dataset: Indian Pines Dataset
We use the benchmark Hyperspectral Dataset (HSI), [AVIRIS Indian Pines Dataset](https://www.ehu.eus/ccwintco/index.php?title=Hyperspectral_Remote_Sensing_Scenes#Indian_Pines). Indian Pines dataset is acquired by the Airborne Visible/Infrared Imaging Spectrometer (AVIRIS) sensor over the Indian Pines test site in North-western Indiana in June 1992. The default dataset folder to save the dataset `./Datasets/`, although this can be modified at runtime using the `--folder` arguments.

## Usage
To train 3D Convolutional network with IndianPines dataset with GPU, one can run:
```
python main.py --model hamida --dataset IndianPines --training_sample 0.8 --cuda 1 --batch_size 16
```

To train 2D Convolutional network with IndianPines dataset
```
python main.py --model hamida2d --dataset IndianPines --training_sample 0.8 --cuda 1 --batch_size 16
```

We used the batch size of 16 and 80% of total Indian Pines dataset for segmentation, you can change the training sample size and batch size accordingly.

## References
The code for running model is heavily dependent on the PyTorch code from [DeepHyperX](https://github.com/nshaud/DeepHyperX) and the code for evaluating hardware is adopted from the code from Lab3 of 6.5930/1 Hardware Architecture for Deep Learning - Spring 2023.