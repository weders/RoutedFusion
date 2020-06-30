# RoutedFusion: Learning Real-time Depth Map Fusion

This is the official and improved implementation of the CVPR 2020 submission "RoutedFusion: Real-time Depth Map Fusion". 

RoutedFusion is a real-time capable depth map fusion method that leverages machine learning for fusing noisy and outlier-contaminated depth maps. It consists of two neural networks components: 1) the depth routing network that performs a 2D prepocessing of the depth maps estimating a de-noised depth map as well as corresponding confidence map. 2) a depth fusion network that predicts optimal updates to the scene representation given a canonical view of the current state of the scene representation as well as the new measurement and confidence map.

## Usage

Below you find instructions on how to use RoutedFusion as a standalone depth map fusion pipeline for training and evaluation.

### Prerequisites
For using RoutedFusion, you need to have the following installed on you machine:

1. docker
2. nvidia-docker

### Installation

There are two possible ways of installing RoutedFusion. The recommended way is to use Docker. You also can use a conda environmnet but this comes without any guarantees.

**Clone the repo**

<pre><code>git clone https://github.com/weders/RoutedFusion.git
</code></pre>

**Build the docker image**
<pre><code>docker build . -t routed-fusion
</code></pre>

**Start and enter the container from the image**
<pre><code>docker run -it routed-fusion:latest /bin/bash
</code></pre>

**Alternatively, create the Anaconda environment**
<pre><code>conda create env -f environment.yml
</code></pre>

### Data Preparation
The models are trained on the ShapeNet v1 dataset. Therefore, the data needs to be downloaded and perpared using mesh-fusion.

**Prerequisites**
1. Download ShapeNet v1 from www.shapenet.org and unzip files
2. Download binvox from https://www.patrickmin.com/binvox/, adjust permissions and move it to /usr/bin/

**Install Data Generation**

Run the installation script *scripts/install_data_generation.sh*

**Generate Shapenet Data**

Run the data generation script *scripts/generate_shapenet_data.sh*
<pre><code>bash scripts/generate_shapenet_data.sh $PATH_TO_SHAPENET $PATH_TO_GENERATE_TO
</code></pre>

