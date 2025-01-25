# Repository for Point-Label Augmentation

Software tool part of my master's thesis: Leveraging foundation models to improve weakly supervised segmentation models in wildlife monitoring applications

<a name="installation"></a>
## Installation

We suggest using Anaconda package manager to install dependencies.

  1. Download Anaconda (or miniconda)
  2. Create Anaconda environment from labelex.yml environment file:

  ```conda env create -f labelex.yml ```

  3. Activate the environment:
     
  ```conda activate pointlabelspix```

  <a name="getting-started"></a>
  ## Getting Started

  ### SAM2

  clone SAM2. Place label_augmentation.py in the SAM2 cloned root directory. 

  ### Point Label Aware Superpixels

  clone Point Label Aware Superpixels. paste snn.py and spixel_utils.py in same directory as label_augmentation.py
