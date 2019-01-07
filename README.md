# Automated highway condition assessment
![POWERED BY TeCSAR](https://raw.githubusercontent.com/TeCSAR-UNCC/Deep_RACE/master/logo/tecsarPowerBy.png)

The focus of this research is on designing deep learning based tools for automated highway condition assessment. At this stage, we primarily focus on developing holistic algorithmic constructs for fully autonomous detection and labeling of road asset items. To this end, we have developed a CNN-based deep classifier based on VGG-Net at front-end of our proposed framework. At the same time, the deep classifier takes the benefits of transfer learning to overcome the challenge of limited data with high sparsity, and binary networks for finetuning of assets classification at the back-end of the framework.
## Prerequisites

In addition to python 3.5, you also need to install the follwoing python packages:
```bash
Tensorflow, pandas, SciPy, scikit-learn, seaborn, Six, PIL, Image, matplotlib, opencv-python
```

## Data preparation

Resize all the images into a unique size by using `imresize.py`. Make sure to have all images under a folder called `imgs` in the same directory.

## Training and testing the model
Save the data in a folder named as `dataset` in two sepearate subfolders as following in a categorical fashion (One folder for each class):
```bash
train_dir = './dataset/train'
valid_dir = './dataset/validation'
```
Apply the required changes based on the size of input images, number of images, etc. Then run `train.py`.

Note: `alex-train.py` is a code that we wrote to try using AlexNet for this project before we started using transfer learning on VGGNet.

### Misclassification analysis
Once training and test is done, use the script `misclass-analysis.py` for misclassification analysis. Save the misclassification raw data extracted from `train.py` in a CSV file named `classifier_results.csv` as the input of this script. Run the code in `Jupyter Notebook` to see the flow of result generation.

## Author
 Sadegh Nouri Gooshki - [snourigo@uncc.edu](snourigo@uncc.edu)

## License
Copyright (c) 2018, University of North Carolina at Charlotte All rights reserved. - see the [LICENSE](?)
