# Rock Ptarmigan Age and Sex Classification with Machine Learning

## Overview

This project explores the use of **machine learning and image classification** to predict the **age and sex of Rock Ptarmigan birds from wing images**.

The goal is to investigate whether subtle visual characteristics in wing images can be learned by a neural network to assist researchers with faster wildlife classification.

## Technologies

* Python
* NumPy
* Pandas
* Matplotlib
* PyTorch
* Neural Networks
* Multilayer Perceptron (MLP)
* Softmax Classification
* ExifTool
* Regular Expressions

## Data Processing

The project includes a preprocessing pipeline for preparing wing images and their associated metadata.

The pipeline:

* Standardizes image filenames
* Extracts metadata using ExifTool
* Uses regex to clean inconsistent metadata
* Extracts age and sex labels
* Creates structured training data for the classifiers

## Machine Learning Approach

We initially experimented with a rule-based color thresholding approach before moving to neural networks.

Two MLP implementations were explored:

* A manually implemented MLP
* A PyTorch-based MLP

Initially, a single model attempted to predict **age and sex simultaneously**, but this achieved only **29.23% validation accuracy**.

The problem was therefore separated into two independent classification models:

* **Age Classification:** Adult vs Juvenile
* **Sex Classification:** Male vs Female

Different hidden-layer sizes of **20, 50, and 100 nodes** were evaluated along with different batch sizes and training configurations.

## Results

| Model                      | Best Accuracy |
| -------------------------- | ------------: |
| Joint Age + Sex Classifier |        29.23% |
| Age Classifier             |        61.54% |
| Sex Classifier             |        76.15% |

The best age classifier used **20 hidden nodes**, while the best sex classifier used **50 hidden nodes**, both with a learning rate of **0.01**.

Training experiments also showed signs of overfitting after approximately **42 epochs**, which was used as the final stopping point.

## Key Takeaway

Separating age and sex prediction significantly improved classification performance compared with predicting both simultaneously.

The project also demonstrated the limitations of MLPs for image-based classification. More advanced architectures such as **Convolutional Neural Networks (CNNs)** could potentially learn detailed feather patterns and spatial features more effectively.

## Authors

**Jay Mhaiskar**

* Developed and implemented the classification algorithms
* Worked on neural network architecture and model design
* Integrated the model with the training dataset
* Analyzed classifier performance and results

**Thuong Nguyen**

* Project research and reporting
* Testing and result documentation

**Khoa Bui**

* Code and results documentation
* Organized project data and presentation materials
