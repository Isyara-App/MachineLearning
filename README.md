# Machine Learning Information

# Indonesian Sign Language Detection (Bisindo)

## Dataset Source
https://www.kaggle.com/datasets/agungmrf/indonesian-sign-language-bisindo

## About the model

## Approach used
#### 1. Convolutional Neural Network (CNN) : A custom model built from scratch to classify 26 BISINDO alphabet classes.
#### 2. Transfer Learning with MobileNetV2 : A pretrained model fine-tuned for BISINDO alphabet classification.

## Team Responsibilities
| Task	| Responsibilityy |
| ------------- |------------- |
| Data Preparation | Collect and preprocess datasets for image classification | 
| Model Development | Develop models using CNN and MobileNetV2 approaches | 
| Model Testing | Evaluate and compare model performance on test datasets. | 
| Integration | Save models and prepare for deployment in formats like .h5 or TFLite. | 


## Steps in Implementation
1. Dataset Preparation:
   - Downloaded dataset from Kaggle.
   - Organized data into training, validation, and test sets.
   - Removed duplicate images using hashing (dhash).
   - Visualized class distribution using bar charts.

2. Data Preparation:
   - Resized images to 224x224.
   - Normalized pixel values for faster convergence during training.
     
3. Modeling Approaches:
   - CNN 5 Layers: Built from scratch with five convolutional layers, max-pooling, and dense layers for classification.
   - Transfer Learning: Used MobileNetV2 pretrained on ImageNet with added custom layers for classification.
     
4. Training:
   - CNN : trained for 50 epochs.
   - MobileNetV2 : trained for 25 epochs.
     
5. Evaluation:
   - Measured accuracy and loss on the test dataset for both models.
     
6. Model Saving:
   - Saved models in H5 and TFLite formats for further use and deployment.

## Results and Comparison
| Model	| Training Accuracy |	Validation Accuracy |	Test Accuracy |
| ------------- |------------- | ------------- | ------------- |
| CNN 5 Layers	| 97% | 94% |	94% |
| Transfer Learning	| 99% |	95% |	96% |

## Installation Guide
#### Image Recognition

## Members
| Name | ID Bangkit | University |
| ------------- |------------- | ------------- |
| Aldi Musneldi  | M497B4KY0328	  | Universitas Putra Indonesia Yptk Padang  |
| Cathleen Davina Hendrawan  | M233B4KX0906  | Universitas Katolik Soegijapranata  |
| Muhammad Rafi Abhinaya  | M006B4KY2989  |Universitas Brawijaya  |


