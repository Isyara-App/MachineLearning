# Machine Learning Information

## Indonesian Sign Language Detection (BISINDO)
### Dataset Source
https://www.kaggle.com/datasets/agungmrf/indonesian-sign-language-bisindo

### About the model
The BISINDO Sign Language Detection is build using a CNN architecture for object recognition, in this case it is used to recognize hand gestures from images to be classified into 26 alphabet classes. The models has been trained, tested, and saved in formats compatible with deployment, including TensorFlow Lite (TFLite) for integration into Android applications.

### Approach used
##### 1. Convolutional Neural Network (CNN) : A custom model built from scratch to classify 26 BISINDO alphabet classes.
##### 2. Transfer Learning with MobileNetV2 : A pretrained model fine-tuned for BISINDO alphabet classification.

### Team Responsibilities
| Task	| Responsibilityy |
| ------------- |------------- |
| Data Preparation | Collect and preprocess datasets for image classification | 
| Model Development | Develop models using CNN and MobileNetV2 approaches | 
| Model Testing | Evaluate and compare model performance on test datasets. | 
| Integration | Save models and prepare for deployment in formats like .h5 or TFLite. | 


### Steps in Implementation
1. Dataset Preparation:
   - Downloaded dataset from Kaggle.
   - Organized data into training, validation, and test sets.
   - Removed duplicate images using hashing (dhash).
   - Visualized class distribution using bar charts.

2. Data Preparation:
   - CNN 5 Layers: Resized images to 150x150, normalized pixel values for faster convergence during training.
   - Transfer Learning: Resized images to 224x224, normalization is done within the model.
     
3. Modeling Approaches:
   - CNN 5 Layers: Built from scratch with five convolutional layers, max-pooling, and dense layers for classification.
   - Transfer Learning: Used MobileNetV2 pretrained on ImageNet with added custom layers of preprocessing inputs for classification.
     
4. Training:
   - CNN : trained for 50 epochs.
   - MobileNetV2 : trained for 25 epochs.
     
5. Evaluation:
   - Measured accuracy and loss on the test dataset for both models.
     
6. Model Saving:
   - Saved models in H5 and TFLite formats for further use and deployment.

### Results and Comparison
| Model	| Training Accuracy |	Validation Accuracy |	Test Accuracy |
| ------------- |------------- | ------------- | ------------- |
| CNN 5 Layers	| 97% | 94% |	94% |
| Transfer Learning	| 99% |	95% |	96% |

### Installation Guide 
#### <ins>Prerequisites</ins>
- Python 3.8 or higher
- TensorFlow 2.x
- Required Python packages (specified in requirements.txt files)
- Google Colab or local Python environment
#### <ins>Set Up</ins>
1. Clone the repository
   ```
   git clone https://github.com/Isyara-App/MachineLearning
   ```
2. Install Dependencies
   - **CNN**:
      ```
      pip install -r MachineLearning/BuildAndTrainModel/CNN/requirements.txt
      ```
   - **Transfer Learning**:
      ```
      pip install -r MachineLearning/BuildAndTrainModel/TransferLearning/requirements.txt
      ```
3. Use Models
   
   The pretrained models are saved under:
   - **CNN**: `BuildAndTrainModel/CNN/models/`
   - **Transfer Learning**: `BuildAndTrainModel/TransferLearning/savedModels/`
   
   The models format saved as: `.h5`, `.keras`, `.tflite`
4. Load Models
   - **For .h5 models**:
     
     - **CNN**:
       ```
       from keras.models import load_model
       
       model = load_model('/content/MachineLearning/BuildAndTrainModel/CNN/models/trained_model.h5') # Replace the model path
       ```
     - **Transfer Learning**:
       ```
       from keras.models import load_model
       from custom_layers import PreprocessingLayer

       model = load_model('/content/MachineLearning/BuildAndTrainModel/TransferLearning/savedModels/handgesture_model.h5', custom_objects={'PreprocessingLayer': PreprocessingLayer}) # Replace the model path
       ```
   - **For .tflite models**:
        
        - **CNN**:
          ```
          import tensorflow as tf

          # Replace the model path
          interpreter = tf.lite.Interpreter(model_path="/content/MachineLearning/BuildAndTrainModel/CNN/models/trained_model.tflite")
          interpreter.allocate_tensors()
          
          input_details = interpreter.get_input_details()
          output_details = interpreter.get_output_details()
          print("Inputs:", input_details)
          print("Outputs:", output_details)
          ```
       - **Transfer Learning**:
          ```
          import tensorflow as tf

          # Replace the model path
          interpreter = tf.lite.Interpreter(model_path="/content/MachineLearning/BuildAndTrainModel/TransferLearning/savedModels/model.tflite")
          interpreter.allocate_tensors()
          
          input_details = interpreter.get_input_details()
          output_details = interpreter.get_output_details()
          print("Inputs:", input_details)
          print("Outputs:", output_details)
          ```
5. TBC
   
   

### Members
| Name | ID Bangkit | University |
| ------------- |------------- | ------------- |
| Aldi Musneldi  | M497B4KY0328	  | Universitas Putra Indonesia Yptk Padang  |
| Cathleen Davina Hendrawan  | M233B4KX0906  | Universitas Katolik Soegijapranata  |
| Muhammad Rafi Abhinaya  | M006B4KY2989  |Universitas Brawijaya  |


