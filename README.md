# Machine Learning Information

## Indonesian Sign Language Detection (BISINDO)
### Dataset Source
https://www.kaggle.com/datasets/agungmrf/indonesian-sign-language-bisindo

### About the model
The BISINDO Sign Language Detection is build using a CNN architecture for object recognition, in this case it is used to recognize hand gestures from images to be classified into 26 alphabet classes. The model mostly consist of fully connected convolutional layers which will learn from the provided dataset to find the patterns in hand gestures for each alphabets. The models has been trained, tested, and saved in formats compatible with deployment, including TensorFlow Lite (TFLite) for integration into Android applications.

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
   !git clone https://github.com/Isyara-App/MachineLearning
   ```
2. Install Dependencies
   - **CNN**:
      ```
      !pip install -r MachineLearning/BuildAndTrainModel/CNN/requirements.txt
      ```
   - **Transfer Learning**:
      ```
      !pip install -r MachineLearning/BuildAndTrainModel/TransferLearning/requirements.txt
      ```
3. Use Models
   
   The pretrained models are saved under:
   - **CNN**: `BuildAndTrainModel/CNN/models/`
   - **Transfer Learning**: `BuildAndTrainModel/TransferLearning/savedModels/`
   
   The models format saved as: `.h5` and `.tflite` (including its metadata)
4. Load Models
   - **For .h5 models**:
     
     - **CNN**:
       ```
       import tensorflow as tf
       from keras.models import load_model
       
       model = load_model('/content/MachineLearning/BuildAndTrainModel/CNN/models/trained_model.h5') # Replace the model path
       ```
     - **Transfer Learning**:
       ```
       import sys
       sys.path.append('/content/MachineLearning/BuildAndTrainModel/TransferLearning') # Replace path accordingly
       import tensorflow as tf
       from keras.models import load_model
       from custom_layers import PreprocessingLayer

       model = load_model('/content/MachineLearning/BuildAndTrainModel/TransferLearning/savedModels/handgesture_model.h5', custom_objects={'PreprocessingLayer': PreprocessingLayer}) # Replace the model path
       ```
   - **For .tflite models**:
        
        - **CNN**:
          ```
          import tensorflow as tf
          import numpy as np

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
          import numpy as np

          # Replace the model path
          interpreter = tf.lite.Interpreter(model_path="/content/MachineLearning/BuildAndTrainModel/TransferLearning/savedModels/model.tflite")
          interpreter.allocate_tensors()
          
          input_details = interpreter.get_input_details()
          output_details = interpreter.get_output_details()
          print("Inputs:", input_details)
          print("Outputs:", output_details)
          ```
5. Load and Preprocess Image

   Before making the prediction, the image need to be uploaded and processed first to match the model expected input size.
   - **For .h5 models**:
     
     - **CNN**:
       ```
       import cv2
       import numpy as np
       
       image = cv2.imread('/content/body dot (1).jpg')  # Replace with your image path
       image = cv2.resize(image, (150, 150))
       image = image / 255.0
       image = np.expand_dims(image, axis=0)
       ```
     - **Transfer Learning**:
       ```
       import cv2
       import numpy as np
       
       image = cv2.imread('/content/body dot (1).jpg')  # Replace with your image path
       image = cv2.resize(image, (224, 224))
       image = np.expand_dims(image, axis=0)
       ```
   - **For .tflite models**:
        
        - **CNN**:
          ```
          # Replace with image file path
          image = tf.io.read_file('/content/body dot (1).jpg')
          image = tf.image.decode_jpeg(image, channels=3)
          image = tf.image.resize(image, (150, 150))
          image = tf.cast(image, tf.float32) / 255.0 
          image = tf.expand_dims(image, axis=0)
          ```
       - **Transfer Learning**:
          ```
          # Replace with image file path
          image = tf.io.read_file('/content/body dot (1).jpg')
          image = tf.image.decode_jpeg(image, channels=3)
          image = tf.image.resize(image, (224, 224))
          image = tf.cast(image, tf.float32)
          image = tf.expand_dims(image, axis=0)
          ```
6. Predict the Image
     - **For .h5 models**:
       ```
       predictions = model.predict(image)
       predicted_class = np.argmax(predictions[0])
       
       # Map the labels to classes names
       class_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
       
       # Show class prediction
       print(f"Predicted class name: {class_names[predicted_class]}")
       ```
     - **For .tflite models**:
       ```
       # Set input tensor
       interpreter.set_tensor(input_details[0]['index'], image.numpy())
       
       # Run interpreter
       interpreter.invoke()
        
       # Get output with the highest class possibility
       output = interpreter.get_tensor(output_details[0]['index'])
       predicted_class = np.argmax(output)
       
       # labels
       class_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
       
       print(f"Predicted class name: {class_names[predicted_class]}")
       ```

### Members
| Name | ID Bangkit | University |
| ------------- |------------- | ------------- |
| Aldi Musneldi  | M497B4KY0328	  | Universitas Putra Indonesia Yptk Padang  |
| Cathleen Davina Hendrawan  | M233B4KX0906  | Universitas Katolik Soegijapranata  |
| Muhammad Rafi Abhinaya  | M006B4KY2989  |Universitas Brawijaya  |


