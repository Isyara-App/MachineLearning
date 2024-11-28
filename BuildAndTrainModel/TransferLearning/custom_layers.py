# custom layer for the model
import tensorflow as tf

class PreprocessingLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(PreprocessingLayer, self).__init__(**kwargs)
        self.preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

    def call(self, inputs):
        return self.preprocess_input(inputs)