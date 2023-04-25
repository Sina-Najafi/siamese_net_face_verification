import tensorflow as tf
from tensorflow.keras import applications
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import optimizers
from tensorflow.keras import metrics
from tensorflow.keras import Model
from tensorflow.keras.applications.resnet import ResNet50

def embedding_model(input_shape): # (224, 224, 3)
  '''
  
  this function turn our two images into two vectors and calcuates their
  distance and returns one single vector of distance
  
  '''

  # Use a pretrained model(ResNet50 in our case) for feature extraction.
  base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)

  # Freeze the layers of the pretrained model.
  for layer in base_model.layers:
      layer.trainable = False

  # Add extra layers on top of the pretrained model.
  x = tf.keras.layers.Conv2D(filters = 512, kernel_size = 1, activation = 'relu', padding = 'same')(base_model.output)
  x = tf.keras.layers.Flatten()(x)
  x = tf.keras.layers.Dense(128, activation='relu')(x)

  # Create and return the model.
  model = tf.keras.models.Model(base_model.input, x)
  return model

def siamese_network(input_shape): # (224, 224, 3)
  '''
  
  this function returns the siamese network model that I used.
  you can check out the model by .summary() method after you put the model
  in a variable.

  '''
  
  # Receive two images as inputs.
  left_input = layers.Input(shape=input_shape)
  right_input = layers.Input(shape=input_shape)

  # Create an instance of the embedding model.
  model = embedding_model(input_shape)

  # Pass two images through the embedding model.
  left = model(left_input)
  right = model(right_input)

  # Create a custom layer that calculates the distance between two vectors.
  distance_layer = layers.Lambda(lambda x: tf.math.abs(x[0] - x[1]))

  # Calculate the distance between two embeddings by passing them through the distance layer.
  distance = distance_layer([left, right])

  # Output layer that is supposed to determine whether the two images are of the same person.
  output_layer = layers.Dense(1, activation='sigmoid')(distance)

  # Create and return the siamese network model.
  siamese_model = tf.keras.models.Model(inputs=[left_input, right_input], outputs=[output_layer, distance])
  return siamese_model