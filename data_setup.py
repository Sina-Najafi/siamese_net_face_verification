import os
import tensorflow as tf

def preprocess_single_image(path):

  '''
  This function receives a path to an image and then returns a preprocessed image
  which can be used to train the model.
  '''

  # Read the image into a variable.
  image = tf.io.read_file(path)
  # decode the jgp image into a tensor.
  image = tf.io.decode_jpeg(image)
  # Resize the images to a specific size.
  image = tf.image.resize(image, (224,224))
  # map the image pixels between 0-1
  image = image/255
  return image

def preprocess(first_img, second_img, label):
  '''
  This function is the one that we directly call for preprocessing our images. because we have 
  two images and a label we pass them to this function and get back a tuple of two preprocessed 
  images and the corresponding label
  '''
  return(preprocess_single_image(first_img), preprocess_single_image(second_img), label)

def create_dataset(anchor_dir, positive_dir, negative_dir, train_split, batch_size):
  '''

  call this function with the necessary arguments (which their names are pretty intuitive and
  self_explanatory) to get a train and a validation tensorflow dataset to train a siamese network.
   the shape of each batch of the dataset will be like this:
       anchor_images             other_images          labels
  ((batch_size,224,224,3), (batch_size,224,224,3), (batch_size,1))

  here are the paths I used:

  anchor_dir = '/content/drive/MyDrive/bachelor_project/data/anchor'
  positive_dir = '/content/drive/MyDrive/bachelor_project/data/positive'
  negative_dir = '/content/drive/MyDrive/bachelor_project/data/negative'

  '''
  address_list = []
  for folder in os.listdir(anchor_dir):
    for filename in os.listdir(anchor_dir + '/'+ folder):
      address_list.append(anchor_dir + '/' + folder + '/' + filename)
      
  anchor = tf.data.Dataset.from_tensor_slices(address_list)


  address_list_1 = []
  for folder in os.listdir(positive_dir):
    for filename in os.listdir(positive_dir + '/'+ folder):
      address_list_1.append(positive_dir + '/' + folder + '/' + filename)
      
  positive = tf.data.Dataset.from_tensor_slices(address_list_1)


  negative = tf.data.Dataset.list_files(negative_dir+'/*.jpg')

  positives = tf.data.Dataset.zip((anchor, positive, tf.data.Dataset.from_tensor_slices(tf.ones(len(anchor)))))
  negatives = tf.data.Dataset.zip((anchor, negative, tf.data.Dataset.from_tensor_slices(tf.zeros(len(anchor)))))
  data = positives.concatenate(negatives)

  data = data.map(preprocess)
  data = data.shuffle(len(data))
  data = data.batch(batch_size)
  train_data = data.take(int(len(data) * train_split))
  val_data = data.skip(int(len(data) * train_split))

  return train_data, val_data