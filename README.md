# Siamese Network for Face Verification and recognition
Here you will find an Implementation of a Siamese Network for face verification and face recognition in tensorflow.
next I'll provide some explanation for you, So you can easily use this project for yourself.
don't forget to mention my name :)

## Folder Structure For The Data
your images must be in three folders named anchor, positive, negative all in one data folder:

                                                      
some_path<br>
-------------/data<br>
--------------------/anchor<br>
--------------------/negative<br>
--------------------/positive<br>

for example in my case some_path was: '/content/drive/MyDrive/bachelor_project'

## data_setup.py
In this python script you will find the code that can create and return tensorflow datasets for train and test. you'll have to
specify the necessary paths to your data according to the arguments. all the required preprocessings will be taken care of.

## model.py
This script contains the siamese network model that will be trained. please take note that the model output returns both
a distance vector and one single number tensor between [0,1].

## train.py
This script contains functions that can be used to train the model. you will have to provide the model, loss funnction, optimizer
number of epochs, train data, validation data as input to start the training.

## notebook folder
In this folder, you will find a .ipynb file that I initially used to train the model. everything is there in one notebook.

## How to get things done right away...
First import the create_dataset, siamese_network and train functions from the python scripts then
you can modify the following code to instantly train this siamese network model:

```
train_data, val_data = create_dataset(anchor_dir = '/content/drive/MyDrive/bachelor_project/data/anchor',
               positive_dir = '/content/drive/MyDrive/bachelor_project/data/positive',
               negative_dir = '/content/drive/MyDrive/bachelor_project/data/negative',
               train_split = 0.8,
               batch_size = 8)

sn = siamese_network(input_shape = (224,224,3))

loss_fn = tf.losses.BinaryCrossentropy()
optimizer = tf.keras.optimizers.Adam()
history = train(sn,train_data,val_data,3,loss_fn,optimizer)
```
                            
                            
