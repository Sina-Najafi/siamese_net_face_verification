import tensorflow as tf
import numpy as np

def train_step(model, single_batch, loss_fn, optimizer):

  '''Trains the model on a single batch of data.
  this function will be used inside the main train function'''
  
  with tf.GradientTape() as tape:
    
    X = single_batch[:2] # two images
    y = single_batch[2] # label(-1- if same person and -0- otherwise)

    # forward pass
    y_pred,_ = model(X, training = True)
    # calculate the loss
    loss = loss_fn(y, y_pred)
  # calculate gradients
  grad = tape.gradient(loss, model.trainable_variables)
  
  # update weights via gradient descent
  optimizer.apply_gradients(zip(grad, model.trainable_variables))

  return loss, y_pred


def train(model, train_data, validation_data, num_epochs, loss_fn, optimizer):

  '''
  
  Call this function to train a siamese network model.
  for example:
  history = train(sn,train_data,val_data,3,tf.losses.BinaryCrossentropy(),tf.keras.optimizers.Adam())
  it returns a dictionary of loss and accuracy histories during training.

  '''
  loss_history_per_epoch = []
  accuracy_history_per_epoch = []
  val_loss_history_per_epoch = []
  val_accuracy_history_per_epoch = []

  ####################### training loop and calculation of the train loss and accuracy #######################
  for epoch in range(num_epochs): 
    batch_losses = []
    val_batch_losses = []
    m1 = tf.keras.metrics.Accuracy()
    # Print what epoch you are at.
    print(f'\n Epoch {epoch+1}/{num_epochs}')
    # Progress bar to see the progress.
    progress_bar = tf.keras.utils.Progbar(len(train_data))

    for idx, batch in enumerate(train_data):
      # keep track of the loss of each batch
      loss_per_batch, yHat_per_batch = train_step(model, batch, loss_fn, optimizer)
      batch_losses.append(loss_per_batch)
      m1.update_state(tf.expand_dims(batch[2], axis=1), tf.round(yHat_per_batch))
      progress_bar.update(idx + 1)

    # keep track of the loss and accuracy per epoch.
    loss_history_per_epoch.append(float(np.mean(np.array(batch_losses))))
    accuracy_history_per_epoch.append(float(m1.result().numpy()))
    ###########################################################################################################
    
    ####################### evaluation per epoch on validation set ######################
    if validation_data is not None:
      m2 = tf.keras.metrics.Accuracy()
      for test_batch in validation_data:
        yHat_test_per_batch,_ = model(test_batch[:2])
        val_loss_per_batch = loss_fn(test_batch[2], yHat_test_per_batch)
        val_batch_losses.append(val_loss_per_batch)
        m2.update_state(tf.expand_dims(test_batch[2], axis=1), tf.round(yHat_test_per_batch))

      # keep track of the validation loss and accuracy per epoch.
      val_loss_history_per_epoch.append(float(np.mean(np.array(val_batch_losses))))
      val_accuracy_history_per_epoch.append(float(m2.result().numpy()))
    ######################################################################################
    
    ####################### print the loss and accuracy ########################
    print(f'loss: {loss_history_per_epoch[epoch]}, accuracy: {accuracy_history_per_epoch[epoch]}')
    if validation_data is not None:
      print(f'val_loss: {val_loss_history_per_epoch[epoch]}, val_accuracy: {val_accuracy_history_per_epoch[epoch]}')
    #######################################################################################

    ####################### return the history object #########################
  if validation_data is not None:
    return {'loss_history':loss_history_per_epoch,
            'accuracy_history':accuracy_history_per_epoch,
            'validation_loss_history':val_loss_history_per_epoch,
            'validation_accuracy_history':val_accuracy_history_per_epoch}
  return {'loss_history':loss_history_per_epoch,
          'accuracy_history':accuracy_history_per_epoch}
  
