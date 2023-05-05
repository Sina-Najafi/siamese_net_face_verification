import tensorflow as tf

def predict(model, img1_img2_tuple):
  y_pred_prob,_ = model(img1_img2_tuple)
  y_pred = tf.round(y_pred_prob)
  return y_pred_prob, y_pred