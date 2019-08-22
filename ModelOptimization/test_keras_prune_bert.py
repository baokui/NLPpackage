from tensorflow_model_optimization.sparsity import keras as sparsity
import numpy as np
import tensorflow as tf
# tf.enable_eager_execution()
from tensorflow.examples.tutorials.mnist import input_data
import tensorboard
import tempfile
import zipfile
import os
import keras
from keras_bert import get_base_dict, get_model, compile_model, gen_batch_inputs
model = get_model(
    token_num=20000,
    head_num=5,
    transformer_num=12,
    embed_dim=25,
    feed_forward_dim=100,
    seq_len=20,
    pos_num=20,
    dropout_rate=0.05,
)
compile_model(model)
model.summary()
new_pruning_params = {
    'pruning_schedule': sparsity.PolynomialDecay(initial_sparsity=0.50,
                                                 final_sparsity=0.90,
                                                 begin_step=0,
                                                 end_step=1000,
                                                 frequency=100)
}
new_pruned_model = sparsity.prune_low_magnitude(model, **new_pruning_params)
new_pruned_model.summary()

new_pruned_model.compile(
    loss=tf.keras.losses.categorical_crossentropy,
    optimizer='adam',
    metrics=['accuracy'])
logdir = tempfile.mkdtemp()
print('Writing training logs to ' + logdir)
final_model = sparsity.strip_pruning(new_pruned_model)
final_model.summary()
_, new_pruned_keras_file = tempfile.mkstemp('.h5')
print('Saving pruned model to: ', new_pruned_keras_file)
tf.keras.models.save_model(final_model, new_pruned_keras_file,
                           include_optimizer=False)
_, zip3 = tempfile.mkstemp('.zip')
with zipfile.ZipFile(zip3, 'w', compression=zipfile.ZIP_DEFLATED) as f:
    f.write(new_pruned_keras_file)
print("Size of the pruned model before compression: %.2f Mb"
      % (os.path.getsize(new_pruned_keras_file) / float(2 ** 20)))
print("Size of the pruned model after compression: %.2f Mb"
      % (os.path.getsize(zip3) / float(2 ** 20)))

# convert to tflite
tflite_model_file = '/tmp/sparse_mnist.tflite'
converter = tf.lite.TFLiteConverter.from_keras_model_file(new_pruned_keras_file)
tflite_model = converter.convert()
with open(tflite_model_file, 'wb') as f:
    f.write(tflite_model)
    _, zip_tflite = tempfile.mkstemp('.zip')
with zipfile.ZipFile(zip_tflite, 'w', compression=zipfile.ZIP_DEFLATED) as f:
    f.write(tflite_model_file)
print("Size of the tflite model before compression: %.2f Mb"
      % (os.path.getsize(tflite_model_file) / float(2 ** 20)))
print("Size of the tflite model after compression: %.2f Mb"
      % (os.path.getsize(zip_tflite) / float(2 ** 20)))
