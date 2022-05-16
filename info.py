import tensorflow as tf
from tensorflow.python.client import device_lib
# tf.config.list_physical_devices(
#     device_type=None
# )

# tf.debugging.set_log_device_placement(True)
# gpus = tf.config.list_logical_devices('GPU')
# strategy = tf.distribute.MirroredStrategy(gpus)
# with strategy.scope():
#   inputs = tf.keras.layers.Input(shape=(1,))
#   predictions = tf.keras.layers.Dense(1)(inputs)
#   model = tf.keras.models.Model(inputs=inputs, outputs=predictions)
#   model.compile(loss='mse',
#                 optimizer=tf.keras.optimizers.SGD(learning_rate=0.2))

# gpus = tf.config.list_logical_devices('GPU')
# print(gpus)
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)


devices= device_lib.list_local_devices()
# import os
# import cv2
# import numpy as np
# path = '/home/ubuntu/gan_jose/images/'
# os.chdir(path)
# files = os.listdir(path)
# X_data=[]
# for file in files:
#    img = cv2.imread(file, cv2.IMREAD_COLOR)
#    img= cv2.resize(img,(512,512))
#    X_data.append(img)
# #    


# img_data = np.array(X_data)
# img_data = img_data.astype('float32')
# img_data /=255
# print(img_data.shape)
