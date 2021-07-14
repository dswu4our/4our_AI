#import tensorflow as tf

#hello = tf.constant('hello tensorflowfsdadad')

#tf.print(hello)
#import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow.python.client import device_lib

def get_available_devices():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos]
print(get_available_devices())