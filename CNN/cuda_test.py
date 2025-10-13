import tensorflow as tf

# This will list all physical devices TensorFlow can see.
# You should see your CPU and your GTX 1080 listed as a GPU.
print(tf.config.list_physical_devices())

# A more direct check:
gpu_devices = tf.config.list_physical_devices('GPU')
if gpu_devices:
    print(f"TensorFlow has detected {len(gpu_devices)} GPU(s):")
    for device in gpu_devices:
        print(device)
else:
    print("TensorFlow did NOT detect any GPUs.")