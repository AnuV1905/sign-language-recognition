import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"✅ GPU available: {gpus}")
else:
    print("❌ No GPU detected by TensorFlow")
