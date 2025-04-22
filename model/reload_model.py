# from tensorflow.keras.models import model_from_json, Sequential
# from tensorflow.keras.saving.legacy import model_from_json
# from tensorflow.keras.models import load_weights  # if needed
# import json
from tensorflow.keras.models import model_from_json, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.initializers import VarianceScaling, Zeros

# Step 1: Register known classes (just in case)
custom_objects = {
    'Sequential': Sequential,
    'Conv2D': Conv2D,
    'MaxPooling2D': MaxPooling2D,
    'Flatten': Flatten,
    'Dense': Dense,
    'Dropout': Dropout,
    'VarianceScaling': VarianceScaling,
    'Zeros': Zeros
}


# Load model structure from JSON
with open('C:/Users/HP/OneDrive/Desktop/coding/PROJECTS/SLR-luv/sign-language-recognition/model/model-bw_tkdi.json', 'r') as json_file:
    model_json = json_file.read()

# Load the model structure
model = model_from_json(model_json, custom_objects=custom_objects)

# Load weights if necessary
model.load_weights('model/model-bw_tkdi.h5')

print("Model loaded successfully.")
