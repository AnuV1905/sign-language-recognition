# Importing the tensorflow.Keras libraries and packages
# from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import layers, Sequential
# from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.models import model_from_json
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import os
# Set visible GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sz = 128
BATCH_SIZE = 32
EPOCHS = 15
tf.random.set_seed(42)

#Data loading
training_set = tf.keras.utils.image_dataset_from_directory('data2/train',
                                                 labels='inferred',
                                                 image_size=(sz, sz),
                                                 batch_size=BATCH_SIZE,
                                                 color_mode='grayscale',
                                                 label_mode='categorical',
                                                 shuffle=True)

test_set = tf.keras.utils.image_dataset_from_directory('data2/test',
                                            labels='inferred',
                                            image_size=(sz , sz),
                                            batch_size=BATCH_SIZE,
                                            color_mode='grayscale',
                                            label_mode='categorical',
                                            shuffle=False) 

# Get dynamic number of classes
num_classes = len(training_set.class_names)

#Data Augmentation layer  + Normalisation
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.1),
    layers.Rescaling(1./255)
])

# Normalization Layer
# normalization_layer = layers.Rescaling(1./255)

def process_images(image, label):
    return data_augmentation(image), label

# Apply augmentation + normalization

# training_set = training_set.map(lambda x, y: (data_augmentation(x, training=True), y))
# training_set = training_set.map(lambda x, y: (normalization_layer(x), y))
# test_set = test_set.map(lambda x, y: (normalization_layer(x), y))
training_set = training_set.map(process_images, num_parallel_calls=tf.data.AUTOTUNE)
test_set = test_set.map(lambda x, y: (layers.Rescaling(1./255)(x), y), num_parallel_calls=tf.data.AUTOTUNE)

# AUTOTUNE = tf.data.AUTOTUNE
# training_set = training_set.prefetch(buffer_size=AUTOTUNE)
# test_set = test_set.prefetch(buffer_size=AUTOTUNE)
training_set = training_set.prefetch(buffer_size=tf.data.AUTOTUNE)
test_set = test_set.prefetch(buffer_size=tf.data.AUTOTUNE)

#visualize augmented images
for images, _ in training_set.take(1):
    plt.figure(figsize=(10, 2))
    for i in range(5):
        ax = plt.subplot(1, 5, i + 1)
        plt.imshow(images[i].numpy().squeeze(), cmap='gray')
        plt.axis("off")
    plt.suptitle("Sample Augmented Images")
    plt.show()


# Building the CNN

# # Initializing the CNN
# classifier = Sequential()

# # First convolution layer and pooling
# classifier.add(Convolution2D(64, (3, 3), input_shape=(sz, sz, 1), activation='relu'))
# classifier.add(MaxPooling2D(pool_size=(2, 2)))
# # Second convolution layer and pooling
# classifier.add(Convolution2D(64, (3, 3), activation='relu'))
# # input_shape is going to be the pooled feature maps from the previous convolution layer
# classifier.add(MaxPooling2D(pool_size=(2, 2)))
# #classifier.add(Convolution2D(32, (3, 3), activation='relu'))
# # input_shape is going to be the pooled feature maps from the previous convolution layer
# #classifier.add(MaxPooling2D(pool_size=(2, 2)))

# # Flattening the layers
# classifier.add(Flatten())

# # Adding a fully connected layer
# classifier.add(Dense(units=256, activation='relu'))
# classifier.add(Dropout(0.30))
# classifier.add(Dense(units=128, activation='relu'))
# classifier.add(Dropout(0.30))
# # classifier.add(Dense(units=64, activation='relu'))
# # classifier.add(Dense(units=37, activation='softmax')) # softmax for more than 2
# classifier.add(Dense(units=num_classes, activation='softmax'))
classifier = Sequential([
    layers.Input(shape=(sz, sz, 1)),
    Convolution2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Convolution2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Convolution2D(256, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.3),
    Dense(num_classes, activation='softmax')
])

# # Compiling the CNN
classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) # categorical_crossentropy for more than 2
classifier.summary()


# Step 2 - Preparing the train/test data and training the model
# Code copied from - https://tensorflow.keras.io/preprocessing/image/

# train_datagen = image_dataset_from_directory(
#         rescale=1./255,
#         shear_range=0.2,
#         zoom_range=0.2,
#         horizontal_flip=True)

# test_datagen = image_dataset_from_directory(rescale=1./255)

#callbacks
callbacks = [
    EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
    ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_loss')
]

#train model
history = classifier.fit(
        training_set,
        # steps_per_epoch=12841, # No of images in training set
        epochs=EPOCHS,
        validation_data=test_set,
        # validation_steps=4268, # No of images in test set
        callbacks=callbacks)


# Saving the model
model_json = classifier.to_json()
with open("model-bw.json", "w") as json_file:
    json_file.write(model_json)
print('Model Saved')
classifier.save_weights('model-bw.weights.h5')
print('Weights saved')





#plot accuracy and loss
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Model Accuracy Over Epochs')
plt.legend()
plt.grid(True)
plt.show()

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Model Loss Over Epochs')
plt.legend()
plt.grid(True)
plt.show()

#to cross verify
print("Train classes:", len(training_set.class_names))
print("Test classes:", len(test_set.class_names))

# Confusion Matrix + Class-wise Accuracy
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Step 1: Flatten test_set into arrays
y_true = []
y_pred = []

for images, labels in test_set:
    preds = classifier.predict(images)
    y_true.extend(np.argmax(labels.numpy(), axis=1))
    y_pred.extend(np.argmax(preds, axis=1))

# Step 2: Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=training_set.class_names,
            yticklabels=training_set.class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig('confusion_matrix.png')  # 💾 Save plot
plt.show()

# Step 3: Class-wise Accuracy Report
report = classification_report(y_true, y_pred, target_names=training_set.class_names, output_dict=True)

# Convert to class-wise accuracy bar plot
labels = list(report.keys())[:-3]  # Remove 'accuracy', 'macro avg', 'weighted avg'
accuracies = [report[label]['recall'] * 100 for label in labels]

plt.figure(figsize=(12, 6))
sns.barplot(x=labels, y=accuracies, palette='viridis')
plt.xticks(rotation=45)
plt.ylabel('Accuracy (%)')
plt.title('Class-wise Accuracy')
plt.ylim(0, 100)
plt.tight_layout()
plt.savefig('classwise_accuracy.png')  # 💾 Save bar plot
plt.show()

# Optional: Print accuracy text summary
print("\n🔍 Class-wise Accuracy:")
for cls in labels:
    acc = report[cls]['recall'] * 100
    print(f"{cls:<10} : {acc:.2f}%")
