import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense,Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.applications.mobilenet import MobileNet,preprocess_input
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import time


# Set TensorFlow session to allow dynamic GPU memory allocation
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)

# Load the styles.csv file into a pandas dataframe and extract the 'id' and 'articleType' columns.
df = pd.read_csv('styles.csv', usecols=['id', 'articleType'])


# Load the images from the 'images' folder using the 'id' column and convert them into a numpy array.
img_size = (224, 224)
X = []
y = []
for idx, row in df.iterrows():
    img_path = os.path.join('images_model', str(row['id']) + '.jpg')
    if os.path.exists(img_path):
        img = tf.keras.preprocessing.image.load_img(img_path, target_size=img_size)
        img = tf.keras.preprocessing.image.img_to_array(img)
        X.append(img)
        y.append(row['articleType'])
        print(img_path)


X = np.array(X)

le = LabelEncoder()
y = le.fit_transform(y)


print(X)
print(y)

# Split the data into training and validation sets.
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Resize and normalize the images.
X_train = tf.keras.applications.mobilenet_v2.preprocess_input(X_train)
X_val = tf.keras.applications.mobilenet_v2.preprocess_input(X_val)

# Define the model architecture.
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_size[0], img_size[1], 3), padding='same'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),
    Conv2D(256, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),
    Conv2D(512, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),
    Conv2D(1024, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(1024, activation='relu'),
    Dense(512, activation='relu'),
    Dense(256, activation='relu'),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(len(np.unique(y)), activation='softmax')
])
 #Set up callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, verbose=1, min_lr=1e-6)
checkpoint = ModelCheckpoint('model.h5', monitor='val_loss', verbose=1, save_best_only=True)

# Compile the model.
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model on the training set and validate it on the validation set.

start_time = time.time()
history = model.fit(X_train, y_train, epochs=40, batch_size=32, validation_data=(X_val, y_val),callbacks=[early_stop, reduce_lr, checkpoint])
end_time = time.time()
total_time = end_time - start_time


model.save('cnnmodel_3conv_layer(test)_forgraphs.h5')



# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig('accuracy_4.png')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig('loss_4.png')
plt.show()


from sklearn.metrics import classification_report

# Make predictions on the validation set
y_pred = model.predict(X_val)
y_pred = np.argmax(y_pred, axis=1)

# Convert encoded labels back to original labels
y_val_orig = le.inverse_transform(y_val)
y_pred_orig = le.inverse_transform(y_pred)

# Print classification report
print(classification_report(y_val_orig, y_pred_orig))

print(f"Total time taken: {total_time:.2f} seconds")
