import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, Normalization
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from resampling import resample_data

# Loading the Data
x = np.load('data/observations.npy', allow_pickle=True)
y = np.load('data/actions.npy', allow_pickle=True)


# Output Layer Shape
num_actions = len(np.unique(y))

# One-hot Encoding the Labels
y = tf.keras.utils.to_categorical(y)

# Input Layer Shape
input_shape = x.shape[1:]
#x = np.array(x) / 10.0

# Splitting the Data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)

# Resampling the Data
x_train, y_train = resample_data(x_train, y_train)

tf.random.set_seed(42)


# Building the Model
model = Sequential([
    Conv2D(16, (3, 3), activation='relu', input_shape=input_shape),
    Conv2D(32, (3, 3), activation='relu'),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(num_actions, activation='softmax')

    # Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=input_shape),
    # MaxPooling2D((2, 2)),
    # Conv2D(128, (3, 3), activation='relu', padding='same'),
    # MaxPooling2D((2, 2)),
    # Flatten(),
    # Dense(256, activation='relu'),
    # Dropout(0.5),
    # Dense(num_actions, activation='softmax')
])


# Compiling the Model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=['accuracy'])

# Callbacks
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=0.00001)
early_stopping = EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)

# Training the Model
model.fit(x_train, y_train, 
          batch_size=16,
          epochs=50, 
          validation_data=(x_test, y_test),
          callbacks=[reduce_lr, early_stopping]
          )

# Saving the Model
model.save('models/minigrid_model.keras')

# Making Predictions
preds = model.predict(x_test)
print(list(map(lambda pred: np.argmax(pred), preds)))