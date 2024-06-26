import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

data_path = 'Full_Data_Raw.csv'
data = pd.read_csv(data_path)

X = data.iloc[:, :-1].values  # Features
y = data.iloc[:, -1].values  # Labels

X_reshaped = X.reshape((X.shape[0], X.shape[1], 1))

y_one_hot = to_categorical(y)

X_train, X_test, y_train, y_test = train_test_split(
    X_reshaped,
    y_one_hot,
    test_size=0.25,
    random_state=55,
    shuffle=True
)

# Model definition
model = Sequential([
    # Block 1
    Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)),
    Conv1D(filters=64, kernel_size=3, activation='relu'),
    MaxPooling1D(pool_size=2, strides=2),

    # Block 2
    Conv1D(filters=128, kernel_size=3, activation='relu'),
    Conv1D(filters=128, kernel_size=3, activation='relu'),
    MaxPooling1D(pool_size=2, strides=2),

    # Block 3
    Conv1D(filters=256, kernel_size=3, activation='relu'),
    Conv1D(filters=256, kernel_size=3, activation='relu'),
    Conv1D(filters=256, kernel_size=3, activation='relu'),
    MaxPooling1D(pool_size=2, strides=2),

    # Block 4
    Conv1D(filters=512, kernel_size=3, activation='relu'),
    Conv1D(filters=512, kernel_size=3, activation='relu'),
    Conv1D(filters=512, kernel_size=3, activation='relu'),
    MaxPooling1D(pool_size=2, strides=2),

    # Block 5
    Conv1D(filters=512, kernel_size=3, activation='relu'),
    Conv1D(filters=512, kernel_size=3, activation='relu'),
    Conv1D(filters=512, kernel_size=3, activation='relu'),
    MaxPooling1D(pool_size=2, strides=2),

    Flatten(),
    Dense(200, activation='relu'),
    Dense(200, activation='relu'),
    Dense(y_one_hot.shape[1], activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=100, batch_size=64, validation_split=0.2)

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Test accuracy: {test_acc * 100:.2f}%')

plt.figure(figsize=(14, 5))

# Plot training accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')

# Plot training loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper right')

plt.show()
