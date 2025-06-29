import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint

# Cấu hình chung
IMG_SIZE = 64
BATCH_SIZE = 32
EPOCHS = 20
DATASET_PATH = 'train/'  # Đường dẫn tới folder chứa dữ liệu ảnh

# Tiền xử lý ảnh: scale và chia tập train/val
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_gen = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode='grayscale',
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_gen = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode='grayscale',
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# Số lớp cảm xúc
num_classes = len(train_gen.class_indices)
print("Các lớp cảm xúc:", train_gen.class_indices)

# Xây dựng mô hình CNN đơn giản
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dense(num_classes, activation='softmax')
])

# Biên dịch
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Lưu mô hình tốt nhất
checkpoint = ModelCheckpoint('emotion_model.h5', monitor='val_accuracy', save_best_only=True)

# Train mô hình
model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=[checkpoint]
)
