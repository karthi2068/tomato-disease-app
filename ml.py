import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# -------------------------
# Step 1: Dataset Paths
# -------------------------
train_dir = r"E:\tomato\archive\train"
val_dir   = r"E:\tomato\archive\valid"

# -------------------------
# Step 2: Data Generators
# -------------------------
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen   = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='sparse',   # gives integer labels (0...9)
    shuffle=True
)

val_data = val_datagen.flow_from_directory(
    val_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='sparse',
    shuffle=True
)

# -------------------------
# Step 3: CNN Model
# -------------------------
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)),
    MaxPooling2D((2,2)),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D((2,2)),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')  # 10 classes
])

# -------------------------
# Step 4: Compile Model
# -------------------------
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',  # matches integer labels
    metrics=['accuracy']
)

# -------------------------
# Step 5: Train Model
# -------------------------
history = model.fit(
    train_data,
    epochs=20,
    validation_data=val_data
)

# -------------------------
# Step 6: Save Model
# -------------------------
model.save("tomato_disease_leaf_cnn.h5")
print("✅ Model saved as tomato_disease_leaf_cnn.h5")
