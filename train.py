import pandas as pd
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator

from carnet import Carnet

epochs = 2
batch_size = 64
input_shape = (56, 65, 3)

# Step 1: Read CSV file
data = pd.read_csv('paths/UFPR05.csv', dtype={'label': str})

# Step 2: Define ImageDataGenerator for Data Augmentation and Preprocessing
datagen = ImageDataGenerator(
    rescale=1./255,  # Normalize pixel values
    # rotation_range=20,  # Random rotation
    # width_shift_range=0.2,  # Random horizontal shift
    # height_shift_range=0.2,  # Random vertical shift
    # horizontal_flip=True  # Random horizontal flip
)

# Step 3: Split data into train and test sets
train_data, test_data = train_test_split(data, test_size=0.1, random_state=42)
# train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42)

# Step 4: Create generators
train_generator = datagen.flow_from_dataframe(
    dataframe=train_data,
    directory=None,  # Assuming paths are absolute or relative to current directory
    x_col='image_path',
    y_col='label',
    target_size=input_shape[:-1],   # should be changed!!!
    batch_size=batch_size,
    class_mode='binary'
    # class_mode='categorical'
)

test_generator = datagen.flow_from_dataframe(
    dataframe=test_data,
    directory=None,
    x_col='image_path',
    y_col='label',
    target_size=input_shape[:-1],   # should be changed!!!
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False
)
# Step 5: Define Keras Model
model = Carnet(*input_shape).model

# Step 6: Compile Model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Step 7: Train the Model using fit_generator
model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=epochs,
    validation_data=test_generator,
    validation_steps=len(test_generator)
)
