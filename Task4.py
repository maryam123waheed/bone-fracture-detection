import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

fracture_dir = "C:/Users/marya/Downloads/Tasks data/archive (6)/val/fractured"
non_fracture_dir = "C:/Users/marya/Downloads/Tasks data/archive (6)/val/not fractured"

fracture_images = [os.path.join(fracture_dir, file) for file in os.listdir(fracture_dir)]
non_fracture_images = [os.path.join(non_fracture_dir, file) for file in os.listdir(non_fracture_dir)]

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

train_images = fracture_images + non_fracture_images
train_labels = np.concatenate([np.ones(len(fracture_images)), np.zeros(len(non_fracture_images))])

processed_images = []
for image_path in train_images:
    image = cv2.imread(image_path)
    if image is None:
        continue  # Skip invalid image paths
    image = cv2.resize(image, (64, 64))  
    processed_images.append(image)

train_images = np.array(processed_images)
train_images = train_images / 255.0

model.fit(train_images, train_labels.reshape(-1, 1), epochs=10, batch_size=32)

test_image_path = "C:/Users/marya/Downloads/Tasks data/archive (6)/train/fractured/10.jpg"
test_image = cv2.imread(test_image_path)
test_image_resized = cv2.resize(test_image, (64, 64))
test_image_normalized = test_image_resized / 255.0
test_image_input = np.expand_dims(test_image_normalized, axis=0)

prediction = model.predict(test_image_input)
if prediction > 0.5:
    print("Fractured")
else:
    print("Not Fractured")

test_image_equalized = cv2.cvtColor(test_image_resized, cv2.COLOR_BGR2YUV)
test_image_equalized[:, :, 0] = cv2.equalizeHist(test_image_equalized[:, :, 0])
test_image_equalized = cv2.cvtColor(test_image_equalized, cv2.COLOR_YUV2BGR)

test_image_denoised = cv2.fastNlMeansDenoisingColored(test_image_equalized, None, 10, 10, 7, 21)

plt.figure(figsize=(12, 4))
plt.subplot(131)
plt.title("Original Image")
plt.imshow(cv2.cvtColor(test_image_resized, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(132)
plt.title("Enhanced Image")
plt.imshow(cv2.cvtColor(test_image_equalized, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(133)
plt.title("Denoised Image")
plt.imshow(cv2.cvtColor(test_image_denoised, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.show()
