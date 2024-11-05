import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from dataset_loader import *


model = load_model('autoencoder_model.h5')


watermarked_image = load_data('data/watermarked_images')[:1]
predicted_image = model.predict(watermarked_image)


plt.subplot(1,2,1)
plt.title("Watermarked Image")
plt.imshow(predicted_image[0])

plt.subplot(1, 2, 2)
plt.title("Output without Watermark")
plt.imshow(predicted_image[0])
plt.show()