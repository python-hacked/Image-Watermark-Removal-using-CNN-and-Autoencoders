from dataset_loader import *
from models import *


orignal_image = load_data('data/original_images')
watermarked_images = load_data('data/watermarked_images')

model = create_autoencoder(inpute_shape=(128,128,3))
model.summary()



model.fit(watermarked_images,orignal_image,epochs=50, batch_size=16,)
model.save("autoencoder_model.keras")   



