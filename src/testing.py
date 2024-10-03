import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

image = Image.open('./test_images/1080p_rand.jpg').convert('RGB')  # Convert to grayscale
image_array = np.array(image)

red_channel = image_array[:, :, 0]   # Red channel
green_channel = image_array[:, :, 1] # Green channel
blue_channel = image_array[:, :, 2]  # Blue channel

#Plot the RGB image
plt.figure(figsize=(6, 6))
plt.imshow(image_array)
plt.title("RGB Image")
plt.axis('off')  # Hide axis for image
plt.show()

#Plot specific row of RGB
specific_row = image_array.shape[0] // 2 #Middle Row

red_signal = red_channel[specific_row, :]
green_signal = green_channel[specific_row, :]
blue_signal = blue_channel[specific_row, :]

plt.figure(figsize=(10, 4))
plt.plot(red_signal, color='red', label='Red Channel')
plt.plot(green_signal, color='green', label='Green Channel')
plt.plot(blue_signal, color='blue', label='Blue Channel')
plt.title("RGB Signals from the Middle Row of Pixels")
plt.xlabel("Pixel Index")
plt.ylabel("Pixel Intensity")
plt.legend()
plt.show()