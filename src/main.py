import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

#Local Imports
import blotch_func

def crudeBlotches(image):
    plt.style.use('dark_background')
    fig, axs = plt.subplots(2, 2, figsize=(8, 8))

    image_array = np.array(image)
    #axs[0,0].figure(1, figsize=(6, 6))
    axs[0,0].imshow(image_array)
    axs[0,0].set_title("Image Under Test")
    axs[0,0].axis('off')  # Hide axis for image

    #Generate Blotch Mask
    blotch_mask = blotch_func.generateBlotch_new(image, 10, 0.5, 30)
    blotch_mask = blotch_mask*255 #Conver to 255 range

    #Display Blotch Mask
    #axs[0,1].figure(2, facecolor='black', figsize=(6, 6))
    axs[0,1].imshow(blotch_mask)
    axs[0,1].set_title("Blotch Mask")
    axs[0,1].axis('off')  # Hide axis for image
    #plt.show()

    #Generate Image X Blotch Mask
    Y = np.where(blotch_mask < 100, 1, 0)#Y indicates where a Blotch is detected
    blotch_image = (blotch_mask*Y) + ((1-Y)*image_array)#Use Blotch mask where Y=1, hence where there's a blotch

    #axs[1,0].figure(3, figsize=(6, 6))
    axs[1,0].imshow(blotch_image)
    axs[1,0].set_title("Image X Blotch Mask")
    axs[1,0].axis('off')  # Hide axis for image
    
    plt.tight_layout()
    plt.show()


def advancedBlotches(image):
    #Similar to crude although edges of blotches will be softened
    plt.style.use('dark_background')
    fig, axs = plt.subplots(2, 2, figsize=(8, 8))

    image_array = np.array(image)
    #axs[0,0].figure(1, figsize=(6, 6))
    axs[0,0].imshow(image_array)
    axs[0,0].set_title("Image Under Test")
    axs[0,0].axis('off')  # Hide axis for image

    #Generate Blotch Mask
    blotch_mask = blotch_func.generateBlotch_new(image, 10, 0.5, 100)
    blotch_mask = blotch_mask*255 #Conver to 255 range

    #Display Blotch Mask
    #axs[0,1].figure(2, facecolor='black', figsize=(6, 6))
    axs[0,1].imshow(blotch_mask)
    axs[0,1].set_title("Blotch Mask")
    axs[0,1].axis('off')  # Hide axis for image
    
    #Add some kind of blur on the Mask
    smoothed_mask_gaussian = cv2.GaussianBlur(blotch_mask, (13, 13), sigmaX=3, sigmaY=3)

    # Displaying the results

    axs[1,0].imshow(smoothed_mask_gaussian)
    axs[1,0].set_title("Gaussian Blur")
    axs[1,0].axis('off')  # Hide axis for image

    # Observe Smoothened Signal
    #red_channel = smoothed_mask_gaussian[:, :, 0]   # Red channel
    #axs[1,1].plot(red_channel[500, :], color='red', label='Red Channel')

    plt.tight_layout()
    plt.show()


image = Image.open('./test_images/1080p_rand.jpg')
crudeBlotches(image)
#advancedBlotches(image)