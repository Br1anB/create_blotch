import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import random 

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

    #New Binary Area of the smooth/fuzzy blotches
    _, binary_blotch_mask = cv2.threshold(smoothed_mask_gaussian, 254, 255, cv2.THRESH_BINARY)
    #print(binary_blotch_mask.shape)
    binary_blotch_mask = cv2.bitwise_not(binary_blotch_mask)

    #Use connected components to add colour variants to each blotch
    num_labels, labels = cv2.connectedComponents(binary_blotch_mask[:,:,1])

    blotch_colour_mask = np.zeros_like(labels, dtype=np.uint8)

    print(num_labels)
    for label in range(1,num_labels):
        rand_colour = random.randint(0,255)

        blotch_colour_mask = np.where(labels==label, rand_colour, blotch_colour_mask)

    height, width = labels.shape
    blotch_colour_mask_RGB = np.zeros((height, width, 3), dtype=np.uint8)
    blotch_colour_mask_RGB[:,:,0] = blotch_colour_mask
    blotch_colour_mask_RGB[:,:,1] = blotch_colour_mask
    blotch_colour_mask_RGB[:,:,2] = blotch_colour_mask

    axs[1,1].imshow(blotch_colour_mask_RGB)
    axs[1,1].set_title("Coloured Blotches")
    axs[1,1].axis('off')  # Hide axis for image


    Y= smoothed_mask_gaussian.astype(np.float32) / 255.0 #Percentage Representation
    blotch_image = ((1-Y)*blotch_colour_mask_RGB) + (Y*image_array)
    plt.figure(figsize=(6, 6))
    plt.imshow(blotch_image.astype(np.uint8))
    plt.title("Blotchy")
    plt.axis('off')  # Hide axis for image


    plt.tight_layout()
    plt.show()

    return blotch_image


#image = Image.open('./test_images/1080p_rand.jpg')
#crudeBlotches(image)
#advancedBlotches(image)
