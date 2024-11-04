import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import random 

#Local Imports
import blotch_func

def generate_blotch_colour():
    # Generate a random number between 0 and 1
    r = random.random()
    # 80% chance for ranges [0-55] or [200-255]
    if r < 0.8:
        # Pick randomly between the two ranges
        if random.random() < 0.5:
            return random.randint(0, 55)
        else:
            return random.randint(200, 255)
    # 20% chance for the range [56-199]
    else:
        return random.randint(56, 199)

def advancedBlotches(image):
    #Similar to crude although edges of blotches will be softened
    #plt.style.use('dark_background')
    #fig, axs = plt.subplots(2, 2, figsize=(8, 8))

    image_array = np.array(image)

    #Generate Blotch Mask
    blotch_mask = blotch_func.generateBlotch_new(image, 5, 0.5, 30)
    blotch_mask = blotch_mask*255 #Conver to 255 range
    
    #Add some kind of blur on the Mask
    smoothed_mask_gaussian = cv2.GaussianBlur(blotch_mask, (13, 13), sigmaX=3, sigmaY=3)

    #New Binary Area of the smooth/fuzzy blotches
    _, binary_blotch_mask = cv2.threshold(smoothed_mask_gaussian, 254, 255, cv2.THRESH_BINARY)
    #print(binary_blotch_mask.shape)
    binary_blotch_mask = cv2.bitwise_not(binary_blotch_mask)

    #Use connected components to add colour variants to each blotch
    num_labels, labels = cv2.connectedComponents(binary_blotch_mask[:,:,1])

    blotch_colour_mask = np.zeros_like(labels, dtype=np.uint8)

    #print(num_labels)
    for label in range(1,num_labels):
        rand_colour = generate_blotch_colour()

        blotch_colour_mask = np.where(labels==label, rand_colour, blotch_colour_mask)

    height, width = labels.shape
    blotch_colour_mask_RGB = np.zeros((height, width, 3), dtype=np.uint8)
    blotch_colour_mask_RGB[:,:,0] = blotch_colour_mask
    blotch_colour_mask_RGB[:,:,1] = blotch_colour_mask
    blotch_colour_mask_RGB[:,:,2] = blotch_colour_mask


    Y= smoothed_mask_gaussian.astype(np.float32) / 255.0 #Percentage Representation
    blotch_image = ((1-Y)*blotch_colour_mask_RGB) + (Y*image_array)

    return blotch_image


output_dir = "/home/brian/Desktop/MAI Project/degrade_vid_pres/processed_frames"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


#video_path = "../../CLEAN_VIDEO/FHD/19.mp4"
video_path = "/home/brian/Desktop/MAI Project/degrade_vid_pres/CLEAN_VIDEO/FHD/19.mp4"
cap = cv2.VideoCapture(video_path)

frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("ERROR")
        break  # Stop if the video ends
    
    # Convert the frame from BGR (used by OpenCV) to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Convert the NumPy array frame to a PIL image
    pil_image = Image.fromarray(frame_rgb)

    # Apply your advancedBlotch function
    processed_image = advancedBlotches(frame)

    # Save the processed frame as an image (JPEG/PNG)
    output_path = os.path.join(output_dir, f"frame_{frame_count:05d}.png")
    cv2.imwrite(output_path, processed_image)
    #if cv2.imwrite(output_path, processed_image):
    #    print(f"Saved frame {frame_count} as {output_path}")
    #else:
    #    print(f"Error: Could not save frame {frame_count}.")

    frame_count += 1
    #break

cap.release()  # Release the video capture object
print(f"Processed {frame_count} frames.")
