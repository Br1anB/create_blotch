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
    blotch_mask = blotch_func.generateBlotch_new(image, 1, 0.8, 30) #max num blotches 1, prob 0.8, max size 30
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
    blotch_image = ((1 - Y) * blotch_colour_mask_RGB) + (Y * image_array)

    return blotch_image, binary_blotch_mask


# Cuts input frame into patches, adds blotches, reconstructs.
def processFrame(input_frame, patch_size):
    #print("Processing Frame, Shape:")
    #print(input_frame.shape)
    height, width = input_frame.shape[:2]
    
    #initialise output frame as all zeros
    output_frame = np.zeros_like(input_frame)
    mask_frame = np.zeros_like(input_frame)

    for y in range(0, height, patch_size):
        for x in range(0, width, patch_size):
            #Initialise patch
            patch = input_frame[y:y+patch_size, x:x+patch_size]

            processed_patch, mask_patch = advancedBlotches(patch)

            output_frame[y:y+patch.shape[0], x:x+patch.shape[1]] = processed_patch
            mask_frame[y:y+patch.shape[0], x:x+patch.shape[1]] = mask_patch
    
    #print("Frame Completed")
    return output_frame, mask_frame


#Given inputDir process numMP4 present in folder and output degraded images to outputDir
#numMP4 = 1000 for dataset provided
def genTrainingData(input_dir, output_dir, num_MP4):
    #Ensure outputDir Exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    #Iterate over numMP4
    for i in range(1, (num_MP4 + 1)):
        #Update input and output path for current mp4
        video_path = os.path.join(input_dir, f"{i}.mp4")
        video_output_dir = os.path.join(output_dir, str(i))
        os.makedirs(video_output_dir, exist_ok=True)

        cap = cv2.VideoCapture(video_path)

        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                #print("ERROR")
                break  # Stop if the video ends
    
            # Convert the frame from BGR (used by OpenCV) to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            #process frame
            processed_frame, mask_frame = processFrame(frame_rgb, 256)#convert to patches, process, reformat, return

            frame_file_path = os.path.join(video_output_dir, f"frame_{frame_count}.npy")
            np.save(frame_file_path, processed_frame)

            mask_file_path = os.path.join(video_output_dir, f"mask_{frame_count}.npy")
            np.save(mask_file_path, mask_frame)

            if (frame_count == 5):
                plt.figure(figsize=(6, 6))
                plt.imshow(processed_frame.astype(np.uint8))
                plt.title("Blotchy")
                plt.axis('off')  # Hide axis for image

                plt.tight_layout()
                plt.show()
            
            if (frame_count == 5):
                plt.figure(figsize=(6, 6))
                plt.imshow(mask_frame.astype(np.uint8))
                plt.title("Binary")
                plt.axis('off')  # Hide axis for image

                plt.tight_layout()
                plt.show()


            frame_count += 1


        cap.release()  # Release the video capture object
        print(f"Processed video {i}.mp4 with {frame_count} frames")
        #print(f"Processed {frame_count} frames.")

    print("All videos processed sucessfully... I hope")

input_dir = "/home/brian/Desktop/MAI Project/degrade_vid_pres/CLEAN_VIDEO/FHD/"
output_dir = "/home/brian/Desktop/MAI Project/degrade_vid_pres/patch_testing/"

genTrainingData(input_dir, output_dir, 1)