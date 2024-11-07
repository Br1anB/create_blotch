import os
import numpy as np
import matplotlib.pyplot as plt
import cv2

def process_and_save_frames():
    # Loop over folders 1 to 100
    for folder_num in range(1, 101):
        # Create the folder path
        folder_path = f"C:/Users/brian/Desktop/MAIProject/MP4/training_data/{folder_num}/"
        
        # Read the frames from the folder
        frame_1_path = os.path.join(folder_path, "frame_0.npy")
        frame_2_path = os.path.join(folder_path, "frame_1.npy")
        frame_3_path = os.path.join(folder_path, "frame_2.npy")
        
        # Check if the frame files exist
        if os.path.exists(frame_1_path) and os.path.exists(frame_2_path) and os.path.exists(frame_3_path):
            # Load frames
            frame_1 = np.load(frame_1_path)
            frame_2 = np.load(frame_2_path)
            frame_3 = np.load(frame_3_path)

             # Convert frames to grayscale for optical flow
            gray_1 = cv2.cvtColor(frame_1, cv2.COLOR_BGR2GRAY) if frame_1.ndim == 3 else frame_1
            gray_2 = cv2.cvtColor(frame_2, cv2.COLOR_BGR2GRAY) if frame_2.ndim == 3 else frame_2
            gray_3 = cv2.cvtColor(frame_3, cv2.COLOR_BGR2GRAY) if frame_3.ndim == 3 else frame_3
            # Optical flow
            flow_1_2 = cv2.calcOpticalFlowFarneback(gray_1, gray_2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            flow_2_3 = cv2.calcOpticalFlowFarneback(gray_2, gray_3, None, 0.5, 3, 15, 3, 5, 1.2, 0)

            # Warp frame_1 and frame_3
            h, w = gray_2.shape
            grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
            
            #Calc warp maps, convert them to float32 for remap function
            map_1_to_2_x = (grid_x + flow_1_2[..., 0]).astype(np.float32)
            map_1_to_2_y = (grid_y + flow_1_2[..., 1]).astype(np.float32)
            map_2_to_3_x = (grid_x + flow_2_3[..., 0]).astype(np.float32)
            map_2_to_3_y = (grid_y + flow_2_3[..., 1]).astype(np.float32)
            
            warped_frame_1 = cv2.remap(frame_1, map_1_to_2_x, map_1_to_2_y, cv2.INTER_LINEAR)
            warped_frame_3 = cv2.remap(frame_3, map_2_to_3_x, map_2_to_3_y, cv2.INTER_LINEAR)

            # Calculate the differences
            diff_1_2 = cv2.absdiff(frame_2, warped_frame_1)  # Using absolute difference
            diff_2_3 = cv2.absdiff(warped_frame_3, frame_2)
            # Calculate the differences
            # diff_1_2 = frame_2 - frame_1
            # diff_2_3 = frame_3 - frame_2

            result = np.concatenate((diff_1_2, frame_2, diff_2_3), axis=-1)

            # plt.figure(figsize=(12, 6))
            #
            # # Show diff_1_2
            # plt.subplot(1, 3, 1)
            # plt.imshow(diff_1_2, cmap='gray')
            # plt.title('Difference (Frame 1 -> Frame 2)')
            # plt.axis('off')
            # 
            # # Show frame_2 for reference
            # plt.subplot(1, 3, 2)
            # plt.imshow(frame_2, cmap='gray')
            # plt.title('Frame 2')
            # plt.axis('off')
            # 
            # # Show diff_2_3
            # plt.subplot(1, 3, 3)
            # plt.imshow(diff_2_3, cmap='gray')
            # plt.title('Difference (Frame 2 -> Frame 3)')
            # plt.axis('off')
            #
            # plt.show()

            result_file_path = os.path.join(folder_path, "processed_frames.npy")
            np.save(result_file_path, result)
            print(f"Processed frames saved in {result_file_path}")
        else:
            print(f"Frame files missing in {folder_path}. Skipping folder.")


output_dir = "C:/Users/brian/Desktop/MAIProject/MP4/training_data/"

# Call the function
process_and_save_frames()