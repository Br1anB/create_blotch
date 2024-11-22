import glob 
import os 
import random 
import cv2 
import matplotlib.pyplot as plt 
import numpy as np 
from tqdm import tqdm 

hd_path = 'E:/MAI_data/'

inter4KVideos = 'C:/Users/brian/Desktop/MAIProject/MP4/clean_video/'
listVideos = glob.glob(os.path.join(inter4KVideos, '*.mp4'))

patchSize = 128

numTrain = 900
numTest = len(listVideos) - numTrain

random.seed(0)
random.shuffle(listVideos)

trainVideos = listVideos[:numTrain]
testVideos = listVideos[numTrain:]

saveFrameCounter = 0

#Local Imports
import blotch_func

def generate_blotch_colour():
    # Generate a random number between 0 and 1
    #return 0
    r = random.random()
    # 80% chance for ranges [0-55] or [200-255]
    if r < 0.8:
        # Pick randomly between the two ranges
        if random.random() < 0.5:
            return random.randint(0, 35)
        else:
            return random.randint(230, 255)
    # 20% chance for the range [56-199]
    else:
        return random.randint(36, 219)


def advancedBlotches(image):
    #Similar to crude although edges of blotches will be softened
    #plt.style.use('dark_background')
    #fig, axs = plt.subplots(2, 2, figsize=(8, 8))

    image_array = np.array(image)

    #Generate Blotch Mask
    blotch_mask = blotch_func.generateBlotch_new(image, 1, 0.8, 30) #max num blotches 1, prob 0.8, max size 30
    blotch_mask = blotch_mask*255 #Conver to 255 range
    
    #Add some kind of blur on the Mask
    smoothed_mask_gaussian = cv2.GaussianBlur(blotch_mask, (5, 5), sigmaX=3, sigmaY=3) # 13,13

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
    blotch_image = blotch_image.astype(np.uint8)

    return blotch_image, binary_blotch_mask

# # Generate train content 
# for _video in tqdm(trainVideos):
#     cap = cv2.VideoCapture(_video)
#     frame_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     counter = 0
# 
#     _prev_frame = np.zeros(shape=(1920, 1080, 3))
#     _cur_frame = np.zeros(shape=(1920, 1080, 3))
#     _nxt_frame = np.zeros(shape=(1920, 1080, 3))
# 
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             print("Can't receive frame (stream end?). Exiting ...")
#             break
# 
#         _prev_frame = _cur_frame
#         _cur_frame = _nxt_frame
#         _nxt_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
# 
#         if (counter > 1) and (counter < frame_length - 2):
#             # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
# 
#             _w, _h = random.randint(0, 1920-patchSize-1), random.randint(0, 1080-patchSize-1)
# 
#             # Generate patches for prev,cur and nxt frames
#             _prev_ref = _prev_frame[_h:_h+patchSize, _w:_w+patchSize]
#             _cur_ref = _cur_frame[_h:_h+patchSize, _w:_w+patchSize]
#             _nxt_ref = _nxt_frame[_h:_h+patchSize, _w:_w+patchSize]
# 
#             # _deg = _ref + np.random.uniform(0, 10, _ref.shape)
#             # _deg = np.clip(_deg, 0, 255)
#             # _deg = _deg.astype(np.uint8)
# 
#             #Previous and next mask is not needed
#             _prev_deg, _prev_mask = advancedBlotches(_prev_ref)
#             _cur_deg, _cur_mask = advancedBlotches(_cur_ref)
#             _nxt_deg, _nxt_mask = advancedBlotches(_nxt_ref)
# 
#             # Save referance, degraded frame, degraded frame-1, degraded frame+1, ground truth
#             np.savez(os.path.join(hd_path, 'train', str(saveFrameCounter).zfill(8)), ref=_cur_ref, deg=_cur_deg, prev_deg=_prev_deg, nxt_deg=_nxt_deg, mask=_cur_mask)
#             saveFrameCounter += 1 
# 
#         counter += 1 
# 

saveFrameCounter = 0

# Generate test content 
for _video in tqdm(testVideos):
    cap = cv2.VideoCapture(_video)
    frame_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    counter = 0 

    _prev_frame = np.zeros(shape=(1920, 1080, 3))
    _cur_frame = np.zeros(shape=(1920, 1080, 3))
    _nxt_frame = np.zeros(shape=(1920, 1080, 3))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        _prev_frame = _cur_frame
        _cur_frame = _nxt_frame
        _nxt_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if (counter > 1) and (counter < frame_length - 2):
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            _w, _h = random.randint(0, 1920-patchSize-1), random.randint(0, 1080-patchSize-1)

            # Generate patches for prev,cur and nxt frames
            _prev_ref = _prev_frame[_h:_h+patchSize, _w:_w+patchSize]
            _cur_ref = _cur_frame[_h:_h+patchSize, _w:_w+patchSize]
            _nxt_ref = _nxt_frame[_h:_h+patchSize, _w:_w+patchSize]

            # _deg = _ref + np.random.uniform(0, 10, _ref.shape)
            # _deg = np.clip(_deg, 0, 255)
            # _deg = _deg.astype(np.uint8)

            #Previous and next mask is not needed
            _prev_deg, _prev_mask = advancedBlotches(_prev_ref)
            _cur_deg, _cur_mask = advancedBlotches(_cur_ref)
            _nxt_deg, _nxt_mask = advancedBlotches(_nxt_ref)

            # Save referance, degraded frame, degraded frame-1, degraded frame+1, ground truth
            np.savez(os.path.join(hd_path, 'test', str(saveFrameCounter).zfill(8)), ref=_cur_ref, deg=_cur_deg, prev_deg=_prev_deg, nxt_deg=_nxt_deg, mask=_cur_mask)
            saveFrameCounter += 1
            
        counter += 1 