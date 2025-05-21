"""
27 feb 2025 Julian Kempenaers
conversion to MP4 with OVERLAY of KEY FRAME (the first frame of each npz file is a key frame.)
this is only for viewing purposes. long term storage should be done as .npz files not as mp4 files 
"""
# STEP 1:insert the name of video folder that contains npz files which you want to view 
npz_folder_name = 'picam1' #date and time of video capture
npz_file_name ='picam1_20250321_1613183bees1sponge'
#STEP2: press 'run' 

import cv2
import glob
import numpy as np
import os


filename_pattern = f'/media/admin/EXTERNAL_USB/PICAMERA/Sparse_matrices/{npz_folder_name}/{npz_file_name}/*.npz'
output_folder= '/media/admin/EXTERNAL_USB/PICAMERA/Sparse_matrices/Videos_temporary'
# Ensure the output directory exists
os.makedirs(output_folder, exist_ok=True)





files = glob.glob(filename_pattern)

# Set up the video writer once, before processing files
# We'll use the first file's frame dimensions to set the video dimensions
npz_data = np.load(files[0], allow_pickle=True)
frame_ids = npz_data['frameid']
timestamps = npz_data['timestamp']
bsr_matrices = npz_data['bsr_matrix']

# Get the dimensions from the first frame in the first .npz file
frame_width, frame_height = bsr_matrices[0].shape[::-1]  # Get dimensions from the first frame
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4 codec
out = cv2.VideoWriter(f'{output_folder}/{npz_file_name}_sparse.mp4', fourcc, 10, (frame_width, frame_height), isColor=False)

i_frames = 0

# Process each .npz file and add frames to the video
for file_n in range(len(files)):
    npz_data = np.load(files[file_n], allow_pickle=True)
    print(f'Processing file {file_n} of {len(files)}...')
    
    # Extract data from the .npz file
    frame_ids = npz_data['frameid']
    timestamps = npz_data['timestamp']
    bsr_matrices = npz_data['bsr_matrix']
    
    # Sort everything based on frame_id to ensure correct order
    sorted_indices = np.argsort(frame_ids)
    
    frame_ids = frame_ids[sorted_indices]
    timestamps = timestamps[sorted_indices]
    bsr_matrices = bsr_matrices[sorted_indices]

    
    # Convert each sparse matrix to a dense image and write to video
    for i, bsr_matrix in enumerate(bsr_matrices):
        coo = bsr_matrix.tocoo()
        frame = np.zeros((bsr_matrix.shape[0], bsr_matrix.shape[1]), dtype=np.uint8)
        frame[coo.row, coo.col] = coo.data.astype(np.uint8)
        
        if i == 0:
            key_frame = frame.copy()
            out.write(key_frame)
        else:
            #insert code here 
            key_frame_overlay = key_frame.copy()
            key_frame_overlay[frame > 0] = frame[frame > 0] #where frame>1, add frame pixel values
            out.write(key_frame_overlay)

            
        
        i_frames += 1
    
        # Print progress every 400 frames
        if i_frames % 400 == 0:
            print(f"{i_frames} frames converted so far...")

print(f"All {i_frames} frames processed and saved in the combined video.")

# Release the video writer after all frames are processed
out.release()

print(f"MP4 video saved successfully in {output_folder}!")
