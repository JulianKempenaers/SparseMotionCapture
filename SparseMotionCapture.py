"""
04/03/2025 Julian Kempenaers
This code captures frames at 5fps, identifies bees and only stores 
those pixels in a zipped file format (npz). Use the NpzToMp4 code to 
view the videos.

instructions:
1. press 'run' to start recording
2. In the window that pops up, write s and hit enter to stop recording
"""
# Useful CLI commands for debugging:
# rpicam-hello --qt-preview -t 0
# Imports
import time
import os
from pprint import *
from picamera2 import Picamera2
import imageio
import numpy as np
from multiprocessing import Process
from multiprocessing import Manager
import traceback
import datetime
import scipy.sparse
from scipy.sparse import coo_matrix, kron
import scipy.ndimage
import cv2
import signal #for listening to exit signal
import sys #for listening to exit signal
import select
import skimage.measure #sudo apt-get install python3-skimage 

# Settings
WIDTH = 4056
HEIGHT = 3040
SHUTTERUS = 1000
GAIN = 0
STORECOMPRESSED = True
STORE1FRAMEONLY = True # useful for debugging exposure settings (set to False for video collection)
SUBSAMPLEFRAMES = 1 if STORE1FRAMEONLY else 4 # Can go as low as 1/2 frames, but here using 1/4 to reduce data/load

#added by JK
#determine directory to save the movie chunks to, based on date/time + unique ID e.g. MAC address or IP address of this raspberry Pi. #FIXME
Arena_bounds = 'picam1'
SAVEDIR = f'/media/admin/EXTERNAL_USB/PICAMERA/Sparse_matrices/{Arena_bounds}'
# Ensure the output directory exists
os.makedirs(SAVEDIR, exist_ok=True)
capture_threshold, queue_threshold, resume_threshold = 160, 150, 130 #These MUST descend!
batch_size=100
#Julian's class------------------------------------------
class SparseMovieWriter:
  def __init__(self, arena_bounds, saveDirectory, queue_threshold, batch_size):
      print("initialising SparseMovieWriter...")
      self.queue_threshold = queue_threshold
      self.batch_size = batch_size
      self.arena_bounds = arena_bounds
      self.frame_queue = Manager().Queue()  # Use thread-safe queue
      self.foldertime = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
      self.foldername = f"{saveDirectory}/{self.arena_bounds}_{self.foldertime}"
      self.save_n_index = 1
      os.makedirs(self.foldername, exist_ok=True)
      self._setup_variables()
      self.buffer_size=12 #this is how big of an area around the bee gets saved. 
        
  def _setup_variables(self):
      print("defining setup variables...")
      if self.arena_bounds == 'arena2021':
          self.downsamplefactor = 2
          self.perc_diff = 99.9
          self.num_samples = 50
          self.erosion_value = 1
          self.dilation_value = 10
          image_shape = (1178, 1268)
          arena_mask = np.zeros(image_shape, dtype=np.uint8)
          cv2.circle(arena_mask, (624, 560), 445, 1, thickness=-1)
      else: #FIXME eventually each picam will have its own id 
          self.downsamplefactor = 4 
          self.perc_diff = 80
          self.abs_threshold = 5
          self.num_samples = 50 
          self.erosion_value = 2
          self.dilation_value = 10
          image_shape = (3040, 4056)
          arena_mask = np.ones(image_shape, dtype=np.uint8)
      
      self.arena_mask = arena_mask.astype(bool)
      self.arena_mask_downsized = self.arena_mask[::self.downsamplefactor, ::self.downsamplefactor]
  
  def process_frames(self, frame_queue):
      self.batch_frame_ids, self.batch_timestamps, self.batch_bsr_matrices, self.processing_time = [], [], [], []

      print("Starting frame processing...")  # Add to confirm processing start
      
      while True:#keep running the code unless interrupted 
          try:#extract data from the frame.
              frame_id, timestamp, image = frame_queue.get(timeout=3)  #wait max 3 sec for a frame. queue.get removes ONE frame from the queue at a time
              if frame_id % 50 == 0 and frame_id < 100 or frame_id == 0: 
                  print(f"Processing frame {frame_id}...")  # Track which frame is being processed
          except Exception:
              break #exit the 'while True' loop, essentially ending the processing entirely, moving onto saving 
              
          processing_start_time = time.time()

          #SKIP processing for KEY frames
          #TODO implement another way to trigger key frame acquisition if a large proportion of the frame is new. this will keep storage size lower and also notices lighting changes (e.g. green light turning on)
          if not self.batch_frame_ids: #the first frame in every saved batch will be the key frame. because when a batch is saved, batch_frame_ids gets reset to []
              #print("new Key frame...")
              key_frame = image #update key frame
              bsr_matrix = scipy.sparse.bsr_matrix(coo_matrix(image.astype(np.uint8)))
              
          else: #DO carry out processing on non-key frames
              #step 1: calculate pixel difference
              diff = cv2.absdiff(key_frame[::self.downsamplefactor, ::self.downsamplefactor], image[::self.downsamplefactor, ::self.downsamplefactor])
              #FIXME i think this is the frame where you should check how much of the screen is different now and if so, enter an if statement that saves current batch, clears the batch and then stores this new frame as key frame, skips processing loop but does get added to batch etc. . 
              # ROI MASK step 2.1: ROI identification & noise elimination through thresholding, erosion and dilation
              #---thresholding
              ROI_mask = diff > np.percentile(diff[::10, ::10], self.perc_diff) #create a boolean matrix where only a difference of more than a certain percentile is assigned 'true' difference.
              ROI_mask *= self.arena_mask_downsized #focus only on relevant regions (i.e. inside arena parameters)
              ROI_mask = scipy.ndimage.binary_erosion(ROI_mask, iterations=self.erosion_value) #erosion to eliminate small areas of noise
              
              #ROI mask step 2.2: add rectangular bounding boxes around the ROIs 
              #---to ensure ALL parts of the bee are definitely included, even if not all pixels were identified during thresholding  
              labeled_matrix = skimage.measure.label(ROI_mask) #labels orthogonally or diagonally connected pixels as 'regions' and labels them with unique id numbers
              
              #get bounding box slices for each labeled region
              slices = scipy.ndimage.find_objects(labeled_matrix) #finds slices (bounding boxes) for each labeled region.
              #each slide is a tuple with two slices (slice(start_row, end_row), slice(start_col, end_col)), definint a ROI's boinding box

              # Check if there are any valid slices (i.e., detected ROIs)
              if slices and any(s is not None for s in slices):  # ensure there are non-None slices

                  #convert slices to an array with rows of bounding box coordinates. : [min_row, max_row, min_col, max_col]
                  bbox_array = np.array([
                      [s[0].start, s[0].stop, s[1].start, s[1].stop] for s in slices if s is not None
                  ])
                  
                  #expand bounding box coordinates by fixed buffer size (this is more efficient than dilation) :
                  
                  #---note: np.clip() avoids going out of image bounds 
                  bbox_array[:, 0] = np.clip(bbox_array[:, 0] - self.buffer_size, 0, ROI_mask.shape[0])  # min row
                  bbox_array[:, 1] = np.clip(bbox_array[:, 1] + self.buffer_size, 0, ROI_mask.shape[0])  # max_row
                  bbox_array[:, 2] = np.clip(bbox_array[:, 2] - self.buffer_size, 0, ROI_mask.shape[1])  # min_col
                  bbox_array[:, 3] = np.clip(bbox_array[:, 3] + self.buffer_size, 0, ROI_mask.shape[1])  # max_col
                  
                  #for each bounding box, generate all pixel coordinates inside the bounding box. 
                  row_list = []
                  col_list = []
                  
                  for r0, r1, c0, c1 in bbox_array:
                      #meshgrid creates a grid of coordinates from r0 to r1 and c0 to c1
                      #---r and c are 2D arrays (e.g. ([0, 0], [1, 1])) that contain row&col indices of each pixel iside a bounding box.
                      r, c = np.meshgrid(np.arange(r0, r1), np.arange(c0, c1), indexing="ij")   
                      #---flatten() converts the 2D array into a 1D array (e.g. [0, 0, 1, 1])
                      #---and these 1D arrays per bounding box are appended into the growing row_list of bounding box row coordinates
                      row_list.append(r.flatten()) 
                      col_list.append(c.flatten())
                  
                  # Flatten everything into 1D arrays
                  rows = np.concatenate(row_list) #concatenate each bounding box 1D array in the list into single 1D arrays for the whole mask. this is essential for sparse matrix creation
                  cols = np.concatenate(col_list) 
                  
                  #create sparse csr matrix directly from bounding box coordinates csr matrix is required for kronecker upscaling further down
                  data = np.ones(len(rows), dtype=bool) # coo matrix will contain '1' at rows and cols coordinates
                  sparse_mask = coo_matrix((data, (rows, cols)), shape=ROI_mask.shape).tocsr()
                  
                  #--------------------------------------------------------------------------------------------------------
                  #step 2.3 upscale the ROI mask
                  upscale_kernel = np.ones((self.downsamplefactor, self.downsamplefactor), dtype=int)
                  
                  # Apply Kronecker product for efficient upscaling
                  #---for each True value in sparse_mask, a downsamplefactor x downsamplefactor block of true values is placed in the upscaled mask
                  upscaled_sparse_mask = kron(sparse_mask, upscale_kernel, format="csr")
  
                  #--------------------------------------------------------------------------------------------------------
                  #step 3: use upscaled mask to extract relevant pixel values from this frame. THEN store coordinates & corresponding values as a bsr matrix
                  #---Extract nonzero coordinates from full-res ROI mask
                  full_rows, full_cols = upscaled_sparse_mask.nonzero() #find indices where mask is True in upscaled mask
                  
                  # Retrieve corresponding pixel values from original image
                  values = image[full_rows, full_cols] #maps the ROI mask back onto the original image, extracting pixel values where mask = true
                  
                  # Construct COO sparse matrix
                  #---store non-zero pixels in sparse format
                  coo_sparse_matrix = coo_matrix((values, (full_rows, full_cols)), shape=image.shape)
                                  
                  #convert coo matrix into BSR format for better memory usage when storing blocks of non-zero values.
                  bsr_matrix = scipy.sparse.bsr_matrix(coo_sparse_matrix)
              else: #if no ROIs were detected
                  bsr_matrix = scipy.sparse.bsr_matrix((image.shape[0], image.shape[1]), dtype=np.uint8)  # Empty sparse matrix (all zeros)
                 
          self.batch_frame_ids.append(frame_id)
          self.batch_timestamps.append(timestamp)
          self.batch_bsr_matrices.append(bsr_matrix)
          self.processing_time.append(time.time() - processing_start_time)
          
          if len(self.batch_frame_ids) >= self.batch_size:
              self._save_batch(self.batch_frame_ids, self.batch_timestamps, self.batch_bsr_matrices)
              print(f"Saved frames {frame_id-self.batch_size+1}-{frame_id}...")
              self.batch_frame_ids, self.batch_timestamps, self.batch_bsr_matrices, self.processing_time = [], [], [], []
      
      self._save_remainder(frame_id)
      
  
  def _save_batch(self, frame_ids, timestamps, bsr_matrices):
      self.filename = f"{self.foldername}/{self.arena_bounds}_{self.foldertime}_{str(self.save_n_index).zfill(6)}_{len(frame_ids)}frames.npz" #saving it with a padded index value at the end. 
      np.savez_compressed(self.filename, frameid=np.array(frame_ids), timestamp=np.array(timestamps), bsr_matrix=np.array(bsr_matrices))
      self.save_n_index +=1
  def _save_remainder(self, frame_id):
    if self.batch_frame_ids:  # Save remaining frames
        self._save_batch(self.batch_frame_ids, self.batch_timestamps, self.batch_bsr_matrices)
        print(f"\n\nSaved remaining frames {frame_id-len(self.batch_frame_ids)+1}-{frame_id} to:  \n{self.filename}.  \n\nClose this window and press 'run' to record again")
        self.batch_frame_ids, self.batch_timestamps, self.batch_bsr_matrices, self.processing_time = [], [], [], []
    else:
      print(f"\n \nFinished processing and saving files. \n\nClose this window and press 'run' to record again")
#--------------------------------------------------------------------------
def runCameraAcquisition(queue, stdout):
  """
  Run the camera acquisition loop.
  """
  # We need to record in YUV420 to easily extract grayscale
  FORMAT = 'YUV420'
  # Record in YUV420 format
  picam2 = Picamera2()
  config = picam2.create_still_configuration({
     'format': FORMAT, 'size': (WIDTH, HEIGHT)})
  picam2.configure(config)
  picam2.set_controls({"ExposureTime": SHUTTERUS, "AnalogueGain": GAIN})
  
  # Print camera config
  pprint(picam2.camera_configuration)

  # Start acquisition
  picam2.start()
  # Wait for 2 seconds to ensure the camera has initialized
  time.sleep(2)
  # Remember frame ID
  frameID = -1
  #keep track of printed output
  queue_full_warning_printed = False
  queue_almost_full_warning_printed = False
  skipped_frame_counter = 0
  global total_frames_skipped 
  total_frames_skipped = 0
  last_stop_check_time = time.time()
  
  print("\nRECORDING IN PROGRESS. TO STOP RECORDING: type 's' and press enter \n \n")
  capture_times = []  # List to store capture times for calculating the capture rate
  #first frame capture
  yuv = picam2.capture_array("main")
  grey = yuv[:HEIGHT, :WIDTH]
  frameID +=1
  queue.put_nowait((frameID, time.time(), grey))
  last_capture_time = time.time()  # Track the time of the last capture
  
  # Main acquisition 
  while True: #keep running until stop signal is received
    if time.time() -last_stop_check_time >=3:
      if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
        user_input = sys.stdin.readline().strip().lower()
        if user_input == 's':
          print("stopping frame capture...")
          break #exit the loop
      last_stop_check_time = time.time()
    # Acquire frame and extract the greyscale channel
    yuv = picam2.capture_array("main")
    grey = yuv[:HEIGHT, :WIDTH]
    frameID += 1
    
    #---track frame capture time---------
    current_time = time.time()
    capture_times.append(current_time - last_capture_time)
    last_capture_time=current_time
    if len(capture_times)>=100:
      mean_capture_rate = 1/np.mean(capture_times)
      print(f'Mean capture rate {mean_capture_rate:.2f}fps')
      capture_times=[] #reset capture_times
    
    #----------------------------------------------------
    # Send to frame processing loop
    # If the processing queue gets too far behind, stop adding frames
    qs = queue.qsize()
    if qs < capture_threshold:  
      queue.put_nowait((frameID, time.time(), grey))
      if qs % 10 == 0 and qs > 10:
        print(f'Frame queue size: {qs}...')
      if qs>queue_threshold and not queue_almost_full_warning_printed and not queue_full_warning_printed:
        print('Warning: Frame queue ALMOST FULL...')
        queue_almost_full_warning_printed = True
      if qs<= resume_threshold and queue_full_warning_printed:
        print(f'Queue small enough to resume normal capture again, \nbut {skipped_frame_counter} frames were skipped...')
        queue_full_warning_printed = False
        queue_almost_full_warning_printed = False
        total_frames_skipped += skipped_frame_counter
        skipped_frame_counter = 0
    else: 
      if not queue_full_warning_printed:
        print('Warning: frame queue FULL! Lowering frame rate temporarily until queue decreases...')
        queue_full_warning_printed = True
      skipped_frame_counter += 1
    #
      #----------------------------------------------------
    # Print any messages on the message queue
    while not stdout.empty():
      print(stdout.get())
      
  #this gets activated when stop signal is received
  print("Camera stopped, saving remaining frames...")
  queue.put(None) #stop signal added to queue

def runCameraProcessing(workerid, queue, stdout, SAVEDIR):
    """ 
    Run the camera frame processing/saving loop.
    """
    
    saveDirectory = SAVEDIR

    #create sparse movie writer instance
    writer = SparseMovieWriter(Arena_bounds, saveDirectory, queue_threshold, batch_size)

    #wait for new frame to become available
    #stdout.put('Sub-process has started.')
    
    writer.process_frames(queue) #otherwise, run process_frames() on the queue.
        
def runCameraProcessingSafe(workerid, queue, stdout, SAVEDIR):
  """
  Processing loop, but catch any errors and return the stack trace
  to the main process.
  """
  try:
    runCameraProcessing(workerid, queue, stdout, SAVEDIR)
  except Exception as e:
    stdout.put(traceback.format_exc()) #Sends the error message to the stdout queue
    
if __name__ == "__main__":
  """
  Program entry point.
  """

  with Manager() as manager:
    # Create a queue for inter-process communication
    queue = manager.Queue() #the multiprocessing queue is used to safely share frames between the different processes
    stdout = manager.Queue()

    # Start 1 or more workers
    nworkers = 1
    processes = [Process(target=runCameraProcessingSafe, args=( #this function process frames from the queue and saves them.
      i, queue, stdout, SAVEDIR)) for i in range(nworkers)]
    for process in processes:
      process.start()

    # Run the camera loop in the main thread
    runCameraAcquisition(queue, stdout) #this function captures frames and pushes them to the queue (if queue size is <150). 
  
  if total_frames_skipped >0:
    print(f"\WARNING During recording, n{total_frames_skipped} frames were skipped (due to a backed up queue)")
 
