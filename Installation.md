In the Raspberry pi Terminal:
## 1 Clone the Repository
<pre> git clone https://github.com/JulianKempenaers/SparseMotionCapture.git 
 cd SparseMotionCapture   </pre>

## 2 Install python dependencies
<pre>pip install -r requirements.txt </pre>
  Note: scikit-image (used as skimage) is listed in requirements.txt, but it may fail to install via pip on Raspberry Pi. See the next section.

## 3 Install scikit-image via APT instead of pip:
<pre>sudo apt-get update
sudo apt-get install python3-skimage </pre>
## 4 Troubleshooting:
Picamera2 Setup (if not working via pip)
<pre>sudo apt-get install python3-picamera2
sudo apt-get install libcamera-apps
sudo raspi-config
sudo reboot
 </pre>

 Errors related to 'libcamera' or 'libcamera-dev'
 <pre>sudo apt-get install libcamera-dev libcamera-apps
</pre>
