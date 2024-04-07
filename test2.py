#jetson cam2
import argparse
import base64
import select
import cv2
import os
import numpy as np
import sys
import time
from threading import Thread
import importlib.util

import socket
import pickle
import struct
import time
import subprocess
import time
import subprocess
import logging

# Clear the log file
with open('app.log', 'w'):
    pass
# Configure logging
logging.basicConfig(filename='app.log', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Log messages


print("test2.py started")

def is_port_in_use(port):
    try:
        output = subprocess.check_output(["netstat", "-tuln"])
        lines = output.decode("utf-8").split("\n")
        for line in lines:
            parts = line.split()
            if len(parts) >= 4:
                if parts[3] == f"0.0.0.0:{port}":
                    return True  # Port is in use
        return False  # Port is not in use
    except subprocess.CalledProcessError as e:
        print(f"Error checking port {port}: {e}")
        return None  # Error occurred

def close_port_if_running(port):
    try:
        subprocess.run(["fuser", "-k", f"{port}/tcp"])
        print(f"Closed port {port}.")
    except Exception as e:
        print(f"Error closing port {port}: {e}")

port_to_check = 12351  # Change this to the port you want to check

port_status = is_port_in_use(port_to_check)
if port_status is None:
    print(f"An error occurred while checking port {port_to_check}.")
    logging.info(f"An error occurred while checking port {port_to_check}.")

elif port_status:
    print(f"Port {port_to_check} is in use.")
    logging.info(f"Port {port_to_check} is in use.")

    close_port_if_running(port_to_check)
else:
    print(f"Port {port_to_check} is not in use.")
    logging.info(f"Port {port_to_check} is not in use.")



time.sleep(2)

# Server IP address and port
SERVER_IP = '0.0.0.0'  # Use 0.0.0.0 to listen on all available interfaces
SERVER_PORT = 12351

# Create a socket object
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

try:

    server_socket.bind((SERVER_IP, SERVER_PORT))
    print("Server is listening on {}:{}".format(SERVER_IP, SERVER_PORT))
    logging.info("Server is listening on {}:{}".format(SERVER_IP, SERVER_PORT))

    server_socket.listen(5)
except Exception as e :
    print("already has port")
# Send the size of the serialized frame


logging.info('Starting main script execution...')


# Define VideoStream class to handle streaming of video from webcam in separate processing thread
# Source - Adrian Rosebrock, PyImageSearch: https://www.pyimagesearch.com/2015/12/28/increasing-raspberry-pi-fps-with-python-and-opencv/
class VideoStream:
    """Camera object that controls video streaming from the Picamera"""
    def __init__(self,resolution=(640,480),framerate=30,src=0):
        # Initialize the PiCamera and the camera image stream
        # self.stream = cv2.VideoCapture("rtsp://192.168.144.25:8554/main.264")
        self.stream = cv2.VideoCapture(src)
        ret = self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        ret = self.stream.set(3,resolution[0])
        ret = self.stream.set(4,resolution[1])
            
        # Read first frame from the stream
        (self.grabbed, self.frame) = self.stream.read()

	# Variable to control when the camera is stopped
        self.stopped = False

    def start(self):
	# Start the thread that reads frames from the video stream
        Thread(target=self.update,args=()).start()
        return self

    def update(self):
        # Keep looping indefinitely until the thread is stopped
        while True:
            # If the camera is stopped, stop the thread
            if self.stopped:
                # Close camera resources
                self.stream.release()
                return

            # Otherwise, grab the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
	# Return the most recent frame
        return self.frame

    def stop(self):
	# Indicate that the camera and thread should be stopped
        self.stopped = True



# Define and parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--modeldir', help='Folder the .tflite file is located in',
                    default="model")
parser.add_argument('--graph', help='Name of the .tflite file, if different than detect.tflite',
                    default='detect.tflite')
parser.add_argument('--labels', help='Name of the labelmap file, if different than labelmap.txt',
                    default='labelmap.txt')
parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects',
                    default=0.99)
parser.add_argument('--resolution', help='Desired webcam resolution in WxH. If the webcam does not support the resolution entered, errors may occur.',
                    default='1280x720')
parser.add_argument('--edgetpu', help='Use Coral Edge TPU Accelerator to speed up detection',
                    action='store_true')

args = parser.parse_args()



# from screeninfo import get_monitors

# def get_screen_resolution():
#     monitors = get_monitors()
#     if monitors:
#         monitor = monitors[0]  # Assuming there's at least one monitor
#         return monitor.width, monitor.height
#     else:
#         return None

# # Example usage:
# screen_width, screen_height = get_screen_resolution()
# if screen_width and screen_height:
#     print("Screen width:", screen_width)
#     print("Screen height:", screen_height)
# else:
#     print("Unable to retrieve screen resolution.")


MODEL_NAME = args.modeldir
GRAPH_NAME = args.graph
LABELMAP_NAME = args.labels
min_conf_threshold = float(args.threshold)
resW, resH = args.resolution.split('x')
imW, imH = int(resW), int(resH)

use_TPU = args.edgetpu


# Import TensorFlow libraries
# If tflite_runtime is installed, import interpreter from tflite_runtime, else import from regular tensorflow
# If using Coral Edge TPU, import the load_delegate library
pkg = importlib.util.find_spec('tflite_runtime')
try:
    if pkg:
        from tflite_runtime.interpreter import Interpreter
        if use_TPU:
            from tflite_runtime.interpreter import load_delegate
    else:
        from tensorflow.lite.python.interpreter import Interpreter
        if use_TPU:
            from tensorflow.lite.python.interpreter import load_delegate
except Exception as e:

    logging.info('tpu exception ..',e)
logging.info('Starting main script execution...')

# If using Edge TPU, assign filename for Edge TPU model
if use_TPU:
    # If user has specified the name of the .tflite file, use that name, otherwise use default 'edgetpu.tflite'
    if (GRAPH_NAME == 'detect.tflite'):
        GRAPH_NAME = 'edgetpu.tflite'       

# Get path to current working directory
CWD_PATH = os.getcwd()

# Path to .tflite file, which contains the model that is used for object detection
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,GRAPH_NAME)

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,MODEL_NAME,LABELMAP_NAME)

# Load the label map
with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Have to do a weird fix for label map if using the COCO "starter model" from
# https://www.tensorflow.org/lite/models/object_detection/overview
# First label is '???', which has to be removed.
if labels[0] == '???':
    del(labels[0])

# Load the Tensorflow Lite model.
# If using Edge TPU, use special load_delegate argument
if use_TPU:
    interpreter1 = Interpreter(model_path=PATH_TO_CKPT,
                              experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
    print(PATH_TO_CKPT)
else:
   interpreter1 = Interpreter(model_path=PATH_TO_CKPT)

interpreter1.allocate_tensors()

input_details1 = interpreter1.get_input_details()

output_details1 = interpreter1.get_output_details()


height1 = input_details1[0]['shape'][1]
width1 = input_details1[0]['shape'][2]

floating21_model = (input_details1[0]['dtype'] == np.float32)
input_mean = 127.5
input_std = 127.5

# Check output layer name to determine if this model was created with TF2 or TF1,
# because outputs are ordered differently for TF2 and TF1 models
outname = output_details1[0]['name']

if ('StatefulPartitionedCall' in outname): # This is a TF2 model
    boxes_idx, classes_idx, scores_idx = 1, 3, 0
else: # This is a TF1 model
    boxes_idx, classes_idx, scores_idx = 0, 1, 2

# Initialize frame rate calculation
frame_rate_calc = 1
freq = cv2.getTickFrequency()

# Initialize video stream



# Define the IP addresses and ports for each Stream
Stream_1_IP = "192.168.0.7"
Stream_1_PORT = 20108
Stream_2_IP = "192.168.0.9"
Stream_2_PORT = 20108


# Initialize the two UDP sockets
Stream1_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
Stream2_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)


# Define a function to send UDP packets to the specified destination
def send_udp_packet(socket, data, ip, port):
    try:
        socket.sendto(data.encode(), (ip, port))
    except Exception as e:
        print(f"Error sending UDP packet: {e}")


def map_input_to_movement(value):
    # Map the input value to a movement value between -255 and 255
    movement = int(value * 255 / 600)

    # Ensure the movement value is within the valid range
    if movement > 255:
        movement = 255
    elif movement < -255:
        movement = -255

    return movement


# limit the number of cpus used by high performance libraries
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"



data = "@{},{},{}".format(0, 10, 3)


counter = 0
direction = "6"
vertical_counter = 0
vertical_direction = '8'


try:
    logging.info('waiting for connection')
    conn, addr = server_socket.accept()

except Exception as e:
    print(" restarting test2.py")
    time.sleep(1)
    logging.info('restarting test2.py socket issue',e)

    subprocess.run([sys.executable,"test2.py"])
    sys.exit()

tracker = cv2.TrackerKCF_create()
logging.info('starting camera')

# videostream1 = VideoStream(resolution=(imW,imH),framerate=30,src="rtsp://admin:Admin123@192.168.0.65:554/preview.asp").start()
videostream1 = VideoStream(resolution=(imW,imH),framerate=30,src=0).start()
time.sleep(1)


while not videostream1.stream.isOpened():
    print("waiting for camera")
    logging.info('waiting for camera')
    videostream1.stop()
    print(" restarting test2.py")
    time.sleep(1)
    logging.info('restarting test2.py camera issue')

    subprocess.run([sys.executable,"test2.py"])
    sys.exit()

print("camera started")
logging.info('camera started')

# logging.info('connection', addr)

# Start capturing video from the camera

server_socket.setblocking(False)
bbox = None  # Initialize bounding box variable

logging.info('ml loop starting')

status = True
#for frame1 in camera.capture_continuous(rawCapture, format="bgr",use_video_port=True):
while True:
    try:

        # Start timer (for calculating frame rate)
        if not videostream1.stream.isOpened():
            print("waiting for camera")
        t1 = cv2.getTickCount()
        

        frame2= videostream1.read()


        if frame2 is None:
            print("Failed to capture frame from the video stream 2")
            

        frame21 = frame2.copy()

        frame21_rgb = cv2.cvtColor(frame21, cv2.COLOR_BGR2RGB)

        frame21_resized = cv2.resize(frame21_rgb, (width1, height1))

        input21_data = np.expand_dims(frame21_resized, axis=0)

    
        # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
        if floating21_model:
            input21_data = (np.float32(input21_data) - input_mean) / input_std

        interpreter1.set_tensor(input_details1[0]['index'],input21_data)
        interpreter1.invoke()

    
    # Retrieve detection results
        boxes1 = interpreter1.get_tensor(output_details1[boxes_idx]['index'])[0] # Bounding box coordinates of detected objects
        classes1 = interpreter1.get_tensor(output_details1[classes_idx]['index'])[0] # Class index of detected objects
        scores1 = interpreter1.get_tensor(output_details1[scores_idx]['index'])[0] # Confidence of detected objects


    # Loop over all detections and draw detection box if confidence is above minimum threshold
        for i in range(len(scores1)):
            if ((scores1[i] > min_conf_threshold) and (scores1[i] <= 1.0)):

                # Get bounding box coordinates and draw box
                # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
            

                ymin1 = int(max(1, (boxes1[i][0] * imH)))
                xmin1 = int(max(1, (boxes1[i][1] * imW)))
                ymax1 = int(min(imH, (boxes1[i][2] * imH)))
                xmax1 = int(min(imW, (boxes1[i][3] * imW)))

                top_left = (xmin1, ymin1)
                bottom_right = (xmax1, ymax1)

                # Calculate the width and height of the rectangle
                width = bottom_right[0] - top_left[0]
                height = bottom_right[1] - top_left[1]

                # Draw the rectangle
                bbox = (xmin1, ymin1, xmax1 - xmin1, ymax1 - ymin1)
                if(status):
                    tracker.init(frame21, bbox)
                    status=False
                break

            
        if bbox is not None and status == False:
            
            ok, new_bbox = tracker.update(frame21)
            if ok:
                if len(new_bbox) == 4:  # Ensure new_bbox has four elements (x, y, width, height)
                    bbox = new_bbox
                    p1 = (int(bbox[0]), int(bbox[1]))
                    p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))

                    xmin1 = int(bbox[0])
                    ymin1 = int(bbox[1])
                    xmax1 = int(bbox[0] + bbox[2])
                    ymax1 = int(bbox[1] + bbox[3])

                    cx = xmin1+((xmax1-xmin1)/2)
                    cy = ymin1+((ymax1-ymin1)/2)

                    fh, fw, fc = frame21.shape

                    dx = cx - (fw/2)
                    dy = (fh/2) - cy

                    print('stream #', i, 'horizontal', dx, 'vertical', dy)

                    # Read the object det input
                    x_axis = dx
                    y_axis = dy
                    trigger = 3

                    # Map the joystick input to movement values
                    x = map_input_to_movement(x_axis)
                    y = map_input_to_movement(y_axis)

                    # Determine the direction based on the Stream input
                    
                    if y > 0:
                        speed = abs(y)
                        direction = "5"
                    elif y < 0:
                        speed = abs(y)
                        direction = "8"
                    else:
                        speed = 0
                        direction = "115"

                    # Construct the UDP packet data and send it
                    data = "@{},{},{}".format(speed, direction, trigger)
                    print(data)

                    #Send the UDP packet to the appropriate destination based on the Stream ID

                    
                    send_udp_packet(Stream2_socket, data, Stream_2_IP, Stream_2_PORT)
                    print('Stream', i, Stream2_socket, data, Stream_2_IP, Stream_2_PORT)

                    if x > 0:
                        speed = abs(x)
                        direction = "6"
                    elif x < 0:
                        speed = abs(x)
                        direction = "4"
                    else:
                        speed = 0
                        direction = "115"


                    # Construct the UDP packet data and send it
                    data = "@{},{},{}".format(speed, direction, trigger)

                    #Send the UDP packet to the appropriate destination based on the Stream ID
                    send_udp_packet(Stream2_socket, data, Stream_2_IP, Stream_2_PORT)
                    print('Stream', i, Stream2_socket, data, Stream_2_IP, Stream_2_PORT)


                    
                    cv2.rectangle(frame21, (xmin1,ymin1), (xmax1,ymax1), (10, 255, 0), 2)
                    # Resize the frame to a smaller resolution
                
                    
                    # Draw label
                    object_name1 = labels[int(classes1[i])] # Look up object name from "labels" array using class index
                    label = '%s: %d%%' % (object_name1, int(scores1[i]*100)) # Example: 'person: 72%'
                    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
                    label_ymin = max(ymin1, labelSize[1] + 10) # Make sure not to draw label too close to top of window
                    cv2.rectangle(frame21, (xmin1, label_ymin-labelSize[1]-10), (xmin1+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
                    cv2.putText(frame21, label, (xmin1, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text


                    cv2.rectangle(frame, p1, p2, (0, 0, 255), 2)
                else:
                    print("Invalid bounding box format:", new_bbox)
            else:
                print("Tracker lost the object")
                tracker = cv2.TrackerKCF_create()
                status=True    


        cv2.putText(frame21,'FPS: {0:.2f}'.format(frame_rate_calc),(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)

        # All the results have been drawn on the frame, so it's time to display it.
       
        frame = cv2.resize(frame, (640, 480))  # Example dimensions

        # Encode frame (adjust compression parameters if desired)
        ret, encoded_frame = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])

        # Convert encoded frame to byte array
        data = bytearray(encoded_frame)

        # Pack data length (4 bytes) and send
        data_length = struct.pack('I', len(data))
        conn.sendall(data_length)

        # Send the frame data
        if data:
            conn.sendall(data)
        cv2.imshow("q",frame)
        print("sent frame")
        logging.info('sent frame to pc')


        # Calculate framerate
        t2 = cv2.getTickCount()
        time1 = (t2-t1)/freq
        frame_rate_calc= 1/time1

        # Press 'q' to quit
        if cv2.waitKey(1) == ord('q'):
            break
        # time.sleep(0.1)
    except Exception as e:
        print(" restarting test2.py")
        time.sleep(1)
        subprocess.run([sys.executable,"test2.py"])
        # Your Python code here
        logging.info('restarting test2.py',str(e))
        sys.exit()

# Clean up
cv2.destroyAllWindows()
videostream1.stop()
conn.close()
# Your Python code here
logging.info('Script execution completed.')
