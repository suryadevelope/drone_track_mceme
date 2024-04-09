
import os
import argparse
import cv2
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
import json

#jetson joystick
import py_qmc5883l
import pygame
import socket



import json
import os
#any socket stream
import socket
import asyncio

sensor = py_qmc5883l.QMC5883L(i2c_bus=7)
sensor.declination = 10

compassdata = 0
slavecompassdata = 0


UDP_IP = "255.255.255.255"  # Broadcast address
UDP_PORT = 5005
BUFFER_SIZE = 1024

async def send_receive():
    global slavecompassdata
    sender_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sender_socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    
    receiver_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    receiver_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    receiver_socket.bind((UDP_IP, UDP_PORT))

    print("Sender and Receiver started...")

    while True:
        # Sending random number
        random_number = "LMG2 "+str(sensor.get_bearing())
        sender_socket.sendto(random_number.encode(), (UDP_IP, UDP_PORT))
        
        # Receiving data
        data, addr = receiver_socket.recvfrom(BUFFER_SIZE)
        decoded_message = data.decode()
        if decoded_message.startswith("LMG"):
            parts = decoded_message.split(" ")
            if len(parts) >= 2:
                try:
                    first_number = int(parts[0].replace("LMG",""))
                    second_number = float(parts[1])
                    if(first_number==1):
                        # print("First Number:", first_number)
                        # print("Second Number:", second_number)
                        slavecompassdata = second_number
                except ValueError:
                    print("Invalid numbers received")
            else:
                print("Invalid message format")
        
        await asyncio.sleep(0.1)  # Adjust this for your desired rate

async def main():
    await send_receive()


def run_asyncio_main():
    asyncio.run(main())

asyncio_thread = Thread(target=run_asyncio_main)
asyncio_thread.start()


# Clear the log file
with open('app.log', 'w'):
    pass
# Configure logging
logging.basicConfig(filename='app.log', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

objectdetectstate = 0
# Log messages
logging.info('Starting joystick script execution...')
def joystickm():
    global objectdetectstate
    # Define the IP addresses and ports for each Stream
    # Stream_1_IP = "192.168.0.7"
    # Stream_1_PORT = 20108
    Stream_2_IP = "192.168.0.9"
    Stream_2_PORT = 20108


    # Initialize the pygame library and joysticks
    pygame.init()
    pygame.joystick.init()


    # Initialize the two UDP sockets
    # Stream1_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    Stream2_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)


    # Define a function to map joystick input to movement values
    def map_input_to_movement(value):
        # Map the input value to a movement value between -255 and 255
        movement = int(value * 255)

        # Ensure the movement value is within the valid range
        if movement > 255:
            movement = 255
        elif movement < -255:
            movement = -255

        return movement

    # Define a function to send UDP packets to the specified destination
    def send_udp_packet(socket, data, ip, port):
        try:
            socket.sendto(data.encode(), (ip, port))
        except Exception as e:
            print(f"Error sending UDP packet: {e}")

    while True :
        # Loop through each joystick
        joystickcount = pygame.joystick.get_count()
        if(joystickcount<=0):
            print(" restarting main.py")
            time.sleep(1)
            subprocess.run([sys.executable,"main.py"])
            # Your Python code here
            logging.info('restarting main.py',str(e))
            sys.exit()
        for joystick_id in range(joystickcount):
            joystick = pygame.joystick.Joystick(joystick_id)
            joystick.init()

            # Get the name of the joystick
            joystick_name = joystick.get_name()


            # Handle pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()

            # Read the joystick input
            x_axis = joystick.get_axis(0)
            y_axis = joystick.get_axis(1)
            clicked = joystick.get_button(0)


            # Map the joystick input to movement values
            x_movement = map_input_to_movement(x_axis)
            y_movement = map_input_to_movement(y_axis)

            # Determine the direction based on the joystick input

            # Determine which trigger button is pressed
            if clicked == 1:
                trigger = 2
            else:
                trigger = 3

                            
            if y_movement > 0:
                speed = abs(y_movement)
                direction = "5"
            elif y_movement < 0:
                speed = abs(y_movement)
                direction = "8"
            else:
                speed = 0
                direction = "117"

            if(speed<=20):
                speed=0
                direction = "117"

            # Construct the UDP packet data and send it
            data = "@{},{},{},{},{}".format(speed, direction, trigger,compassdata,slavecompassdata)


            if(direction == "117"):
                if(objectdetectstate == 0):
                    send_udp_packet(Stream2_socket, data, Stream_2_IP, Stream_2_PORT)
            else:
                send_udp_packet(Stream2_socket, data, Stream_2_IP, Stream_2_PORT)


            if x_movement > 0:
                speed = abs(x_movement)
                direction = "6"
            elif x_movement < 0:
                speed = abs(x_movement)
                direction = "4"
            else:
                speed = 0
                direction = "117"
            
            if(speed<=20):
                speed=0
                direction = "117"

            # Construct the UDP packet data and send it
            data = "@{},{},{},{},{}".format(speed, direction, trigger,compassdata,slavecompassdata)
            
            # print("2",data,objectdetectstate,objectdetectstate == 0)
        
            if(direction == "117"):
                if(objectdetectstate == 0):
                   
                    send_udp_packet(Stream2_socket, data, Stream_2_IP, Stream_2_PORT)
            else:
                send_udp_packet(Stream2_socket, data, Stream_2_IP, Stream_2_PORT)
            # logging.info(data)
            

            # Wait for a short amount of time to avoid overloading the CPU
            pygame.time.wait(10)
threadj = Thread(target=joystickm)
threadj.start()


logging.info('Starting main script execution...')

# Log messages


print("main.py started")

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
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

try:

    server_socket.bind((SERVER_IP, SERVER_PORT))
    print("Server is listening on {}:{}".format(SERVER_IP, SERVER_PORT))
    logging.info("Server is listening on {}:{}".format(SERVER_IP, SERVER_PORT))

    server_socket.listen(5)
except Exception as e :
    print("already has port")
    print(" restarting main.py")
    time.sleep(1)
    subprocess.run([sys.executable,"main.py"])
    # Your Python code here
    logging.info('already has port restarting main.py')
    sys.exit()
# Send the size of the serialized frame


# Define VideoStream class to handle streaming of video from webcam in separate processing thread
class VideoStream:
    """Camera object that controls video streaming from the webcam"""
    def __init__(self, resolution=(640,480), framerate=30):
        self.stream = cv2.VideoCapture(0)
        # self.stream = cv2.VideoCapture("rtsp://admin:Admin123@192.168.0.65:554/preview.asp")
        ret = self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        ret = self.stream.set(3, resolution[0])
        ret = self.stream.set(4, resolution[1])
        self.grabbed, self.frame = self.stream.read()
        self.stopped = False

    def start(self):
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        while True:
            if self.stopped:
                self.stream.release()
                return
            self.grabbed, self.frame = self.stream.read()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True

# Define and parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--modeldir', help='Folder the .tflite file is located in', default="./model")
parser.add_argument('--graph', help='Name of the .tflite file, if different than detect.tflite', default='detect.tflite')
parser.add_argument('--labels', help='Name of the labelmap file, if different than labelmap.txt', default='labelmap.txt')
parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects', default=0.98)
parser.add_argument('--resolution', help='Desired webcam resolution in WxH. If the webcam does not support the resolution entered, errors may occur.', default='1280x720')
parser.add_argument('--edgetpu', help='Use Coral Edge TPU Accelerator to speed up detection', action='store_true')

args = parser.parse_args()

MODEL_NAME = args.modeldir
GRAPH_NAME = args.graph
LABELMAP_NAME = args.labels
min_conf_threshold = float(args.threshold)
resW, resH = args.resolution.split('x')
imW, imH = int(resW), int(resH)
use_TPU = args.edgetpu

# Import TensorFlow libraries
pkg = importlib.util.find_spec('tflite_runtime')
if pkg:
    from tflite_runtime.interpreter import Interpreter
    if use_TPU:
        from tflite_runtime.interpreter import load_delegate
else:
    from tensorflow.lite.python.interpreter import Interpreter
    if use_TPU:
        from tensorflow.lite.python.interpreter import load_delegate

# If using Edge TPU, assign filename for Edge TPU model
if use_TPU:
    if (GRAPH_NAME == 'detect.tflite'):
        GRAPH_NAME = 'edgetpu.tflite'       

# Get path to current working directory
CWD_PATH = os.getcwd()

# Path to .tflite file and label map
PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_NAME, GRAPH_NAME)
PATH_TO_LABELS = os.path.join(CWD_PATH, MODEL_NAME, LABELMAP_NAME)

# Load the label map
with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

if labels[0] == '???':
    del(labels[0])

# Load the TensorFlow Lite model
if use_TPU:
    interpreter = Interpreter(model_path=PATH_TO_CKPT,
                              experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
else:
    interpreter = Interpreter(model_path=PATH_TO_CKPT)

interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

floating_model = (input_details[0]['dtype'] == np.float32)
input_mean = 127.5
input_std = 127.5

outname = output_details[0]['name']
if 'StatefulPartitionedCall' in outname:
    boxes_idx, classes_idx, scores_idx = 1, 3, 0
else:
    boxes_idx, classes_idx, scores_idx = 0, 1, 2

# Initialize video stream
videostream = VideoStream(resolution=(imW, imH), framerate=30).start()
time.sleep(1)


while not videostream.stream.isOpened():
    print("waiting for camera")
    logging.info('waiting for camera')
    videostream.stop()
    print(" restarting main.py")
    time.sleep(1)
    logging.info('restarting main.py camera issue')

    subprocess.run([sys.executable,"main.py"])
    sys.exit()

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


# Initialize OpenCV tracker
tracker = cv2.TrackerCSRT_create()  # You can change the tracker type if needed

# Initialize variables for object tracking
bbox = None
line_counter = 0
max_line_counter = 10
tracking_active = False
cotracker = None


try:
    logging.info('waiting for connection')
    conn, addr = server_socket.accept()

except Exception as e:
    print(" restarting main.py")
    time.sleep(1)
    logging.info('restarting main.py socket issue',e)

    subprocess.run([sys.executable,"main.py"])
    sys.exit()


while True:

    t1 = cv2.getTickCount()
    while not videostream.stream.isOpened():
        print("waiting for camera")
        logging.info('waiting for camera')
        videostream.stop()
        print(" restarting main.py")
        time.sleep(1)
        logging.info('restarting main.py camera issue')

        subprocess.run([sys.executable,"main.py"])
        sys.exit()
    frame1 = videostream.read()

    frame = frame1.copy()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (width, height))
    input_data = np.expand_dims(frame_resized, axis=0)
   

    if floating_model:
        input_data = (np.float32(input_data) - input_mean) / input_std

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    boxes = interpreter.get_tensor(output_details[boxes_idx]['index'])[0]
    classes = interpreter.get_tensor(output_details[classes_idx]['index'])[0]
    scores = interpreter.get_tensor(output_details[scores_idx]['index'])[0]

    # Reset tracking if object is lost
    i=0
    if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):
            bbox = None
            ymin1 = int(max(1, (boxes[i][0] * imH)))
            xmin1 = int(max(1, (boxes[i][1] * imW)))
            ymax1 = int(min(imH, (boxes[i][2] * imH)))
            xmax1 = int(min(imW, (boxes[i][3] * imW)))
            bbox = (xmin1, ymin1, xmax1 - xmin1, ymax1 - ymin1)
            cv2.rectangle(frame, (xmin1,ymin1), (xmax1,ymax1), (10, 255, 0), 2)

            # Draw label
            object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index
            label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
            label_ymin = max(ymin1, labelSize[1] + 10) # Make sure not to draw label too close to top of window
            cv2.rectangle(frame, (xmin1, label_ymin-labelSize[1]-10), (xmin1+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
            cv2.putText(frame, label, (xmin1, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text


            # Initialize the tracker with the bounding box
            cotracker = tracker.init(frame, bbox)
            tracking_active = True

    if tracking_active:
        # Update the tracker
        tracking_active, bbox = tracker.update(frame)

        if tracking_active:
            # Draw bounding box
            (x, y, w, h) = [int(v) for v in bbox]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            objectdetectstate = 1

            xmin1 = int(x)
            ymin1 = int(y)
            xmax1 = int(w +x)
            ymax1 = int(h+ y)

            cx = xmin1+((xmax1-xmin1)/2)
            cy = ymin1+((ymax1-ymin1)/2)

            fh, fw, fc = frame.shape

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
                direction = "116"

            

            # Construct the UDP packet data and send it
            data = "@{},{},{},{},{}".format(speed, direction, trigger,compassdata,slavecompassdata)
            print(data)

            #Send the UDP packet to the appropriate destination based on the Stream ID

            
            send_udp_packet(Stream2_socket, data, Stream_2_IP, Stream_2_PORT)
            # print('Stream', i, Stream2_socket, data, Stream_2_IP, Stream_2_PORT)

            if x > 0:
                speed = abs(x)
                direction = "6"
            elif x < 0:
                speed = abs(x)
                direction = "4"
            else:
                speed = 0
                direction = "116"


            # Construct the UDP packet data and send it
            data = "@{},{},{},{},{}".format(speed, direction, trigger,compassdata,slavecompassdata)

            #Send the UDP packet to the appropriate destination based on the Stream ID
            send_udp_packet(Stream2_socket, data, Stream_2_IP, Stream_2_PORT)
            # print('Stream', i, Stream2_socket, data, Stream_2_IP, Stream_2_PORT)
        else:
            objectdetectstate = 0
            
      
      
    cv2.putText(frame, 'FPS: {0:.2f}'.format(cv2.getTickFrequency() / (cv2.getTickCount() - t1)),
                (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
    
    frame0 = cv2.resize(frame.copy(), (900, 800))  # Adjust resolution as needed

    # Compress the frame
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 50]
    _, img_encoded = cv2.imencode('.jpg', frame0, encode_param)
    data = pickle.dumps(img_encoded)

    # Pack the serialized frame and its size
    message_size = struct.pack("L", len(data))

    # Send the size of the serialized frame
    try:
        conn.sendall(message_size)

        # Send the serialized frame
        conn.sendall(data)
        # print("send data")
    except Exception as e:
        print("send  feed error")
        print(" restarting main.py")
        time.sleep(1)
        subprocess.run([sys.executable,"main.py"])
        # Your Python code here
        logging.info(f"send feed error: {e}")
        sys.exit()

    
    cv2.imshow('Object detector', frame)

    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()
videostream.stop()
