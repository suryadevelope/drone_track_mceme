# import cv2
# import socket
# import pickle
# import struct

# # Replace with your desired IP address
# HOST = '0.0.0.0'  # Standard loopback interface address (localhost)
# PORT = 65432        # Port to listen on (non-privileged ports are > 1023)

# # Open the video capture object
# cap = cv2.VideoCapture(0)

# # Check if the camera opened successfully
# if not cap.isOpened():
#     print("Error opening video capture object")
#     exit()

# # Create a socket object
# # s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# # Bind the socket to the address and port
# # s.bind((HOST, PORT))

# # Listen for incoming connections
# # s.listen(1)
# # conn, addr = s.accept()
# # print('Connected by', addr)

# while True:
#     # Capture frame-by-frame
#     ret, frame = cap.read()

#     # Check if frame is read correctly
#     if not ret:
#         print("Can't receive frame (stream end?). Exiting...")
#         break

#     # Resize frame for efficiency (optional, adjust dimensions as needed)
#     # frame = cv2.resize(frame, (640, 480))  # Example dimensions

#     # # Encode frame (adjust compression parameters if desired)
#     # ret, encoded_frame = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])

#     # # Convert encoded frame to byte array
#     # data = bytearray(encoded_frame)

#     # # Pack data length (4 bytes) and send
#     # data_length = struct.pack('I', len(data))
#     # conn.sendall(data_length)

#     # # Send the frame data
#     # if data:
#     #     conn.sendall(data)
#     cv2.imshow("q",frame)

# # When everything done, release the capture object
# cap.release()
# # s.close()
# cv2.destroyAllWindows()


import cv2
import numpy as np

# Create a VideoCapture object to capture video from the webcam (0 is usually the default webcam)
# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture("rtsp://admin:Admin123@192.168.0.64:554/preview.asp")

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the original and grayscale frames
    cv2.imshow('Original', frame)
    cv2.imshow('Grayscale', gray_frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the VideoCapture object and close all windows
cap.release()
cv2.destroyAllWindows()



