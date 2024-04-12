import cv2
import socket

# Initialize VideoCapture object
cap = cv2.VideoCapture(0)

# Initialize socket
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind(('0.0.0.0', 8000))  # Use appropriate IP and port
server_socket.listen(0)
connection, address = server_socket.accept()

print("connected")
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert frame to bytes
        frame_bytes = cv2.imencode('.jpg', frame)[1].tobytes()

        # Send frame size and frame data
        connection.sendall((str(len(frame_bytes))).encode().ljust(16) + frame_bytes)
except (ConnectionResetError, BrokenPipeError):
        print("Client disconnected.")
        
finally:
    connection.close()
    server_socket.close()
    cap.release()
