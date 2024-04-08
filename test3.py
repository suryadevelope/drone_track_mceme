# server.py
import socket
import subprocess
import time




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

port_to_check = 9999  # Change this to the port you want to check

port_status = is_port_in_use(port_to_check)
if port_status is None:
    print(f"An error occurred while checking port {port_to_check}.")
 
elif port_status:
    print(f"Port {port_to_check} is in use.")

    close_port_if_running(port_to_check)
else:
    print(f"Port {port_to_check} is not in use.")
 

time.sleep(2)


def start_server():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(('0.0.0.0', 9999))  # Change the IP and port as needed
    server_socket.listen(5)
    print("Server listening...")

    while True:
        client_socket, client_address = server_socket.accept()
        print(f"Connection from {client_address}")
        client_socket.sendall(b"Welcome to the server!")
        client_socket.close()

if __name__ == "__main__":
    start_server()
