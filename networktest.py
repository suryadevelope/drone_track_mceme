#any socket stream
import socket
import asyncio
import py_qmc5883l
sensor = py_qmc5883l.QMC5883L(i2c_bus=7)
sensor.declination = 10


UDP_IP = "255.255.255.255"  # Broadcast address
UDP_PORT = 5005
BUFFER_SIZE = 1024

async def send_receive():
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
                        print("First Number:", sensor.get_bearing())
                        print("Second Number:", second_number)
                except ValueError:
                    print("Invalid numbers received")
            else:
                print("Invalid message format")
        
        await asyncio.sleep(0.5)  # Adjust this for your desired rate

async def main():
    await send_receive()

if __name__ == "__main__":
    asyncio.run(main())
