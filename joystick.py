#jetson joystick
import pygame
import socket

import logging
# Clear the log file
with open('joystick.log', 'w'):
    pass
# Configure logging
logging.basicConfig(filename='joystick.log', level=logging.INFO)

# Log messages
logging.info('Starting joystick script execution...')

# Define the IP addresses and ports for each Stream
# Stream_1_IP = "192.168.0.7"
# Stream_1_PORT = 20108
Stream_2_IP = "192.168.0.7"
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
    for joystick_id in range(pygame.joystick.get_count()):
        joystick = pygame.joystick.Joystick(joystick_id)
        joystick.init()

        # Get the name of the joystick
        joystick_name = joystick.get_name()

        # print(f"Joystick {joystick_id}: {joystick_name}")
        # Loop through each joystick

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
            direction = "115"

        # Construct the UDP packet data and send it
        data = "@{},{},{}".format(speed, direction, trigger)
       
        if joystick_id == 0 :
            send_udp_packet(Stream2_socket, data, Stream_2_IP, Stream_2_PORT)


        if x_movement > 0:
            speed = abs(x_movement)
            direction = "6"
        elif x_movement < 0:
            speed = abs(x_movement)
            direction = "4"
        else:
            speed = 0
            direction = "115"


        # Construct the UDP packet data and send it
        data = "@{},{},{}".format(speed, direction, trigger)
        

        if joystick_id == 0:
            send_udp_packet(Stream2_socket, data, Stream_2_IP, Stream_2_PORT)

        # Wait for a short amount of time to avoid overloading the CPU
        pygame.time.wait(10)