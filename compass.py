# import smbus

# # Define I2C bus number
# bus = smbus.SMBus(7)  # For Jetson Nano, the I2C bus number is usually 1

# def scan_i2c():
#     devices = []
#     for address in range(0, 128):
#         try:
#             bus.read_byte(address)
#             devices.append(hex(address))
#         except Exception as e:
#             pass
#     return devices

# try:
#     print("Scanning I2C bus for devices...")
#     i2c_devices = scan_i2c()
#     if i2c_devices:
#         print("I2C devices found:")
#         for device in i2c_devices:
#             print(device)
#     else:
#         print("No I2C devices found.")
# except Exception as e:
#     print(f"Error: {str(e)}")






import py_qmc5883l
sensor = py_qmc5883l.QMC5883L(i2c_bus=7)
sensor.declination = 10
while True:
    m = sensor.get_bearing()
    # m = sensor.get_magnet()
    print(m)