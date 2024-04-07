# import torch
# properties = torch.cuda.get_device_properties(torch.device('cuda'))
# print(properties['name'])

# print(properties['total_memory'])
# python
import cv2
print(cv2.cuda.getCudaEnabledDeviceCount())