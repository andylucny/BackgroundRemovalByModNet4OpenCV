import numpy as np
import cv2

# load image
frame = cv2.imread('input.jpg')
blob = cv2.resize(frame,(672,512), cv2.INTER_AREA)
blob = blob.astype(np.float)
blob /= 255
blob = 2*blob-1
channels = cv2.split(blob)
blob = np.array([[channels[2],channels[1],channels[0]]])

# load model
modelPath = "modnet_photographic_portrait_matting_opset9.onnx"
#modelPath = "modnet_webcam_portrait_matting_opset9.onnx"
net = cv2.dnn.readNetFromONNX(modelPath)

# select CPU or GPU
#net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV) #CPU
#net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)      #CPU
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)    #GPU
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)      #GPU

# Sets the input to the network
net.setInput(blob)

# Runs the forward pass to get output of the output layers
outs = net.forward()

# Process the result
mask = outs[0][0]
mask = cv2.resize(mask,(frame.shape[1],frame.shape[0]))
mask = cv2.merge([mask,mask,mask])
result = (mask * frame + (1-mask)*np.ones_like(frame)*255).astype(np.uint8)

# save the result
cv2.imwrite('output.jpg',result)

