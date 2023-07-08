import PySimpleGUI as sg
import numpy as np
import cv2 as cv
import os

sg.theme("LightGreen")

layout = [
    [ sg.Image(filename="", key="mask"), sg.Image(filename="", key="result") ],
    [
        sg.Input(key="Open", enable_events=True, visible=True),
        sg.FileBrowse(button_text='Open', size=(5, 1), initial_folder=os.getcwd(), tooltip="load image from a file", file_types=(("image files", "*.jpg;*.png"),), target="Open")
    ],
    [
        sg.Input(key="Save", enable_events=True, visible=True, disabled=True),
        sg.FileSaveAs(button_text='Save', size=(5, 1), initial_folder=os.getcwd(), tooltip="save image into a file", file_types=(("image files", "*.jpg;*.png"),), target="Save")
    ],
    [ sg.Button("Exit", size=(5, 1), key="Exit") ]
]

# Create the window and show it without the plot
window = sg.Window("Background remover", layout, finalize=True)

#sg.popup_no_buttons("Please wait...", non_blocking=True)
#print('loading')
# load model
modelPath = "modnet_photographic_portrait_matting_opset9.onnx"
net = cv.dnn.readNetFromONNX(modelPath)
#print('loaded')

# select CPU or GPU
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV) #CPU
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)      #CPU
#net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)    #GPU
#net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)      #GPU

def update(frame):

    # prepare blob
    blob = cv.resize(frame,(672,512), cv.INTER_AREA)
    blob = blob.astype(np.float32)
    blob /= 255
    blob = 2*blob-1
    channels = cv.split(blob)
    blob = np.array([[channels[2],channels[1],channels[0]]])

    # Sets the input to the network
    net.setInput(blob)

    # Runs the forward pass to get output of the output layers
    outs = net.forward()

    # Process the result
    mask = outs[0][0]
    mask = cv.resize(mask,(frame.shape[1],frame.shape[0]))
    mask = cv.merge([mask,mask,mask])
    result = (mask * frame + (1-mask)*np.ones_like(frame)*255).astype(np.uint8)

    return result, (mask*255).astype(np.uint8)

window.bind("<Escape>", "Exit")

result = None
while True:
    event, values = window.read(timeout=1)
    if event == "Exit" or event == sg.WIN_CLOSED:
        break
    elif event == "Open" and values["Open"] != '':
        # load image
        name = values["Open"]
        frame = cv.imread(name)
        if frame is not None:
            result, mask = update(frame)
            result_, mask_ = result, mask
            while result_.shape[0] > 400 or result_.shape[1] > 500:
                result_ = cv.resize(result_,(result_.shape[1]//2,result_.shape[0]//2))
                mask_ = cv.resize(mask_,(mask_.shape[1]//2,mask_.shape[0]//2))
            window["mask"].update(data=cv.imencode(".png", mask_)[1].tobytes())
            window["result"].update(data=cv.imencode(".png", result_)[1].tobytes())
            window["Save"].update(disabled=False)
    elif event == "Save" and values["Save"] != '':
        name = values["Save"]
        if result is not None:
            cv.imwrite(name,result)

window.close()

# deployment
# pyinstaller --noconsole bgremovergui.py 
# + copy onnx into dist
