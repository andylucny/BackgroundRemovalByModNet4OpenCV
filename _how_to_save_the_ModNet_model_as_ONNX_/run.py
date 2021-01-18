import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Variable
from torchsummary import summary

from src.models.modnet import MODNet

torch_transforms = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

print('Load pre-trained MODNet...')
#pretrained_ckpt = './modnet_webcam_portrait_matting.ckpt'
pretrained_ckpt = './modnet_photographic_portrait_matting.ckpt'
modnet = MODNet(backbone_pretrained=False)
modnet = nn.DataParallel(modnet)

GPU = True if torch.cuda.device_count() > 0 else False
if GPU:
    print('Use GPU...')
    modnet = modnet.cuda()
    modnet.load_state_dict(torch.load(pretrained_ckpt))
else:
    print('Use CPU...')
    modnet.load_state_dict(torch.load(pretrained_ckpt, map_location=torch.device('cpu')))

modnet.eval()

print('Init WebCam...')
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print('Start matting...')
while(True):
    _, frame0 = cap.read()
    frame_np = cv2.cvtColor(frame0, cv2.COLOR_BGR2RGB)
    frame_np = cv2.resize(frame_np, (910, 512), cv2.INTER_AREA)
    frame_np = frame_np[:, 120:792, :]
    frame_np = cv2.flip(frame_np, 1)

    frame_PIL = Image.fromarray(frame_np)
    frame_tensor = torch_transforms(frame_PIL)
    frame_tensor = frame_tensor[None, :, :, :]
    if GPU:
        frame_tensor = frame_tensor.cuda()
    print('->',frame_tensor.shape)
    
    with torch.no_grad():
        matte_tensor = modnet(frame_tensor)
    print('<-',matte_tensor.shape)
    matte_tensor = matte_tensor.repeat(1, 3, 1, 1)
    matte_np = matte_tensor[0].data.cpu().numpy().transpose(1, 2, 0)
    fg_np = matte_np * frame_np + (1 - matte_np) * np.full(frame_np.shape, 255.0)
    view_np = np.uint8(np.concatenate((frame_np, fg_np), axis=1))
    view_np = cv2.cvtColor(view_np, cv2.COLOR_RGB2BGR)

    cv2.imshow('MODNet - WebCam [Press \'Q\' To Exit]', view_np)
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break
    elif key & 0xFF == ord('s'):
        cv2.destroyAllWindows()
        #cv2.imwrite('frame.png',frame_np)
        #fixed=cv2.cvtColor(frame_np,cv2.COLOR_RGB2BGR)
        #cv2.imwrite('fixed.png',fixed)
        cv2.imwrite('frame0.png',frame0)
        
        #dummy = torch.randn(1, 3, 512, 672)
        #torch.onnx.export(modnet.module.to('cpu'), Variable(dummy), 'bgremover.onnx', verbose=True)
        
        # Input to the model
        torch_model = modnet.module
        #summary(torch_model,(3, 512, 672))
        dummy = torch.randn(1, 3, 512, 672, requires_grad=True).to('cuda')
        #with torch.no_grad():
        #    torch_out = torch_model(dummy)

        # Export the model
        torch.onnx.export(torch_model,       # model being run
                  dummy,                     # model input (or a tuple for multiple inputs)
                  "bgremover.onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=9,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  #input_names = ['input'],   # the model's input names
                  #output_names = ['conv_f'], # the model's output names
                  #dynamic_axes={'input' : {0 : 'batch_size'},    # variable lenght axes
                  #              'output' : {0 : 'batch_size'}}
        )
                                
        #torch.onnx.export(modnet.module, frame_tensor.to('cuda'), 'bgremover.onnx', verbose=True)
        break

print('Exit...')
