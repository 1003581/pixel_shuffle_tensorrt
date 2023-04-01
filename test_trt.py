import sys
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import time
import copy
import numpy as np
import os
import torch
import cv2


TRT_LOGGER = trt.Logger(trt.Logger.INFO)
a=(int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
device='cuda:0'

engine_file='./pixel_shuffle.engine'
usinghalf=True
batch_size=1
onnx_path = './pixel_shuffle.onnx'

def GiB(val):
    return val * 1 << 30

def allocate_buffers(engine, is_explicit_batch=False, input_shape=None):
    inputs = []
    outputs = []
    bindings = []
    class HostDeviceMem(object):
        def __init__(self, host_mem, device_mem):
            self.host = host_mem
            self.device = device_mem

        def __str__(self):
            return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

        def __repr__(self):
            return self.__str__()
    for binding in engine:
        
        dims = engine.get_binding_shape(binding)
        print("*******"+str(dims)+" dims[-1] "+str(dims[-1]))

        if dims[-1] == -1:
            assert(input_shape is not None)
            dims[-2],dims[-1] = input_shape
        size = trt.volume(dims) * engine.max_batch_siz
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        bindings.append(int(device_mem))
        if engine.binding_is_input(binding):  
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings
def preprocess_image(imagepath):
    origin_img = cv2.imread(imagepath)  # BGR
    origin_height=origin_img.shape[0]
    origin_width=origin_img.shape[1]
    new_height=512
    new_width=512
    pad_img=cv2.resize(origin_img,(new_height,new_width))
    pad_img = pad_img[:, :, ::-1].transpose(2, 0, 1)
    pad_img = pad_img.astype(np.float32)
    pad_img /= 255.0
    pad_img = np.ascontiguousarray(pad_img)
    pad_img = np.expand_dims(pad_img, axis=0)
    return pad_img,(new_height,new_width),(origin_height,origin_width)
def profile_trt(engine, imagepath,batch_size):
    assert(engine is not None)  
    
    input_image,input_shape=preprocess_image(imagepath)

    segment_inputs, segment_outputs, segment_bindings = allocate_buffers(engine, True,input_shape)
    print(segment_inputs)
    stream = cuda.Stream()    
    with engine.create_execution_context() as context:
        context.active_optimization_profile = 0
        origin_inputshape=context.get_binding_shape(0)
        feat_inputshape = context.get_binding_shape(1)
        if (origin_inputshape[-1]==-1):
            origin_inputshape[-2],origin_inputshape[-1]=(input_shape)
            feat_inputshape[-2],feat_inputshape[-1] = (input_shape)
            context.set_binding_shape(0,(origin_inputshape))
            context.set_binding_shape(1,(feat_inputshape))
        input_img_array = np.array([input_image] * batch_size)
        img = torch.from_numpy(input_img_array).float().numpy()
        segment_inputs[0].host = img
        [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in segment_inputs]
        stream.synchronize()
       
        context.execute_async(bindings=segment_bindings, stream_handle=stream.handle)
        stream.synchronize()
        [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in segment_outputs]
        stream.synchronize()
        results = np.array(segment_outputs[0].host).reshape(batch_size, input_shape[0],input_shape[1])    
    return results.transpose(1,2,0)

def build_engine(onnx_path, using_half,engine_file,dynamic_input=True):
    trt.init_libnvinfer_plugins(None, '')
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(EXPLICIT_BATCH) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
        config = builder.create_builder_config()
        config.max_workspace_size = GiB(1)
        if using_half:
            config.set_flag(trt.BuilderFlag.FP16)
        with open(onnx_path, 'rb') as model:
            if not parser.parse(model.read()):
                print ('ERROR: Failed to parse the ONNX file.')
                for error in range(parser.num_errors):
                    print (parser.get_error(error))
                return None
        if dynamic_input:
            profile = builder.create_optimization_profile();
            profile.set_shape("color_input", (1,16,100,100), (1,16,300,300), (1,16,500,500)) 
            config.add_optimization_profile(profile)
        return builder.build_engine(network, config) 
trt_engine=build_engine(onnx_path,usinghalf,engine_file,dynamic_input=True)
print('engine built successfully!')
with open(engine_file, "wb") as f:
   f.write(trt_engine.serialize())
