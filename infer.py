from caffe2.proto import caffe2_pb2
from caffe2.python import workspace, model_helper, helpers, brew, optimizer
from caffe2.python import core, dyndep
import caffe2.python.predictor.predictor_exporter as pe
import mobExporter as mobile_exporter 

from caffe2.proto.caffe2_pb2 import NetDef
import numpy as np
import os

import cv2

dyndep.InitOpsLibrary("../OpLibs_CPU/Ubuntu 16.4/libcaffe2_CUST_OPS_AddRemovePadding.so")


IMG_S = 513

workspace.ResetWorkspace()

arg_scope = {
    'order': 'NCHW'
}

modelsPath = "../models"
with open(modelsPath + "/GoogleDeepLabInitNet.pb") as f:
    init_net = f.read()
with open(modelsPath + "/GoogleDeepLabPredictNet.pb") as f:
    predict_net = f.read()

p = workspace.Predictor(init_net, predict_net)

with core.DeviceScope(core.DeviceOption(caffe2_pb2.CPU, 0)):

    print("The blobs in the workspace after reset: {}".format(workspace.Blobs()))

    total_iters = 1
    for i in range(total_iters):

        inpImgStr = "../database/1.jpg"        

        inpImg = cv2.imread(inpImgStr)
        inpImg = cv2.resize(inpImg, (IMG_S, IMG_S))
        inpImg = np.array(inpImg,dtype=np.float32)
        inpImg = inpImg.transpose((2, 0, 1))


        inpImg = inpImg / 0.007843137718737125;
        inpImg = inpImg - 1;

        inpImg = np.reshape(inpImg, (1,3,IMG_S,IMG_S))

        outImg = p.run({'inpImg': inpImg})
        outImg = np.reshape(outImg,(21,65,65))
        
        inpImg = np.array(inpImg[0]).transpose(1,2,0)
        
        finalSegmentation = np.argmax(outImg, axis=0)
        finalSegmentation = np.reshape(finalSegmentation, (65,65))
        finalSegmentation *= 10;


        inpImg = inpImg + 1;
        inpImg = inpImg / 0.007843137718737125;

        cv2.imwrite("../output/img_{}.jpg".format(str(i).zfill(5)), np.array(inpImg,dtype=np.uint8) )
        cv2.imwrite("../output/label_{}.jpg".format(str(i).zfill(5)), np.array(finalSegmentation,dtype = np.uint8))


        #raw_input()




























