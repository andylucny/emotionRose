from tensorflow.keras.models import load_model
import numpy as np
import cv2 as cv

backbone = load_model('mobilenet_7_backbone.h5')
face = cv.imread('happy.png')
blob = (cv.resize(cv.cvtColor(face,cv.COLOR_BGR2RGB),(224, 224)) - np.array([123.68, 116.779, 103.939]))/255.0
features = backbone(np.array([blob]))[0].numpy()
#features.shape #1024
