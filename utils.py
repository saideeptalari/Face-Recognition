import cv2
import numpy as np

def enhance_image(image):
    image_YCrCb = cv2.cvtColor(image, cv2.COLOR_BGR2YCR_CB)
    Y, Cr, Cb = cv2.split(image_YCrCb)
    Y = cv2.equalizeHist(Y)
    image_YCrCb = cv2.merge([Y, Cr, Cb])
    image = cv2.cvtColor(image_YCrCb, cv2.COLOR_YCR_CB2BGR)
    return image

def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)