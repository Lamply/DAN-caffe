from __future__ import print_function
# -*- coding: UTF-8 -*-
# caffe_root = 'DAN/caffe/'
# import sys
# sys.path.insert(0, caffe_root + 'python')

# from ImageServer import ImageServer
import caffe
import numpy as np
from scipy import ndimage
import utils
from matplotlib import pyplot as plt

class FaceAlignment(object):
    def __init__(self, net_work_path, weight_path, height, width, nChannels, stage):
        if stage != 1 and stage != 2 and stage != 3:
            print('ERROR: input stage should be 1 or 2 or 3')
            return
        caffe.set_mode_cpu()
        # caffe.set_device(0)
        self.net = caffe.Net(net_work_path, weight_path, caffe.TEST)
        self.net.name = 'DAN'
        self.imageHeight = height
        self.imageWidth = width
        self.nChannels = nChannels
        self.stage = stage
        self.init_bias = [28.0000012, 41.52647385, 28.18473534, 48.89341398, 29.00594527, 56.22849459, 30.54294376, 63.44273345, 33.39682386, 70.1509758, 37.81301279, 75.96648242, 43.24753411, 80.8196526, 49.22704001, 84.74777134, 55.99999891, 85.90433046, 62.77296086, 84.74777136, 68.75246374, 80.81965265, 74.18698812, 75.96648249, 78.60317555, 70.15097589, 81.45705415, 63.44273355, 82.99405419, 56.22849469, 83.81526719, 48.89341409, 84.00000136, 41.52647395, 33.20508288, 36.05814056, 36.70989054, 32.88769673, 41.65818614, 31.95748288, 46.76025352, 32.70177806, 51.53587937, 34.70101794, 60.46412169, 34.70101795, 65.23974756, 32.70177809, 70.34181493, 31.95748293, 75.29010749, 32.8876968, 78.79491666, 36.05814065, 55.999999, 40.50158622, 55.99999899, 45.29959221, 55.99999898, 50.06187029, 55.99999897, 54.97181038, 50.36656546, 58.21004389, 53.07869107, 59.19360786, 55.99999896, 60.06598282, 58.9213099, 59.19360787, 61.63343552, 58.21004391, 38.90346102, 41.06531973, 41.91323263, 39.29239318, 45.56029355, 39.34859745, 48.73460122, 41.80838019, 45.30799628, 42.4491319, 41.68195912, 42.39384911, 63.26539982, 41.80838022, 66.43970446, 39.34859749, 70.08676842, 39.29239324, 73.0965385, 41.06531979, 70.31803888, 42.39384916, 66.69200172, 42.44913194, 45.14172212, 67.10212643, 49.1410169, 65.52928901, 53.18444242, 64.65379684, 55.99999895, 65.38025847, 58.81555854, 64.65379685, 62.85898405, 65.52928904, 66.85828186, 67.10212647, 62.98245642, 70.94508574, 59.06514608, 72.62448044, 55.99999894, 72.94832477, 52.93485484, 72.62448043, 49.01754451, 70.94508572, 46.82482707, 67.32302035, 53.14206949, 67.0548055, 55.99999895, 67.36606539, 58.85793145, 67.05480551, 65.17517387, 67.32302038, 58.91110973, 69.0006451, 55.99999895, 69.3473227, 53.08889121, 69.00064509]

    def setNorm(self, meanImg, stdDevImg, initLandmarks):
        self.meanImg = meanImg
        self.stdDevImg = stdDevImg
        self.initLandmarks = initLandmarks

    def processImg(self, img, inputLandmarks, video_flag=0):
        ## Debug 
        self.tmp_img = img[0]

        inputImg, transform = self.CropResizeRotate(img, inputLandmarks)
        inputImg = inputImg - self.meanImg
        inputImg = inputImg / self.stdDevImg

        inputLandmarks_Align = None
        if video_flag != 0:
            inputLandmarks_Align = utils.bestFit(self.initLandmarks, inputLandmarks, False)

        ## Debug 
        self.tmp_transform = transform

        inputImg = np.reshape(inputImg, (1, 1, inputImg.shape[1], inputImg.shape[2]))
        # output = self.generate_network_output([inputImg])[0][0]
        # print(inputImg.shape)
        output = self.generate_network_output(inputImg, video_flag, inputLandmarks_Align)
        # print(output)
        # output += self.init_bias
        landmarks = output.reshape((-1, 2))

        return np.dot(landmarks - transform[1], np.linalg.inv(transform[0]))

    def CropResizeRotate(self, img, inputShape):
        A, t = utils.bestFit(self.initLandmarks, inputShape, True)

        A2 = np.linalg.inv(A)
        t2 = np.dot(-t, A2)

        outImg = np.zeros((self.nChannels, self.imageHeight, self.imageWidth), dtype=np.float32)
        for i in range(img.shape[0]):
            outImg[i] = ndimage.interpolation.affine_transform(img[i], A2, t2[[1, 0]],
                                                               output_shape=(self.imageHeight, self.imageWidth))

        return outImg, [A, t]

    ## Debug 
    def debug_show_predict_inside(self, landmarks):
        landmarks = np.dot(landmarks.reshape((-1, 2)) - self.tmp_transform[1], np.linalg.inv(self.tmp_transform[0]))
        plt.imshow(self.tmp_img, cmap=plt.cm.gray)
        plt.plot(landmarks[:, 0], landmarks[:, 1], '.')
        plt.show()

    def generate_network_output(self, input_data, video_flag=0, last_land=None):
        self.net.blobs['data'].data[...] = input_data

        if video_flag == 0:
            self.net.forward()
        # else:
        #     self.net.blobs['pred_plus'].data[...] = last_land.reshape((-1))           # Bug
        #     self.net.blobs['fc5'].data[...] = self.net.blobs['s2_fc5'].data
        #     self.net.forward(start='pred_plus')
        #     predict = self.net.blobs['pred_final'].data[0].flatten()
        #     print(predict.shape)
        #     return predict

        if self.stage == 1:
            predict = self.net.blobs['pred_plus'].data[0].flatten()
        elif self.stage == 2:
            predict = self.net.blobs['pred_final'].data[0].flatten()
        elif self.stage == 3:
            ## Debug 
            # predict = self.net.blobs['pred_plus'].data[0].flatten()
            # self.debug_show_predict_inside(predict)
            # predict = self.net.blobs['pred_final'].data[0].flatten()
            # self.debug_show_predict_inside(predict)

            self.net.blobs['pred_plus'].data[...] = self.net.blobs['pred_final'].data
            self.net.blobs['fc5'].data[...] = self.net.blobs['s2_fc5'].data
            self.net.forward(start='pred_plus')
            predict = self.net.blobs['pred_final'].data[0].flatten()
            # self.debug_show_predict_inside(predict)

        return predict

    def vis_square(self, data):
        """Take an array of shape (n, height, width) or (n, height, width, 3) and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)"""

        # normalize data for display
        data = (data - data.min()) / (data.max() - data.min())

        # force the number of filters to be square
        n = int(np.ceil(np.sqrt(data.shape[0])))
        padding = (((0, n ** 2 - data.shape[0]),
                (0, 1), (0, 1))                 # add some space between filters
                + ((0, 0),) * (data.ndim - 3))  # don't pad the last dimension (if there is one)
        data = np.pad(data, padding, mode='constant', constant_values=1)  # pad with ones (white)

        # tile the filters into an image
        data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
        data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])

        plt.imshow(data, cmap='gray'); plt.axis('off')
        plt.show()
