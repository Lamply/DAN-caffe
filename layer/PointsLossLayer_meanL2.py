# -*- coding: UTF-8 -*-
caffe_root = 'caffe/'
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe
import numpy as np


# Input: bottom[0]: Predict Landmarks [b, 136]
#        bottom[1]: Init Landmarks [b, 68, 2]
#        bottom[2]: Ground True Landmarks [b, 68, 2]
#        bottom[3]: Eye Distances [b, 1]
#
# Output: top[0]: Loss [b, 1]

class PointsLossLayer(caffe.Layer):

    def setup(self, bottom, top):
        if len(bottom) != 4:
            raise Exception("Need 4 inputs.")

    def reshape(self, bottom, top):
        if bottom[0].count != bottom[2].count:
            raise Exception("Inputs must have the same dimension.")

        self.res_land = np.reshape(bottom[0].data, (bottom[2].shape[0], bottom[2].shape[1], bottom[2].shape[2]))
        top[0].reshape(1)
        self.diff = np.zeros_like(bottom[2].data, dtype=np.float32)
        self.l2norm = np.zeros(np.array([bottom[2].shape[0], bottom[2].shape[1]]), dtype=np.float32)
        # init_shape = bottom[1].data[0]
        # self.board_initshape = np.reshape(init_shape, [1, init_shape.shape[0], init_shape.shape[1]])
        self.board_eyedist = np.reshape(bottom[3].data, [bottom[3].shape[0], 1, 1])

    def forward(self, bottom, top):
        self.diff[...] = self.res_land - bottom[2].data

        # Mean Centers-L2Norm Distance
        self.l2norm[...] = np.sqrt(np.sum(self.diff ** 2, axis=2))
        meanError = np.mean(self.l2norm, axis=1)
        top[0].data[0] = np.mean(meanError / bottom[3].data)

        # EyedistNorm-Square Distance
        # top[0].data[...] = np.sum(self.diff ** 2 / self.board_eyedist) / (bottom[0].shape[0] * 2.)

    def backward(self, top, propagate_down, bottom):
        board_l2norm = np.reshape(self.l2norm, [self.l2norm.shape[0], self.l2norm.shape[1], 1])
        bottom[0].diff[...] = (self.diff / (board_l2norm * bottom[0].shape[0] * 68 * self.board_eyedist)).reshape(bottom[2].shape[0], -1)

        # bottom[0].diff[...] = self.diff / (bottom[0].shape[0] * self.board_eyedist)


        
