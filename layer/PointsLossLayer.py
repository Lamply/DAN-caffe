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

        bottom[0].reshape(*bottom[2].shape)
        top[0].reshape(1)
        self.diff = np.zeros_like(bottom[0].data, dtype=np.float32)
        self.l2norm = np.zeros(np.array([bottom[2].shape[0], bottom[2].shape[1]]), dtype=np.float32)
        init_shape = bottom[1].data[0]
        # self.board_initshape = np.reshape(init_shape, [1, init_shape.shape[0], init_shape.shape[1]])
        self.board_eyedist = np.reshape(bottom[3].data, [bottom[3].shape[0], 1, 1])

    def forward(self, bottom, top):
        # diff = bottom[0].data + self.board_initshape - bottom[2].data
        diff = bottom[0].data - bottom[2].data
        # self.diff[...] = bottom[0].data + bottom[1].data - bottom[2].data
        
        self.diff[...] = diff
        # print(bottom[0].data.shape)
        # print(bottom[1].data.shape)
        # print(bottom[2].data.shape)
        # print(bottom[3].data.shape)
        # print(bottom[0].count)
        # print(bottom[2].count)
        # raise Exception("Need 4 inputs.")

        # self.l2norm[...] = np.sqrt(np.sum(self.diff ** 2, axis=2))
        # meanError = np.mean(self.l2norm, axis=1)
        # top[0].data[0] = np.mean(meanError / bottom[3].data)

        top[0].data[...] = np.sum(self.diff ** 2 / self.board_eyedist) / (bottom[0].num * 2.)

    def backward(self, top, propagate_down, bottom):
            # board_l2norm = np.reshape(self.l2norm, [self.l2norm.shape[0], self.l2norm.shape[1], 1])
            # bottom[0].diff[...] = self.diff / (board_l2norm * bottom[0].num * 68 * self.board_eyedist)
                # bottom[0].diff[...] = self.diff / (board_eyedist * bottom[0].num)
                # bottom[0].diff[...] = self.diff * top[0].diff[0] / (bottom[0].num * board_eyedist)

        bottom[0].diff[...] = self.diff / (bottom[0].num * self.board_eyedist)


        
