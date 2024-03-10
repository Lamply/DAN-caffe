# -*- coding: UTF-8 -*-
# caffe_root = 'caffe/'
# import sys
# sys.path.insert(0, caffe_root + 'python')

import caffe
import numpy as np


# Input: bottom[0]: Stage1 output landmarks [b, 136]
#        bottom[1]: Affine parameters [b, 6]
#
# Output: top[0]: Transformed landmarks [b, 136]

class LandmarkTransformLayer(caffe.Layer):

    def setup(self, bottom, top):
        if len(bottom) != 2:
            raise Exception("Need 2 inputs.")
        params = eval(self.param_str)
        # self.inverse = params['inverse']
        if params['inverse'] == 'true':
            self.inverse = True
        else:
            self.inverse = False
        self.A = np.zeros((2,2))
        
    def affine_transform_helper(self, landmarks, transform):
        A = np.zeros((2, 2))

        A[0, 0] = transform[0]
        A[0, 1] = transform[1]
        A[1, 0] = transform[2]
        A[1, 1] = transform[3]
        t = transform[4:6]

        if self.inverse:
            A = np.linalg.inv(A)
            t = np.dot(-t, A)

        self.A = A

        output = (np.dot(landmarks.reshape((-1, 2)), A) + t).flatten()
        return output

    def reshape(self, bottom, top):
        top[0].reshape(*bottom[0].shape)

    def forward(self, bottom, top):
        for n in range(bottom[0].shape[0]):
            top[0].data[n, :] = self.affine_transform_helper(bottom[0].data[n], bottom[1].data[n])

    def backward(self, top, propagate_down, bottom):
        if propagate_down[0]:
            for n in range(bottom[0].shape[0]):
                X = np.reshape(bottom[0].data[n], (-1, 2))
                diff = top[0].diff[n]

                bottom[0].diff[n, ...] = (np.dot(diff.reshape((-1, 2)), self.A.transpose())).flatten()

                bottom[1].diff[n, 0:4] = np.dot(X.transpose(), diff.reshape((-1, 2))).flatten()

                B = np.ones((X.shape[0], X.shape[1]), float)
                Yt = np.dot(B.transpose(), diff.reshape((-1, 2)))
                bottom[1].diff[n, 4:6] = Yt[0]




