# -*- coding: UTF-8 -*-
caffe_root = 'caffe/'
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe
import numpy as np


# Input: bottom[0]: Stage1 output landmarks [b, 136]
#
# Output: top[0]: Affine parameters [b, 6]

class TransformParamsLayer(caffe.Layer):

    def setup(self, bottom, top):
        if len(bottom) != 1:
            raise Exception("Require 1 inputs.")
        self.src_temp = np.zeros((bottom[0].shape[0], 1), float)
        self.sv_temp = np.zeros((bottom[0].shape[0], 1), float)
        self.src_vec = np.zeros((bottom[0].shape[0], bottom[0].shape[1]), float)
        self.des_vec = np.zeros((bottom[0].shape[0], bottom[0].shape[1]), float)
        self.destinationx = [28.0000012, 41.52647385, 28.18473534, 48.89341398, 29.00594527, 56.22849459, 30.54294376,
                          63.44273345, 33.39682386, 70.1509758, 37.81301279, 75.96648242, 43.24753411, 80.8196526,
                          49.22704001, 84.74777134, 55.99999891, 85.90433046, 62.77296086, 84.74777136, 68.75246374,
                          80.81965265, 74.18698812, 75.96648249, 78.60317555, 70.15097589, 81.45705415, 63.44273355,
                          82.99405419, 56.22849469, 83.81526719, 48.89341409, 84.00000136, 41.52647395, 33.20508288,
                          36.05814056, 36.70989054, 32.88769673, 41.65818614, 31.95748288, 46.76025352, 32.70177806,
                          51.53587937, 34.70101794, 60.46412169, 34.70101795, 65.23974756, 32.70177809, 70.34181493,
                          31.95748293, 75.29010749, 32.8876968, 78.79491666, 36.05814065, 55.999999, 40.50158622,
                          55.99999899, 45.29959221, 55.99999898, 50.06187029, 55.99999897, 54.97181038, 50.36656546,
                          58.21004389, 53.07869107, 59.19360786, 55.99999896, 60.06598282, 58.9213099, 59.19360787,
                          61.63343552, 58.21004391, 38.90346102, 41.06531973, 41.91323263, 39.29239318, 45.56029355,
                          39.34859745, 48.73460122, 41.80838019, 45.30799628, 42.4491319, 41.68195912, 42.39384911,
                          63.26539982, 41.80838022, 66.43970446, 39.34859749, 70.08676842, 39.29239324, 73.0965385,
                          41.06531979, 70.31803888, 42.39384916, 66.69200172, 42.44913194, 45.14172212, 67.10212643,
                          49.1410169, 65.52928901, 53.18444242, 64.65379684, 55.99999895, 65.38025847, 58.81555854,
                          64.65379685, 62.85898405, 65.52928904, 66.85828186, 67.10212647, 62.98245642, 70.94508574,
                          59.06514608, 72.62448044, 55.99999894, 72.94832477, 52.93485484, 72.62448043, 49.01754451,
                          70.94508572, 46.82482707, 67.32302035, 53.14206949, 67.0548055, 55.99999895, 67.36606539,
                          58.85793145, 67.05480551, 65.17517387, 67.32302038, 58.91110973, 69.0006451, 55.99999895,
                          69.3473227, 53.08889121, 69.00064509]

    def reshape(self, bottom, top):
        self.destinationx = np.reshape(self.destinationx, (68, 2))
        self.source = np.reshape(bottom[0].data, (-1, 68, 2))
        top[0].reshape(bottom[0].shape[0], 6)

    def forward(self, bottom, top):
        for n in range(bottom[0].shape[0]):
            sourc = self.source[n]
            destMean = np.mean(self.destinationx, axis=0)
            srcMean = np.mean(sourc, axis=0)

            srcVec = (sourc - srcMean).flatten()
            destVec = (self.destinationx - destMean).flatten()

            # self.src_vec[n, :] = srcVec
            # self.des_vec[n, :] = destVec

            self.src_temp[n, 0] = np.linalg.norm(srcVec, 2) ** 2
            self.sv_temp[n, 0] = np.dot(srcVec, destVec)

            a = self.sv_temp[n, 0] / self.src_temp[n, 0]
            b = 0
            for i in range(self.destinationx.shape[0]):
                b += srcVec[2 * i] * destVec[2 * i + 1] - srcVec[2 * i + 1] * destVec[2 * i]
            b = b / np.linalg.norm(srcVec, 2) ** 2

            A = np.zeros((2, 2))
            A[0, 0] = a
            A[0, 1] = b
            A[1, 0] = -b
            A[1, 1] = a
            srcMean = np.dot(srcMean, A)

            top[0].data[n, :] = np.concatenate((A.flatten(), destMean - srcMean))

    def backward(self, top, propagate_down, bottom):
        pass








