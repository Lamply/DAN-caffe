# -*- coding: UTF-8 -*-
# caffe_root = 'caffe/'
# import sys
# sys.path.insert(0, caffe_root + 'python')

import caffe
import itertools
import numpy as np

# Input: bottom[0]: Transformed landmarks (flatten)
#        bottom[1]: Source images
#
# Output: top[0]: Output images

class LandmarkImageLayer(caffe.Layer):

    def setup(self, bottom, top):
        if len(bottom) != 2:
            raise Exception("Need 2 inputs.")
        # params = eval(self.param_str)
        params = 16
        self.num_points = bottom[0].shape[1]/2
        self.img_shape = bottom[1].shape[2:]
        self.patch_size = params
        self.half_size = params / 2
        self.offsets = np.array(list(itertools.product(range(-self.half_size, self.half_size + 1), range(-self.half_size, self.half_size + 1))))
        self.mid_value = np.zeros((bottom[0].shape[0], self.num_points, self.offsets.shape[0], self.offsets.shape[1]), dtype=np.float32)
        self.mid_value_l2 = np.zeros((bottom[0].shape[0], self.num_points, self.offsets.shape[0]), dtype=np.float32)
        self.mid_value_sq = np.zeros((bottom[0].shape[0], self.num_points, self.offsets.shape[0]), dtype=np.float32)

        self.max_mask = np.zeros((bottom[0].shape[0], 1, self.img_shape[0], self.img_shape[1]), int)

    def reshape(self, bottom, top):
        top[0].reshape(bottom[0].shape[0], 1, self.img_shape[0], self.img_shape[1])
        # top[0].reshape(*bottom[1].shape)

    def forward(self, bottom, top):
        for n in range(bottom[0].shape[0]):
            landmarks = bottom[0].data[n].reshape((-1, 2))
            landmarks[:, 0] = np.clip(landmarks[:, 0], self.half_size, self.img_shape[1] - 1 - self.half_size)
            landmarks[:, 1] = np.clip(landmarks[:, 1], self.half_size, self.img_shape[0] - 1 - self.half_size)

            imgs = np.zeros((landmarks.shape[0], 1, self.img_shape[0], self.img_shape[1]), dtype=np.float32)

            for i in range(landmarks.shape[0]):
                img = np.zeros((1, self.img_shape[0], self.img_shape[1]), dtype=np.float32)

                intLandmark = landmarks[i].astype('int32')
                locations = self.offsets + intLandmark
                dxdy = landmarks[i] - intLandmark

                offsetsSubPix = self.offsets - dxdy
                self.mid_value[n, i, :] = offsetsSubPix
                self.mid_value_l2[n, i, :] = np.sqrt(np.sum(offsetsSubPix * offsetsSubPix, axis=1) + 1e-6)
                self.mid_value_sq[n, i, :] = (1 + self.mid_value_l2[n, i, :]) * (1 + self.mid_value_l2[n, i, :])
                vals = 1 / (1 + self.mid_value_l2[n, i, :])

                img[0, locations[:, 1], locations[:, 0]] = vals
                imgs[i, :] = img.copy()

            max_mix = np.max(imgs, 0)

            for ii in range(landmarks.shape[0]):
                self.max_mask[n, :] = np.where(max_mix == imgs[ii, :], ii, self.max_mask[n, :])

            top[0].data[n, :] = max_mix

    def backward(self, top, propagate_down, bottom):
        pass

