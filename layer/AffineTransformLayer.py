# -*- coding: UTF-8 -*-
# caffe_root = 'caffe/'
# import sys
# sys.path.insert(0, caffe_root + 'python')

import caffe
import numpy as np


# Input: bottom[0]: Source images
#        bottom[1]: Affine parameters
#
# Output: top[0]: Transformed images

class AffineTransformLayer(caffe.Layer):

    def setup(self, bottom, top):
        if len(bottom) != 2:
            raise Exception("Need 2 inputs.")

        img_shape = bottom[0].shape
        # if img_shape[2] != img_shape[3]:
        #     raise Exception("Need 1:1 square images inputs.")

        self.out_img_height = img_shape[2]
        self.out_img_width = img_shape[3]

        self.in_img_height = img_shape[2]
        self.in_img_width = img_shape[3]

        # self.A = np.zeros((bottom[0].shape[0], 2, 2), float)
        # self.t = np.zeros((bottom[0].shape[0], 2), float)
        self.outPixels = np.zeros((bottom[0].shape[0], self.out_img_height * self.out_img_width, 2), float)

        self.dx = np.zeros((bottom[0].shape[0], self.out_img_height * self.out_img_width), float)
        self.dy = np.zeros((bottom[0].shape[0], self.out_img_height * self.out_img_width), float)
        self.pixel_minmin = np.zeros((bottom[0].shape[0], 1, 2), float)
        self.pixel_maxmin = np.zeros((bottom[0].shape[0], self.out_img_width, 2), float)
        self.pixel_minmax = np.zeros((bottom[0].shape[0], self.out_img_height, 2), float)
        self.pixel_maxmax = np.zeros((bottom[0].shape[0], self.out_img_height * self.out_img_width, 2), float)

    def reshape(self, bottom, top):
        top[0].reshape(bottom[0].shape[0], 1, self.out_img_height, self.out_img_width)

    def forward(self, bottom, top):
        for n in range(bottom[0].shape[0]):
            img = bottom[0].data[n]
            transform = bottom[1].data[n]
            A = np.zeros((2, 2))

            A[0, 0] = transform[0]
            A[0, 1] = transform[1]
            A[1, 0] = transform[2]
            A[1, 1] = transform[3]
            t = transform[4:6]

            # print A
            A = np.linalg.inv(A)
            t = np.dot(-t, A)

            # self.A[n, :] = A
            # self.t[n, :] = t

            pixels = [(x, y) for x in range(self.out_img_width) for y in range(self.out_img_height)]
            pixels = np.array(pixels, dtype=np.float32)

            outPixels = np.dot(pixels, A) + t

            outPixels[:, 0] = np.clip(outPixels[:, 0], 0, self.in_img_height - 2)
            outPixels[:, 1] = np.clip(outPixels[:, 1], 0, self.in_img_height - 2)

            self.outPixels[n, :] = outPixels

            outPixelsMinMin = outPixels.astype('int32')
            outPixelsMaxMin = outPixelsMinMin + [1, 0]
            outPixelsMinMax = outPixelsMinMin + [0, 1]
            outPixelsMaxMax = outPixelsMinMin + [1, 1]

            dx = outPixels[:, 0] - outPixelsMinMin[:, 0]
            dy = outPixels[:, 1] - outPixelsMinMin[:, 1]

            self.dx[n, :] = dx
            self.dy[n, :] = dy
            self.pixel_minmin[n, 0] = outPixelsMinMin[0]
            self.pixel_maxmin[n, :] = outPixelsMaxMin[:self.out_img_width*self.out_img_height:self.out_img_height]
            self.pixel_minmax[n, :] = outPixelsMinMax[:self.out_img_height]
            self.pixel_maxmax[n, :] = outPixelsMaxMax

            pixels = pixels.astype('int32')

            outImg = np.zeros((1, self.out_img_height, self.out_img_width), float)
            outImg[0, pixels[:, 1], pixels[:, 0]] += (1 - dx) * (1 - dy) * img[0, outPixelsMinMin[:, 1], outPixelsMinMin[:, 0]]
            outImg[0, pixels[:, 1], pixels[:, 0]] += dx * (1 - dy) * img[0, outPixelsMaxMin[:, 1], outPixelsMaxMin[:, 0]]
            outImg[0, pixels[:, 1], pixels[:, 0]] += (1 - dx) * dy * img[0, outPixelsMinMax[:, 1], outPixelsMinMax[:, 0]]
            outImg[0, pixels[:, 1], pixels[:, 0]] += dx * dy * img[0, outPixelsMaxMax[:, 1], outPixelsMaxMax[:, 0]]

            top[0].data[n, :] = outImg

    def backward(self, top, propagate_down, bottom):
        pass

