# -*- coding: UTF-8 -*-
# caffe_root = 'caffe/'
import sys
# sys.path.insert(0, caffe_root + 'python')

from ImageServer import ImageServer
# import ImageServer
import caffe
import argparse
import numpy as np
import os
from timer import Timer
from matplotlib import pyplot as plt

from caffe.proto import caffe_pb2
import google.protobuf as pb2
import google.protobuf.text_format

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a caffe network')
    parser.add_argument('--gpu', dest='gpu_id',
                        help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--solver', dest='solver',
                        help='solver prototxt',
                        default=None, type=str)
    parser.add_argument('--iters', dest='max_iters',
                        help='number of iterations to train',
                        default=40000, type=int)
    parser.add_argument('--weights', dest='pretrained_model',
                        help='initialize with pretrained model weights',
                        default='', type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args


class SolverWrapper(object):
    """A simple wrapper around Caffe's solver.
    This wrapper gives us control over he snapshotting process.
    """

    def __init__(self, solver_prototxt, output_dir,
                 pretrained_model=None):
        self.x_errors = []
        self.errors = []
        self.x_errorsTrain = []
        self.errorsTrain = []
        """Initialize the SolverWrapper."""
        self.output_dir = output_dir

        # use SGD solver
        self.solver = caffe.SGDSolver(solver_prototxt)

        # load pretrained model
        if '.caffemodel' in pretrained_model:
            print ('Loading pretrained model '
                   'weights from {:s}').format(pretrained_model)
            self.solver.net.copy_from(pretrained_model)
        elif '.solverstate' in pretrained_model:
            print ('Restore state from {:s}').format(pretrained_model)
            self.solver.restore(pretrained_model)

        # parse solver.prototxt
        self.solver_param = caffe_pb2.SolverParameter()
        with open(solver_prototxt, 'rt') as f:
            pb2.text_format.Merge(f.read(), self.solver_param)


    def drawErrors(self):
        plt.plot(self.x_errors, self.errors)
        plt.plot(self.x_errorsTrain, self.errorsTrain)
        plt.ylim(ymax=np.max([self.errors[0], self.errorsTrain[0]]))
        plt.savefig("./errors.jpg")
        plt.clf()

    def saveErrors(self):
        np.save("./tmp/errors.npy", self.errors)
        np.save("./tmp/errorsTrain.npy", self.errorsTrain)

    def snapshot(self):
        """Take a snapshot of the network. This enables easy use at test-time.
        """
        net = self.solver.net
        filename = (self.solver_param.snapshot_prefix +
                    '_iter_{:d}'.format(self.solver.iter) + '.caffemodel')
        filename = os.path.join(self.output_dir, filename)
        net.save(str(filename))
        print 'Wrote snapshot to: {:s}'.format(filename)
        return filename

    def train_model(self, max_iters):
        """Network training loop."""
        timer = Timer()
        x_train = 0
        train_loss = 0
        train_loss_single = 0
        x_test = 0
        test_loss = 0
        test_loss_single = 0
        while self.solver.iter < max_iters:
            # train and update one iter
            self.solver.step(1)

            # calculate and draw train/test error
            train_loss_single += self.solver.net.blobs['mean_loss'].data[0]

            if self.solver.iter % self.solver_param.display == 0:
                train_loss = train_loss_single / self.solver_param.display
                x_train += self.solver_param.display
                self.x_errorsTrain.append(x_train)
                self.errorsTrain.append(train_loss)
                train_loss_single = 0

            if self.solver.iter % self.solver_param.test_interval == 0:
                for test_it in range(self.solver_param.test_iter[0]):
                    self.solver.test_nets[0].forward()
                    test_loss_single += self.solver.test_nets[0].blobs['mean_loss'].data
                test_loss = test_loss_single / self.solver_param.test_iter[0]
                x_test += self.solver_param.test_interval
                self.x_errors.append(x_test)
                self.errors.append(test_loss)
                test_loss_single = 0

                self.drawErrors()
                # self.saveErrors()
                # textRepresentation = np.column_stack((range(len(self.errors)), self.errors, self.errorsTrain))
                # np.savetxt("../errors.txt", textRepresentation)



def train_net(solver_prototxt, output_dir,
              pretrained_model=None, max_iters=40000):

    sw = SolverWrapper(solver_prototxt, output_dir,
                       pretrained_model=pretrained_model)

    print 'Solving...'
    sw.train_model(max_iters)
    print 'done solving'

    net = sw.solver.net


if __name__ == '__main__':
    args = parse_args()
    # caffe.set_mode_cpu()
    caffe.set_mode_gpu()
    caffe.set_device(args.gpu_id)

    train_net(args.solver, 'output',
              pretrained_model=args.pretrained_model, max_iters=args.max_iters)
              





