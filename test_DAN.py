import numpy as np
from ImageServer import ImageServer
from FaceAlignment import FaceAlignment
import tests

datasetDir ="data/"

verbose = False
showResults = False
showCED = True

normalization = 'centers'
failureThreshold = 0.08

net_work_path = './prototxt/stage2/deploy_c.prototxt'
weight_path = './output/stage2/DAN_s2_iter_204000.caffemodel'

with np.load("./DAN.npz") as f:
    meanImg = f["meanImg"]
    stdDevImg = f["stdDevImg"]
    initLandmarks = f["initLandmarks"]

network = FaceAlignment(net_work_path, weight_path, 112, 112, 1, 2)
network.setNorm(meanImg, stdDevImg, initLandmarks)

# print ("Network being tested: " + networkFilename)
print ("Normalization is set to: " + normalization)
print ("Failure threshold is set to: " + str(failureThreshold))


# commonSet = ImageServer.Load(datasetDir + "commonSet.npz")
challengingSet = ImageServer.Load(datasetDir + "challengingSet.npz")
# w300 = ImageServer.Load(datasetDir + "w300Set.npz")

# print ("Processing common subset of the 300W public test set (test sets of LFPW and HELEN)")
# commonErrs = tests.LandmarkError(commonSet, network, normalization, showResults, verbose)
print ("Processing challenging subset of the 300W public test set (IBUG dataset)")
challengingErrs = tests.LandmarkError(challengingSet, network, normalization, showResults, verbose)

# fullsetErrs = commonErrs + challengingErrs
# print ("Showing results for the entire 300W pulic test set (IBUG dataset, test sets of LFPW and HELEN")
# print("Average error: {0}".format(np.mean(fullsetErrs)))
# tests.AUCError(fullsetErrs, failureThreshold, showCurve=showCED)

# print ("Processing 300W private test set")
# w300Errs = tests.LandmarkError(w300, network, normalization, showResults, verbose)
# tests.AUCError(w300Errs, failureThreshold, showCurve=showCED)


# filters = network.net.params['conv4_1'][0].data
# network.vis_square(filters)

