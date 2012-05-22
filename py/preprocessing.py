#_*_ coding:utf-8_*_
from facerec.dataset import DataSet
from facerec.feature import LBP,Fisherfaces
from facerec.distance import EuclideanDistance, CosineDistance
from facerec.classifier import NearestNeighbor
from facerec.model import PredictableModel
from facerec.validation import KFoldCrossValidation
from facerec.visual import plot_eigenvectors
from facerec.lbp import ExtendedLBP
from facerec.preprocessing import HistogramEqualization, TanTriggsPreprocessing,LBPPreprocessing
from PIL import Image
import scipy.misc.pilutil as smp
import numpy as np
import logging,sys

# set up a handler for logging
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
# add handler to facerec modules
logger = logging.getLogger("facerec")
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)
# load a dataset
dataSet = DataSet("/Users/gsj987/Desktop/sjtu-thesis-xelatex/figures/chap3/faces/")
#model = HistogramEqualization()
#model = TanTriggsPreprocessing()
model = LBPPreprocessing(ExtendedLBP(9,8))
features = model.compute(dataSet.data, None)

img = smp.toimage(np.asarray(features[0]))
img.save("t9.jpg")
# define a 1-NN classifier with Euclidean Distance
#classifier = NearestNeighbor(dist_metric=EuclideanDistance())
# define Fisherfaces as feature extraction method

#feature = Fisherfaces(num_components=10)
#print feature.compute(dataSet.data, dataSet.labels)
# now stuff them into a PredictableModel
#model = PredictableModel(feature=feature, classifier=classifier)
# show fisherfaces
#model.compute(dataSet.data,dataSet.labels)

#print model.feature.eigenvectors.shape
#plot_eigenvectors(model.feature.eigenvectors, 9, sz=dataSet.data[0].shape, filename=None)
# perform a 5-fold cross validation
#cv = KFoldCrossValidation(model, k=5)
#cv.validate(dataSet.data, dataSet.labels)

#print cv.at(None)
