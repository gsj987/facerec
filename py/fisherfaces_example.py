#_*_ coding:utf-8_*_
from facerec.dataset import DataSet
from facerec.feature import LBP,Fisherfaces, ChainOperator, PCA,RadiusInvariantUniformLBP, ExtendedLBP, MulitiScalesLBP
from facerec.distance import EuclideanDistance, CosineDistance
from facerec.classifier import NearestNeighbor
from facerec.model import PredictableModel
from facerec.validation import KFoldCrossValidation
from facerec.visual import plot_eigenvectors
from facerec.preprocessing import HistogramEqualization, LBPPreprocessing
import scipy.misc.pilutil as smp

import numpy as np
import logging,sys

# set up a handler for logging
#handler = logging.StreamHandler(sys.stdout)
#formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#handler.setFormatter(formatter)
# add handler to facerec modules
#logger = logging.getLogger("facerec")
#logger.addHandler(handler)
#logger.setLevel(logging.DEBUG)
# load a dataset
dataSet = DataSet("/Users/gsj987/Desktop/毕设资料/faces_girls")
# define a 1-NN classifier with Euclidean Distance
for pc in range(60, 200, 20):
  classifier = NearestNeighbor(dist_metric=EuclideanDistance(), k=25)
# define Fisherfaces as feature extraction method
  

  feature = ChainOperator(HistogramEqualization(),
                          MulitiScalesLBP())
  feature = ChainOperator(feature, PCA(pc))
#print feature.compute(dataSet.data, dataSet.labels)
# now stuff them into a PredictableModel
  model = PredictableModel(feature=feature, classifier=classifier)
# show fisherfaces
  model.compute(dataSet.data,dataSet.labels)
#print feature.eigenvectors.shape

#rec = feature.extract(np.array(dataSet.data[0]))
#im = smp.toimage(rec.reshape(120,120))
#im.show()

#plot_eigenvectors(model.feature.eigenvectors, 9, sz=dataSet.data[0].shape, filename=None)
# perform a 5-fold cross validation
  cv = KFoldCrossValidation(model, k=5)
  cv.validate(dataSet.data, dataSet.labels)

  print pc, cv.tp, cv.fp, "%.4f" %(cv.tp/(cv.tp+cv.fp+0.001))
