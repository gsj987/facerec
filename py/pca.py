#_*_ coding:utf-8_*_
from facerec.dataset import DataSet
from facerec.feature import ChainOperator
from facerec.feature import PCA
from facerec.distance import EuclideanDistance, CosineDistance
from facerec.classifier import NearestNeighbor
from facerec.model import PredictableModel
from facerec.validation import KFoldCrossValidation
import scipy.misc.pilutil as smp
import numpy as np
from facerec.visual import plot_eigenvectors
from facerec.preprocessing import HistogramEqualization, TanTriggsPreprocessing


import logging,sys
import random
# set up a handler for logging
#handler = logging.StreamHandler(sys.stdout)
#formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#handler.setFormatter(formatter)
# add handler to facerec modules
#logger = logging.getLogger("facerec")
#logger.addHandler(handler)
#logger.setLevel(logging.DEBUG)
# load a dataset
random.seed()
dataSet = DataSet("/Users/gsj987/Desktop/毕设资料/faces_girls")
#idx = np.argsort([random.random() for i in xrange(len(dataSet.labels))])
#dataSet.labels = dataSet.labels[idx]
for ind in range(5,55,5):
# define a 1-NN classifier with Euclidean Distance
  classifier = NearestNeighbor(dist_metric=EuclideanDistance(), k=ind)
# define Fisherfaces as feature extraction method

  feature = ChainOperator(HistogramEqualization(), PCA(num_components=270))
#print feature.compute(dataSet.data, dataSet.labels)
# now stuff them into a PredictableModel
  model = PredictableModel(feature=feature, classifier=classifier)
# show fisherfaces
  model.compute(dataSet.data,dataSet.labels)

#print model.feature.model2.eigenvectors.shape, dataSet.data
#es = model.feature.model2.eigenvectors

#img = smp.toimage(np.dot(es,dd[0]).reshape(120,120))
#img.save("pca100.jpg")
#plot_eigenvectors(model.feature.model2.eigenvectors, 9, sz=dataSet.data[0].shape, filename=None)
# perform a 5-fold cross validation
  cv = KFoldCrossValidation(model, 10)
  cv.validate(dataSet.data, dataSet.labels)

  print ind,cv.at(None)
