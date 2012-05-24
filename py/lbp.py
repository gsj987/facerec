#_*_ coding:utf-8_*_
from facerec.dataset import DataSet
from facerec.feature import ChainOperator
from facerec.feature import LBP,MulitiScalesLBP,ExtendedLBP, RadiusInvariantUniformLBP, SingleGridLBP
from facerec.distance import HistogramIntersection,ChiSquareDistance,BinRatioDistance,ChiSquareBRD, ChiSquareWeightedDistance
from facerec.facemask import FACEMASK
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
dataSet = DataSet("/Users/gsj987/Desktop/毕设资料/faces_boys")
#idx = np.argsort([random.random() for i in xrange(len(dataSet.labels))])
#dataSet.labels = dataSet.labels[idx]
# define a 1-NN classifier with Euclidean Distance

for w in [8]:
  for r in range(1,2):
    #convert_table = RadiusInvariantUniformLBP.build_convert_table(w)
    lbp_operator = RadiusInvariantUniformLBP(r,w)
    for s in range(3,11):
        
        classifier = NearestNeighbor(
            dist_metric=ChiSquareDistance(),
            k=25
        )
# define Fisherfaces as feature extraction method

        #feature = ChainOperator(HistogramEqualization(), LBP(sz=(s,s)))
        feature = LBP(lbp_operator,sz=(s,s))
# now stuff them into a PredictableModel
        model = PredictableModel(feature=feature, classifier=classifier)
# show fisherfaces
        model.compute(dataSet.data,dataSet.labels)

       #print model.feature.model2.eigenvectors.shape, dataSet.data
#es = model.feature.model2.eigenvectors

        #plot_eigenvectors(model.feature, 9, sz=dataSet.data[0].shape, filename=None)
# perform a 5-fold cross validation
        cv = KFoldCrossValidation(model, 5)
        cv.validate(dataSet.data, dataSet.labels)
        
        print s, r, w,cv.tp, cv.fp, "%.4f" %(cv.tp/(cv.tp+cv.fp+0.001)) 
