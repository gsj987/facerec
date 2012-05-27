#_*_ coding:utf-8_*_
from facerec.dataset import DataSet
from facerec.feature import ChainOperator
from facerec.feature import LBP
from facerec.feature import MulitiScalesLBP
from facerec.feature import ExtendedLBP
from facerec.feature import RadiusInvariantUniformLBP
from facerec.feature import SingleGridLBP
from facerec.feature import PCA
from facerec.distance import HistogramIntersection
from facerec.distance import ChiSquareDistance
from facerec.distance import BinRatioDistance
from facerec.distance import ChiSquareBRD
from facerec.distance import ChiSquareWeightedDistance
from facerec.distance import EuclideanDistance

from facerec.facemask import FACEMASK
from facerec.classifier import NearestNeighbor
from facerec.classifier import SVM
from facerec.classifier import svm_parameter
from facerec.classifier import AdaBoost
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
#formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s -%(message)s')
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
    #lbp_operator = RadiusInvariantUniformLBP(r,w)
    for s in range(8,9):
        
        #classifier = SVM(svm_parameter('-t 1 -c 2 -g 2 -r 262144 -q'))
        classifier = AdaBoost(NearestNeighbor(EuclideanDistance(),10), 50)
# define Fisherfaces as feature extraction method

        #feature = ChainOperator(HistogramEqualization(), LBP(sz=(s,s)))
        model1 = MulitiScalesLBP(sz=(s,s))
        data1 = model1.compute(dataSet.data, dataSet.labels)
        feature = PCA(200)
        #feature = ChainOperator(model1, model2)
# now stuff them into a PredictableModel
        model = PredictableModel(feature=feature, classifier=classifier)
# show fisherfaces
        model.compute(data1,dataSet.labels)

       #print model.feature.model2.eigenvectors.shape, dataSet.data
#es = model.feature.model2.eigenvectors

        #plot_eigenvectors(model.feature, 9, sz=dataSet.data[0].shape, filename=None)
# perform a 5-fold cross validation
        cv = KFoldCrossValidation(model, 5)
        cv.validate(data1, dataSet.labels)
        
        print s, r, w,cv.tp, cv.fp, "%.4f" %(cv.tp/(cv.tp+cv.fp+0.001)) 
