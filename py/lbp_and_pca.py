#_*_ coding:utf-8_*_
from facerec.dataset import DataSet
from facerec.feature import ChainOperator
from facerec.feature import LBP
from facerec.feature import MulitiScalesLBP
from facerec.feature import ExtendedLBP
from facerec.feature import RadiusInvariantUniformLBP
from facerec.feature import SingleGridLBP
from facerec.feature import CombineOperatorFirstN
from facerec.feature import PCA
from facerec.distance import EuclideanDistance
from facerec.distance import HistogramIntersection
from facerec.distance import ChiSquareDistance 
from facerec.distance import BinRatioDistance
from facerec.distance import ChiSquareBRD
from facerec.distance import ChiSquareWeightedDistance
from facerec.facemask import FACEMASK
from facerec.classifier import NearestNeighbor
from facerec.model import PredictableModel
from facerec.validation import KFoldCrossValidation
import scipy.misc.pilutil as smp
import numpy as np
from facerec.visual import plot_eigenvectors
from facerec.preprocessing import HistogramEqualization, TanTriggsPreprocessing


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
np.seterr(all="raise")
random.seed()
dataSet = DataSet("/Users/gsj987/Desktop/毕设资料/faces_boys",
                  samplename="/Users/gsj987/Desktop/毕设资料/boys_score.csv")
#idx = np.argsort([random.random() for i in xrange(len(dataSet.labels))])
#dataSet.labels = dataSet.labels[idx]
# define a 1-NN classifier with Euclidean Distance

for w in [8]:
  for r in range(1,2):
    #convert_table = RadiusInvariantUniformLBP.build_convert_table(w)
    lbp_operator = RadiusInvariantUniformLBP(r,w)
    for nm in range(50, 500, 25):
      # define Fisherfaces as feature extraction method

      ##feature = ChainOperator(HistogramEqualization(), LBP(sz=(s,s)))
      model1 = LBP(lbp_operator) 
      #feature = MulitiScalesLBP()
      model2 = PCA(400)
      feature = CombineOperatorFirstN(model1, model2, dataSet,
                                      500,model1_limit=nm)
      
      #f1 = model1.compute(dataSet.data, dataSet.labels)
      #f2 = model2.compute(dataSet.data, dataSet.labels)
      #d1 = f1[0].shape
      #d2 = f2[0].shape
      #d3 = feature.idx.shape
      #count = 0
      #for d in feature.idx:
      #  if d<=640:
      #    count += 1
      #  else:
      #    print count
      #    break
      #print feature.idx
      #print d1
      #break
      #print nm, count, "%.4f" %((nm-count+0.0001)/count)
      
      
      #feature = PCA(400)
      for k in [25]:
        classifier = NearestNeighbor(
            dist_metric=EuclideanDistance(),
            k=k
        )

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
        
        print nm, cv.tp, cv.fp, "%.4f" %(cv.tp/(cv.tp+cv.fp+0.001)) 
