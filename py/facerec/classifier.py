from facerec.distance import EuclideanDistance
from facerec.util import asRowMatrix
import logging
import numpy as np
import operator as op
import math

class AbstractClassifier(object):
	weights = 0

	def compute(self,X,y):
		raise NotImplementedError("Every AbstractClassifier must implement the compute method.")
	
	def predict(self,X):
		raise NotImplementedError("Every AbstractClassifier must implement the predict method.")

	def set_uniform_weights(self, y):
		N = len(y)
		weights = (1.0/N)*np.ones(N)
		self.weights = weights

class NearestNeighbor(AbstractClassifier):
	"""
	Implements a k-Nearest Neighbor Model for a generic distance metric.
	"""
	def __init__(self, dist_metric=EuclideanDistance(), k=1):
		AbstractClassifier.__init__(self)
		self.k = k
		self.dist_metric = dist_metric

	def compute(self, X, y):
		self.X = X
		self.y = y
	
	def predict(self, q):
		distances = []
		for xi in self.X:
			xi = xi.reshape(-1,1)
			d = self.dist_metric(xi, q)
			distances.append(d)
		if len(distances) > len(self.y):
			raise Exception("More distances than classes. Is your distance metric correct?")
		idx = np.argsort(np.array(distances))
		sorted_y = self.y[idx]
		sorted_y = sorted_y[0:self.k]
		hist = dict((key,val) for key, val in enumerate(np.bincount(sorted_y)) if val)
		return max(hist.iteritems(), key=op.itemgetter(1))[0]
		
	def __repr__(self):
		return "NearestNeighbor (k=%s, dist_metric=%s)" % (self.k, repr(self.dist_metric))

class AdaBoost(AbstractClassifier):
	def __init__(self, weak_classifier_type, T=100):
		AbstractClassifier.__init__(self)
		self.WeakClassifierType = weak_classifier_type
		self.T = T

	def compute(self, X, y):
		Y = np.array(y)
		N = len(Y)
		w = (1.0/N)*np.ones(N)
		self.weak_classifier_ensemble = []
		self.alpha = []
		for t in range(self.T):
			weak_learner = self.WeakClassifierType
			weak_learner.weights = w
			weak_learner.compute(X,y)
			Y_pred = []
			for x in X:
				Y_pred.append(weak_learner.predict(x))
			# (Y=-1, Y_pred=1) False Positive
			# (Y=1, Y_pred=-1) Missing  should be assigned more weights
			#ww = np.log(k)*(numpy.exp( (Y-Y_pred)>1 ) - 1)/(numpy.exp(1)-1) + 1
			e = sum(0.5*w*abs((Y-Y_pred)))/sum(w)
			#e = sum(0.5*w*abs(Y-Y_pred))
			ee = (1-e)/(e*1.0)
			alpha = 0.5*math.log(ee+0.00001)
			w *= np.exp(-alpha*Y*Y_pred) #*ww) # increase weights for wrongly classified
			w /= sum(w)
			self.weak_classifier_ensemble.append(weak_learner)
			self.alpha.append(alpha)

	def predict(self,X):
		X = np.array(X)
		N, d = X.shape
		Y = np.zeros(N)
		for t in range(self.T):
			#sys.stdout.write('.')
			weak_learner = self.weak_classifier_ensemble[t]
			#print Y.shape, self.alpha[t], weak_learner.predict(X).shape
			Y += self.alpha[t]*weak_learner.predict(X)
		return int(Y[0])

# libsvm
try:
	from svmutil import *
	# for suppressing output
except ImportError:
	from libsvm.svmutil import *
	logger = logging.getLogger("facerec.classifier.SVM")
	logger.debug("Import Error: libsvm bindings not available.")
except:
	logger = logging.getLogger("facerec.classifier.SVM")
	logger.debug("Import Error: libsvm bindings not available.")

import sys
from StringIO import StringIO
# function handle to stdout
bkp_stdout=sys.stdout

class SVM(AbstractClassifier):
	"""
	This class is just a simple wrapper to use libsvm in the 
	CrossValidation module. If you don't use this framework
	use the validation methods coming with LibSVM, they are
	much easier to access (simply pass the correct class 
	labels in svm_predict and you are done...).
	
	The grid search method in this class is somewhat similar
	to libsvm grid.py, as it performs a parameter search over
	a logarithmic scale.	Again if you don't use this framework, 
	use the libsvm tools as they are much easier to access.
	
	Please keep in mind to normalize your input data, as expected
	for the model. There's no way to assume a generic normalization
	step.
	"""
	
	def __init__(self, param=svm_parameter("-q")):
		AbstractClassifier.__init__(self)
		self.logger = logging.getLogger("facerec.classifier.SVM")
		self.param = param
		self.svm = svm_model()
		self.param = param
		
	def compute(self, X, y):
		self.logger.debug("SVM TRAINING (C=%.2f,gamma=%.2f,p=%.2f,nu=%.2f,coef=%.2f,degree=%.2f)" % (self.param.C, self.param.gamma, self.param.p, self.param.nu, self.param.coef0, self.param.degree))
		# turn data into a row vector (needed for libsvm)
		X = asRowMatrix(X)
		y = np.asarray(y)
		problem = svm_problem(y, X.tolist())		
		self.svm = svm_train(problem, self.param)
		self.y = y
		
	def predict(self, X):
		X = np.asarray(X).reshape(1,-1)
		sys.stdout=StringIO() 
		p_lbl, p_acc, p_val = svm_predict([0], X.tolist(), self.svm)
		sys.stdout=bkp_stdout
		return int(p_lbl[0])
		
	def __repr__(self):		
		return "Support Vector Machine (kernel_type=%s, C=%.2f,gamma=%.2f,p=%.2f,nu=%.2f,coef=%.2f,degree=%.2f)" % (KERNEL_TYPE[self.param.kernel_type], self.param.C, self.param.gamma, self.param.p, self.param.nu, self.param.coef0, self.param.degree)
