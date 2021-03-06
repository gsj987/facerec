from facerec.feature import AbstractFeature
from facerec.util import asColumnMatrix
import numpy as np
import scipy.spatial.distance as ssd

class FeatureOperator(AbstractFeature):
	"""
	A FeatureOperator operates on two feature models.
	
	Args:
		model1 [AbstractFeature]
		model2 [AbstractFeature]
	"""
	def __init__(self,model1,model2):
		if (not isinstance(model1,AbstractFeature)) or (not isinstance(model2,AbstractFeature)):
			raise Exception("A FeatureOperator only works on classes implementing an AbstractFeature!")
		self.model1 = model1
		self.model2 = model2
	
	def __repr__(self):
		return "FeatureOperator(" + repr(self.model1) + "," + repr(self.model2) + ")"
	
class ChainOperator(FeatureOperator):
	"""
	The ChainOperator chains two feature extraction modules:
		model2.compute(model1.compute(X,y),y)
	Where X can be generic input data.
	
	Args:
		model1 [AbstractFeature]
		model2 [AbstractFeature]
	"""
	def __init__(self,model1,model2):
		FeatureOperator.__init__(self,model1,model2)
		
	def compute(self,X,y):
		X = self.model1.compute(X,y)
		return self.model2.compute(X,y)
		
	def extract(self,X):
		X = self.model1.extract(X)
		return self.model2.extract(X)
	
	def __repr__(self):
		return "ChainOperator(" + repr(self.model1) + "," + repr(self.model2) + ")"
		
class CombineOperator(FeatureOperator):
	"""
	The CombineOperator combines the output of two feature extraction modules as:
	  (model1.compute(X,y),model2.compute(X,y))
	, where	the output of each feature is a [1xN] or [Nx1] feature vector.
		
		
	Args:
		model1 [AbstractFeature]
		model2 [AbstractFeature]
		
	"""
	def __init__(self,model1,model2):
		FeatureOperator.__init__(self, model1, model2)
		
	def compute(self,X,y):
		A = self.model1.compute(X,y)
		B = self.model2.compute(X,y)
		C = []
		for i in range(0, len(A)):
			ai = np.asarray(A[i]).reshape(1,-1)
			bi = np.asarray(B[i]).reshape(1,-1)
			C.append(np.hstack((ai,bi)))
		return C
	
	def extract(self,X):
		ai = self.model1.extract(X)
		bi = self.model2.extract(X)
		ai = np.asarray(ai).reshape(1,-1)
		bi = np.asarray(bi).reshape(1,-1)
		return np.hstack((ai,bi))

	def __repr__(self):
		return "CombineOperator(" + repr(self.model1) + "," + repr(self.model2) + ")"


class CombineOperatorFirstN(CombineOperator):
  def __init__(self, 
               model1, 
               model2, 
               dataSet, 
               nm=100, 
               operator=ssd.cosine,
               model1_limit=None):
    CombineOperator.__init__(self, model1, model2)
    self.nm = nm
    self.operator = operator
    self.idx = None
    self.__calculate_firstn_features(dataSet.data, 
                                     dataSet.labels, 
                                     dataSet.samples,
                                     model1_limit)

  def __calculate_firstn_features(self, X, y, S, model1_limit):
    f1 = self.model1.compute(X, y)
    f2 = self.model2.compute(X,y)

    SX = np.asarray(S).reshape(1,-1)
    corr1 = []
    corr2 = []
    for feature in asColumnMatrix(f1):
      if abs(np.linalg.norm(feature))<1e-12:
        corr1.append(1)
      else:
        corr1.append(-abs(1-self.operator(feature, SX)))

    for feature in asColumnMatrix(f2):
      if abs(np.linalg.norm(feature))<1e-12:
        corr2.append(1)
      else:
        corr2.append(-abs(1-self.operator(feature, SX)))
   
    if model1_limit==None:
      corr1.extend(corr2)
      self.idx = np.argsort(corr1)
    else:
      idx1 = np.argsort(corr1)
      idx2 = np.argsort(corr2)
      if model1_limit>len(idx1):
        model1_limit = len(idx1)
      idx2 = idx2+len(idx1)
      p1 = idx1[:model1_limit]
      p2 = idx2[:self.nm-model1_limit]
      p3 = idx1[model1_limit:]
      p4 = idx2[len(p2):]
      self.idx = np.hstack((p1,p2))
      self.idx = np.hstack((self.idx,p3))
      self.idx = np.hstack((self.idx, p4))


  def compute(self, X, y):
    features = CombineOperator.compute(self, X, y)
    C = []
    for feature in features:
      C.append(feature.flatten()[self.idx][:self.nm])
    return C

  def extract(self, X):
    feature = CombineOperator.extract(self, X)
    return feature.flatten()[self.idx][:self.nm]

  

class CombineOperatorND(FeatureOperator):
	"""
	The CombineOperator combines the output of two multidimensional feature extraction modules.
		(model1.compute(X,y),model2.compute(X,y))
		
	Args:
		model1 [AbstractFeature]
		model2 [AbstractFeature]
		hstack [bool] stacks data horizontally if True and vertically if False
		
	"""
	def __init__(self,model1,model2, hstack=True):
		FeatureOperator.__init__(self, model1, model2)
		self._hstack = hstack
		
	def compute(self,X,y):
		A = self.model1.compute(X,y)
		B = self.model2.compute(X,y)
		C = []
		for i in range(0, len(A)):
			if self._hstack:
				C.append(np.hstack((A[i],B[i])))
			else:
				C.append(np.vstack((A[i],B[i])))
		return C
	
	def extract(self,X):
		ai = self.model1.extract(X)
		bi = self.model2.extract(X)
		if self._hstack:
			return np.hstack((ai,bi))
		return np.vstack((ai,bi))

	def __repr__(self):
		return "CombineOperatorND(" + repr(self.model1) + "," + repr(self.model2) + ", hstack=" + str(self._hstack) + ")"
