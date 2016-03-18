import numpy as np 
from sklearn import linear_model
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt



def logistic_regression(F,T,random_state=None):

	X_train, X_test, y_train, y_test = train_test_split(F, T,
		test_size=0.33,random_state=None)

	# create the regression object
	logregr = linear_model.LogisticRegression()

	# train the model
	logregr.fit(X_train, y_train)

	preds = logregr.predict(X_test)
	score = logregr.score(X_test,y_test)
	proba = logregr.predict_proba(X_test)[:,1]

	logloss = log_loss(proba,y_test)
	adjlog = adj_log_loss(proba,y_test)

	return preds,score,proba,logloss,adjlog


def log_loss(prob,actual):
	"""
	Calculate the log loss the probabilities and the acutal
	outcome of the game
	"""

	prob[prob>.98]=.98
	prob[prob<=.02] = .02
	LL = actual*np.log(prob) + (1-actual)*np.log(1-prob)
	LL = -np.mean(LL)

	return LL

def adj_log_loss(prob,actual):
	"""
	makes the predictions a little more definitive
	"""

	prob[prob<.4] = prob[prob<.4]-.15
	prob[prob>.6] = prob[prob>.6]+.15

	prob[prob>.98]=.98
	prob[prob<=.02] = .02
	LL = actual*np.log(prob) + (1-actual)*np.log(1-prob)
	LL = -np.mean(LL)

	return LL

def visualize_boundary_2d(F,T,classifier=None,resolution=100,cmap='coolwarm'):
	"""
	Visualize 2d decision boundary
	"""

	f0_min = np.min(F[:,0])
	f0_max = np.max(F[:,0])

	f1_min = np.min(F[:,1])
	f1_max = np.max(F[:,1])

	L0 = np.linspace(f0_min,f0_max,num=resolution)
	L1 = np.linspace(f1_min,f1_max,num=resolution)
	xx,yy = np.meshgrid(L0,L1)
	f1 = xx.reshape(resolution**2,1)
	f2 = yy.reshape(resolution**2,1)
	all_possible = np.hstack((f1,f2))

	pred_space = classifier.predict_proba(all_possible)[:,1]

	fig,ax = plt.subplots()
	im = ax.contourf(xx,yy,pred_space.reshape(resolution,resolution),
		cmap=cmap,alpha=.5)
	ax.scatter(F[:,0],F[:,1],c=T,cmap='coolwarm');
	ax.set_xlim(f0_min,f0_max)
	ax.set_ylim(f1_min,f1_max)
	plt.colorbar(im)












