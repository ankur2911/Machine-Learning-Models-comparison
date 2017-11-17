# Machine-Learning-Models-comparison

Data set used:  https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data
Pre=processing: Standard Scalar is used for pre-processing dataset. 

		ss = StandardScaler()
X_train_scaled = ss.fit_transform(X)


Model      -       Accuracy   -   Precision

Naïve Bayes		      81.7%	      74%

Decision Tree	    	68%	        70%

Perceptron	      	70%	        75%

SVM	              	77%	        78%

k-nearest neighbors	73%	        71%

Random Forest	    	75%	        75%

Bagging	          	75%	        75%

Adaboost	        	75%	        76%

Gradient Boosting		75%	        69%

Neural Network		  71%	        65%
