# LazyRegressor-bimal

# Lazypredict using LazyRegressor
Lazy Predict helps build a lot of basic models without much code and helps understand which models works better without any parameter tuning.

!pip install lazypredict

from lazypredict.Supervised import LazyRegressor
from sklearn import datasets
from sklearn.utils import shuffle
import numpy as np
boston = datasets.load_boston()
# making  train_test here :
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2, random_state = 2)
# ML in two lines
reg = LazyRegressor(verbose=0, ignore_warnings=False, custom_metric=None)
models, predictions = reg.fit(X_train, X_test, y_train, y_test)

# printing Models
print(models)

Output can be seen by running the .ipynb file
Thank you 
Bimal
+91 7012013239

