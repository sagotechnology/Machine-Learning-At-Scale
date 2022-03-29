# Databricks notebook source
# MAGIC %md # wk7 Demo - Regularization & Gradient Descent con't
# MAGIC __`MIDS w261: Machine Learning at Scale | UC Berkeley School of Information | Spring 2019`__
# MAGIC 
# MAGIC Last week we talked about optimization theory in the context of the supervised learning of a parametric model. Gradient descent allows us to find the model parameters that minimize a loss function, provided that the loss function is (pseudo) convex. Full batch GD is easy to parallelize, however the need to make many passes over the entire dataset is still a cost we'd like to minimize when processing large scale data. Today we'll dive a bit further into techniques for improving the training of supervised ML models. By the end of this live session you should be able to:
# MAGIC * __... explain__ why and how we regularize linear and logistic regression models.
# MAGIC * __... compare__ L1 and L2 regularization in terms of their effects on model parameters.
# MAGIC * __... identify__ a few common gradient descent variants.
# MAGIC * __... describe__ the numerical approximization method & when we might use it.

# COMMAND ----------

# MAGIC %md ### Notebook Set-Up

# COMMAND ----------

# imports
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# COMMAND ----------

# RUN THIS CELL AS IS
# This code snippet reads the user directory name, and stores is in a python variable.
# Next, it creates a folder inside your home folder, which you will use for files which you save inside this notebook.
username = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().apply('user')
userhome = 'dbfs:/user/' + username
print(userhome)
demo7_path = userhome + "/demo7/" 
demo7_path_open = '/dbfs' + demo7_path.split(':')[-1] # for use with python open()
dbutils.fs.mkdirs(demo7_path)

# COMMAND ----------

import sys

# Add the path to system, local or mounted S3 bucket, e.g. /dbfs/mnt/<path_to_bucket>
sys.path.append('/dbfs/mnt/mids-w261/')
sys.path.append('/dbfs/mnt/mids-w261/modules')
sys.path.append('/dbfs/mnt/mids-w261/modules/demo7_helperFunc.py')
sys.path.append('/dbfs/mnt/mids-w261/modules/demo7_homegrown.py')

# COMMAND ----------

# MAGIC %md # Regularization: what and how? 
# MAGIC 
# MAGIC In any supervised learning context we are concerned with making sure our trained models are going to generalize well to unseen data. Regularization techniques help us avoid overfitting. In general they boil down to strategicalling limiting the variance of a model so that we don't accidentally learn patterns in the noise of our data. As we learned early on this reduction in variance will come at the expense of a potential increase in bias. Early stopping is one of the most popular ways to do this but it requires that you are able to set aside a validation set which may not always be feasible, especially with very high dimensional data. Another strategy is to add a penalty term to the loss function when performing your gradient updates. In this section we'll return to the small example from last class to build some intuitions for what this looks like and how this works.
# MAGIC 
# MAGIC ### Demo 1: Small example revisited
# MAGIC 
# MAGIC Run the provided code below to see how regularized and unregularized gradiet descent compare on our small example. Try changing the hyperparameters one at a time to explore their effects paying particular attention to both the path that the training takes and the location of the "final" model. Use the discussion questions below to help you draw some conclusions based on what you see.

# COMMAND ----------

import demo7_helperFunc as hf

#################### Demo Parameters #################### 
# TRY CHANGING THESE & SEE HOW IT AFFECTS OUR SEARCH
REGULARIZATION_PARAMETER = 0.25
LEARNING_RATE = 0.075
ORIGINAL_MODEL = [1,0]  #[parameter, bias]
N_STEPS = 15

################### Effect of Regularization Demo #################### 
### small example data
'''
fivePoints.csv
1,2
3,4
5,5
4,3
2,3
'''

X = [1,3,5,4,2]
y = np.array([2,4,5,3,3])

### normalize X 
# mean = np.mean(X)
# sd = np.std(X)
# X = [(x-mean)/sd for x in X]

### augment x by appending a 1 (the bias term) to each element of X
X = np.column_stack((X, len(X)*[1])) 

### Unregularized Gradient descent training history
no_reg, _ = hf.GDUpdate(X, y, N_STEPS, ORIGINAL_MODEL, LEARNING_RATE)

### Regularized Gradient descent training history
w_reg, _ = hf.GDUpdate(X, y, N_STEPS, ORIGINAL_MODEL, LEARNING_RATE, 
                       reg_type = 'l2', reg_param = REGULARIZATION_PARAMETER)

### plot comparison
display(hf.compareDescent(X, y, no_reg, w_reg, labels=['No Regularization', 'With L2 Regularization']))


# COMMAND ----------

# MAGIC %md  > __DISCUSSION QUESTIONS:__  
# MAGIC  * _How does the regularized training path compare to the unregularized path?_
# MAGIC  * _How do the final regularized models compare to their unregularized counterparts? (Pay particular attention to the weight 0 parameter.) Does this change when you adjust the learning rate?_ 
# MAGIC  * _Given that regularization is something we do to improve the performance of our model does anything seem surprising about the loss where each training path ends?_
# MAGIC  * _How does the regularization parameter affect the training path? How does it affect the final model?_
# MAGIC  * Adjust the learning rate back up to accentuate the zig-zags... take a close look at the direction of the updates in the regularized training path. _Do these updates seem to be going 'down hill'? what conclusions can you draw about the relationship between the regularized model updates and the loss function? -- try to ground your discussion in what you see on the plot rather than theory from other sources._

# COMMAND ----------

# MAGIC %md ### Demo 2: Penalizing the loss for our small example.
# MAGIC 
# MAGIC As you saw in the exercise above, the regularized gradient descent path doesn't exactly seem to follow the principle of taking steps "directly downhill." Under the hood, this is because the regularized training is performing updates based on gradients from a slightly different curve, called the _penalized loss function_ which will (by intention) have a different minimum (one that we think will be more generalizable to unseen data). 
# MAGIC 
# MAGIC Before diving in to why we think this, let's briefly review the two kinds of penalties commonly used in training linear models. As you saw in chapter 6 of Intro to Statistical Learning:
# MAGIC * __Ridge Regression__ adds an __l2__ penalty term to the OLS loss function:
# MAGIC $$
# MAGIC \lambda\sum\_{i=1}^m (w\_{i})^2
# MAGIC $$
# MAGIC * __LASSO Regression__ adds an __l1__ penalty term to the OLS loss function:
# MAGIC $$
# MAGIC \lambda\sum\_{i=1}^m |w\_{i}|
# MAGIC $$
# MAGIC 
# MAGIC Each of these penalty terms includes a hyperparameter, \\(\lambda\\), which allows us to tune how strongly the penalty term changes the shape of the loss curve. It is important to note that by using this penalty term we aren't making a choice to measure _error_ differently, we are creating an entirely different equation to optimize instead. Also note that, unlike the loss function itself, the regularization term is _only_ a function of the weights (not including bias!) and does not depend on the data.
# MAGIC 
# MAGIC Run the code below to view these penalty terms plotted in 3D. Try changing the regularization parameter to see how this affects the curves (hint, due to autoscaling of the plots you'll need to look carefully at the axes for this). Finally, modify the `REG_TYPE` to see how the OLS loss curve in the second row changes when a penalty term is added.

# COMMAND ----------

import demo7_helperFunc as hf

#################### Demo Parameters #################### 
# TRY CHANGING THESE & SEE HOW IT AFFECTS OUR SEARCH
REGULARIZATION_PARAMETER = 150
REG_TYPE = 'l2'  # try 'l1' or 'l2'

################### Penalized Loss Curves Demo #################### 
### small example data
X = np.array([[1,1],[3,1],[5,1],[4,1],[2,1]])
y = np.array([2,3,4,3,3])
data_domain = [0,5]

### set up the grid for the surface plots
grid = np.mgrid[-5:5:0.5,-5:15:0.5]
size = int(np.product(grid.shape)/2)
models_grid = grid.reshape(2,size)

### plot comparison
display(hf.compareLossCurves(X, y, reg_type = REG_TYPE, reg_param = REGULARIZATION_PARAMETER, grid = models_grid))

# COMMAND ----------

# MAGIC %md  > __DISCUSSION QUESTIONS:__  
# MAGIC  * _Compare the l1 and l2 penalty curves? Do they have a 'minimum' per se? How can we see in the graph that the regularization term doesn't depend on the bias value, only the coefficient?_
# MAGIC  * _Compare contrast the penalty terms to the OLS loss curve itself. (hint: pay careful attention to the z axis here)_
# MAGIC  * _How does the regularization parameter affect the shape of the penalty curves_
# MAGIC  * Predict what will happen to the OLS curve's shape if we add l1/l2 regularization. Then adjust the `REG_TYPE` to see it in action. Do the results match your expectation?
# MAGIC  * BONUS: what is the connection between these plots and Figure 6.7 on p222 of _Intro to Statistical Learning_.

# COMMAND ----------

# MAGIC %md # Regularization: why?
# MAGIC  As mentioned at the top, the basic motivation for using a penalize loss function is to avoid overfitting. But why do these specific penalties help with that? And why might we chose one or the other depending on the circumstance? Let's take a closer look at some simulated datasets that can help us build intuitions.

# COMMAND ----------

# MAGIC %md ### Demo 3: Ridge Regression
# MAGIC 
# MAGIC Consider the following 10 point dataset:

# COMMAND ----------

# create simulated dataset - RUN THIS CELL AS IS
np.random.seed(2019)
X = np.random.uniform(size = (10,9))
y = np.random.uniform(size = 10)

# plot the first 3 dimensions vs y
import demo7_helperFunc as hf
fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(20,4))
for i, ax in enumerate((ax1, ax2, ax3)):
    hf.plot2D(ax, X, y, xdim = i)
display(plt.show())    

# COMMAND ----------

# MAGIC %md Each of the plots above represents 2D projection of this dataset (one of the x dimensions plotted with y).
# MAGIC 
# MAGIC > __DISCUSSION QUESTIONS:__  
# MAGIC  * _Examine the code that generates this dataset. How many dimensions does the data have? (feel free to add an extra cell to take a look at \\(X\\) if you like)_ 
# MAGIC  * _Given what you observe in the plot and what you know about how the data were generated, how well would you expect a linear model to fit this data? what would it look like?_ 

# COMMAND ----------

# MAGIC %md The code below fits a regularized and unregularized model to the dataset generated above and plots a 2D view of the resulting models in the same projection as above.

# COMMAND ----------

from sklearn.linear_model import LinearRegression, Ridge
import demo7_helperFunc as hf
import numpy as np

#################### Demo Parameters #################### 
# TRY CHANGING THESE
REGULARIZATION_PARAMETER = 20
PROJECTION_DIMENSION = 1
nPOINTS = 10
nDim = 9

################### Penalized Loss Curves Demo #################### 
### data (same as above unless you change the number of points / dimensions)
np.random.seed(2019)
X = np.random.uniform(size = (nPOINTS,nDim))
y = np.random.uniform(size = nPOINTS)

### fit an OLS model and a ridge model using sklearn
ols = LinearRegression().fit(X,y)
ridge = Ridge(alpha = REGULARIZATION_PARAMETER).fit(X,y)



### plot comparison in 2D
labels=['OLS model in 2D', 'Ridge model in 2D']
display(hf.compare2Dprojections(X, y, xdim = PROJECTION_DIMENSION, models = [ols, ridge], labels = labels))

# COMMAND ----------

### print out the model parameters side by side
print('OLS   coefficients: ' + '  '.join([str(round(w,2)) for w in ols.coef_]))
print('Ridge coefficients: ' + '  '.join([str(round(w,2)) for w in ridge.coef_]))

# COMMAND ----------

# MAGIC %md 
# MAGIC > __DISCUSSION QUESTIONS:__  
# MAGIC  * _Compare the coefficients of the regularized & unregularized models. What do you notice? How does changing the regularization parameter affect the relationship between the OLS and Ridge coefficients_ 
# MAGIC  * _Given what you know about this dataset, is there anything surprising about the OLS model as plotted in the 2D projection? What might be going on here?_
# MAGIC  * _Compare the OLS and Ridge models based on their plots. Which model is overfitting?_
# MAGIC  * _Try a few different regularization parameter and projection dimensions... how does this affect the overfitting problem?_
# MAGIC  * _If time permits, try increasing the number of data points... how does this affect the overfitting problem?_
# MAGIC  
# MAGIC __Notes:__
# MAGIC * _This demo exercise was inspired by one of the examples presented in_ [these lecture notes by Wessel N. van Wieringen, VU University](https://www.statisticshowto.datasciencecentral.com/wp-content/uploads/2017/07/lecture-notes.pdf) _which the author has published under a creative commons license._
# MAGIC * _The sklearn package has very nice alternalte approach to illustrating how ridge regression reduces the variance of the linear model that gets fit. If time permits, check it out_ [here](https://scikit-learn.org/stable/auto_examples/linear_model/plot_ols_ridge_variance.html#sphx-glr-auto-examples-linear-model-plot-ols-ridge-variance-py).

# COMMAND ----------

# MAGIC %md ### Demo 4: Lasso Regression
# MAGIC Consider the following 10 point dataset.

# COMMAND ----------

# create simulated dataset - RUN THIS CELL AS IS
np.random.seed(2)
N = 10
x0 = np.random.uniform(size = N)
x1 = x0 + np.random.normal(0,0.1,N)
x2 = - x0 + np.random.normal(0,0.1,N)
x3thru9 = np.random.uniform(size = (N,7))
x0thru2 = np.array([x0,x1,x2]).T
X = np.hstack([x0thru2, x3thru9])
y = 5*x0 + np.random.normal(0,0.2,N)

# plot the first dimension vs y
import demo7_helperFunc as hf
fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1,5, figsize=(20,4))
for i, ax in enumerate((ax1, ax2, ax3, ax4, ax5)):
    hf.plot2D(ax, X, y, xdim = i)
display(plt.show())    

# COMMAND ----------

# MAGIC %md > __DISCUSSION QUESTIONS:__  
# MAGIC  * _Examine the code that generates this dataset. How many dimensions does the data have? (feel free to add an extra cell to take a look at \\(X\\) if you like)_ 
# MAGIC  * _Given what you observe in the plot and what you know about how the data were generated, how many of the features (dimensions) are going to be helpful in creating a linear model? which ones?_
# MAGIC  * _Based on what you saw in the last demo, what are the risks associated with a model that includes all 9 features?_

# COMMAND ----------

# MAGIC %md The code below fits ols, ridge and lasso models to the dataset generated above. Use the code provided to explore the effects of lasso regularization. Can you identify the key difference between lasso and ridge?

# COMMAND ----------

from sklearn.linear_model import LinearRegression, Ridge, Lasso
import demo7_helperFunc as hf

#################### Demo Parameters #################### 
# TRY CHANGING THESE
REGULARIZATION_PARAMETER = 0.05
PROJECTION_DIMENSION = 2
nPOINTS = 10
nDim = 9

################### Penalized Loss Curves Demo #################### 
### data (same as above)
np.random.seed(2)
x0 = np.random.uniform(size = nPOINTS)
x1 = x0 + np.random.normal(0,0.1,nPOINTS)
x2 = - x0 + np.random.normal(0,0.1,nPOINTS)
x3thruN = np.random.uniform(size = (nPOINTS,nDim - 3))
x0thru2 = np.array([x0,x1,x2]).T
X = np.hstack([x0thru2, x3thruN])
y = 5*x0 + np.random.normal(0,0.2,nPOINTS)

### fit an OLS model and a ridge model using sklearn
ols = LinearRegression().fit(X,y)
ridge = Ridge(alpha = REGULARIZATION_PARAMETER).fit(X,y)
lasso = Lasso(alpha = REGULARIZATION_PARAMETER).fit(X,y)


### plot comparison in 2D
labels=['OLS model in 2D', 'Lasso model in 2D', 'Ridge model in 2D']
display(hf.compare2Dprojections(X, y, xdim = PROJECTION_DIMENSION, models = [ols, lasso, ridge], labels = labels))

# COMMAND ----------

### print out the model parameters side by side
print('OLS   coefficients: ' + '  '.join([str(round(w,2)) for w in ols.coef_]))
print('Ridge coefficients: ' + '  '.join([str(round(w,2)) for w in ridge.coef_]))
print('Lasso coefficients: ' + '  '.join([str(round(w,2)) for w in lasso.coef_]))

# COMMAND ----------

# MAGIC %md > __DISCUSSION QUESTIONS:__  
# MAGIC  * _Examine the ols, lasso and ridge regression coefficients. How do these three models differ?_ 
# MAGIC  * _Based on the plots, does lasso seem like a better model? What might be the benefits to using lasso despite the seemingly worse fit?_
# MAGIC  * _Try changing the regularization parameter, what do you notice about the difference between "reasonable" looking regularization terms for ridge and lasso?_
# MAGIC  
# MAGIC  __Notes:__
# MAGIC  * _ISL is the best reference for understanding LASSO since Tibshirani was one of its main architects... however we also recommend the following_ [paper by Andrew Ng](https://icml.cc/Conferences/2004/proceedings/papers/354.pdf). _And these accompanying_ [lecture slides](http://cseweb.ucsd.edu/~elkan/254spring05/Hammon.pdf) _may also be of interest._

# COMMAND ----------

# MAGIC %md # Why don't we penalize the bias term?

# COMMAND ----------

# MAGIC %md <img src="https://github.com/kyleiwaniec/w261_assets/blob/master/images/wk7-demo/dont-shrink-the-intercept.png?raw=true">

# COMMAND ----------

# MAGIC %md # Converging Faster
# MAGIC 
# MAGIC Regularization addresses the issue of trying to make sure that our gradient descent will end up at a minimum point that represents a model that generalizes well. However we should also be concerned with optimizing the speed (i.e. number of steps/epochs/updates needed) at which GD converges. In general we do this by being smart about the learning rate.
# MAGIC 
# MAGIC #### Some Common Techniques to improve Gradient Descent's speed
# MAGIC 
# MAGIC 
# MAGIC 1. __Slow down as we get closer to a minimum__ 
# MAGIC > _GD does this naturally because the magnitude of the update vector will get smaller as the slopes get shallower around a minimum, however we can help it along strategically by reducing lambda as we perceive that our traing updates are getting smaller._ (look up: "__Adagrad__", "__Adadelta__", "__RMSprop__")
# MAGIC 
# MAGIC 2. __Using momentum__ : 
# MAGIC > _Take bigger updates when the direction of the next update is similar to the last one._  (look up: "__Nesterov accelerated gradient__").
# MAGIC 
# MAGIC 3. __Perform different size updates for each parameter__: 
# MAGIC > _This is particularly helpful for unnormalized data but can help in general as some of your parameters may not contribute much to improving the model performance_ (look up: "__Adam__" (adaptive moment estimation) and "__AdaMax__")
# MAGIC 
# MAGIC 3. __Train each parameter independently__ (Coordinate Descent): 
# MAGIC > _This is a newer approach that has interesting implications for parallelized SGD._
# MAGIC 
# MAGIC The following images come from this [article by Sebastian Ruder](http://ruder.io/optimizing-gradient-descent/) ([arxiv link])(https://arxiv.org/pdf/1609.04747.pdf) which is a good starting point if you want to dive into any of these GD variants more deeply.   
# MAGIC 
# MAGIC <img src="https://github.com/kyleiwaniec/w261_assets/blob/master/images/wk6-demo/saddle_point_evaluation_optimizers.gif?raw=true"/>
# MAGIC <img src="https://github.com/kyleiwaniec/w261_assets/blob/master/images/wk6-demo/contours_evaluation_optimizers.gif?raw=true"/>

# COMMAND ----------

# MAGIC %md # If all else fails
# MAGIC 
# MAGIC With more complicated Loss Functions we may run the risk of making a mistake when computing the gradient. In some cases it may even be impossible to compute an analytical gradient. In this section we'll look at a simple way to check that our gradient function is right. This "numerical approximation of the gradient" can be used as an alternative to perform gradient updates.
# MAGIC 
# MAGIC __(Numerical Approximation Method)__
# MAGIC 
# MAGIC $$
# MAGIC \nabla f(\mathbf{x}) =  \begin{bmatrix} \frac{\partial f}{\partial x\_1} \\\ \vdots \\\ \frac{\partial f}{\partial x\_m} \end{bmatrix}
# MAGIC $$
# MAGIC 
# MAGIC $$
# MAGIC \text{ where } m \text{ is the space dimension} 
# MAGIC $$
# MAGIC $$
# MAGIC \frac{\partial f}{\partial x\_1} = \lim\_{\alpha \rightarrow 0} \frac{f(x\_1 + \alpha, x\_2 \ldots x\_m) - f(x\_1, x\_2 \ldots x\_m)}{\alpha}
# MAGIC $$
# MAGIC 
# MAGIC For sufficiently small \\(\alpha\\) one can approximate partial derivative by simply throwing out the limit operator
# MAGIC 
# MAGIC $$
# MAGIC \frac{\partial f}{\partial x\_1} \approx \frac{f(x\_1 + \alpha, x\_2 \ldots x\_m) - f(x\_1, x\_2 \ldots x\_m)}{\alpha} = \left( \frac{\partial f}{\partial x\_1} \right)\_{\text{num}}
# MAGIC $$
# MAGIC 
# MAGIC Then the final approximation of the gradient is:
# MAGIC 
# MAGIC $$
# MAGIC \nabla f(\mathbf{x}) \approx \nabla\_{\text{num}\,\,} f(\mathbf{x}) = \begin{bmatrix} \left( \frac{\partial f}{\partial x\_1} \right)\_{\text{num}} \\\ \vdots \\\ \left( \frac{\partial f}{\partial x\_m} \right)\_{\text{num}} \end{bmatrix}
# MAGIC $$
# MAGIC 
# MAGIC The common way of measuring the difference between vectors is the following:
# MAGIC $$
# MAGIC \text{er} = \frac{\|\nabla f(\mathbf{x}) - \nabla\_{\text{num}\,\,}f(\mathbf{x})\|\_2^2}{\|\nabla f(\mathbf{x})\|\_2^2} = \frac{\sum\_{j=1}^{m}\left(\nabla^j f(\mathbf{x}) - \nabla^j\_{\text{num}\,\,}f(\mathbf{x})\right)^2}{\sum\_{j=1}^{m}\left(\nabla^j f(\mathbf{x})\right)^2}
# MAGIC $$
# MAGIC 
# MAGIC The code below uses a dataset from sklearn to compare this approximation to the true gradient using a dataset from sklearn & some helper classes in an attached file.

# COMMAND ----------

import demo7_homegrown as hg
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Load the data & take a look.
boston = load_boston()
#print(boston.DESCR) # -- uncomment to run

# Create data frame & test/train split.
X = pd.DataFrame(boston.data, columns=boston.feature_names)
y = boston.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# normalize
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# create & fit models with & without numerical approximation
model_homegrown = hg.BasicLinearRegressionHomegrown()
model_homegrown.fit(X_train, y_train, max_iter=100, alpha=0.001)
model_homegrown_check_grad = hg.TweakedLinearRegressionHomegrown()
model_homegrown_check_grad.fit(X_train, y_train, max_iter=100, alpha=0.001)

# plot the training curve
plt.figure(figsize=(10, 8))
plt.plot(model_homegrown.history["cost"], label="True Gradient")
plt.plot(model_homegrown_check_grad.history["cost"], label="Numerical Approximation")
plt.xlabel("Iteration")
plt.ylabel("RMSE")
plt.title("Linear Regression via Gradient Descent")
plt.legend()
display(plt.show())

# COMMAND ----------

# compare the RMSE & gradients for all 100 iterations
grad_num = np.array(model_homegrown_check_grad.history["grad_num"])
grad = np.array(model_homegrown_check_grad.history["grad"])
def relative_error(grad, grad_num):
    return np.sum((grad - grad_num) ** 2, axis=1) * 1. / np.sum(grad ** 2, axis=1)
def absolute_error(grad, grad_num):
    return np.sum((grad - grad_num) ** 2, axis=1) * 1.
plt.figure(figsize=(20, 8))
plt.suptitle("Numerical approximation of gradient quality")
plt.subplot(121)
plt.plot(relative_error(grad, grad_num))
plt.xlabel("Iteration")
plt.ylabel("Relative error")
plt.subplot(122)
plt.plot(absolute_error(grad, grad_num))
plt.xlabel("Iteration")
plt.ylabel("Absolute error")
display(plt.show())

# COMMAND ----------

