# Databricks notebook source
# MAGIC %md # wk6 Demo - Supervised Learning & Gradient Descent
# MAGIC __`MIDS w261: Machine Learning at Scale | UC Berkeley School of Information | Spring 2019`__
# MAGIC 
# MAGIC In Supervised Machine Learning we use labeled training data to learn a decision function (a.k.a 'model') and make evaluations about how well that decision function might perform when applied to new data. Of course the biggest factor that will determine the performance of your model is the quality of the data you train on. However another key challenge is the question of what models to consider & how to compare their performance so that you can choose the best one. Gradient Descent solves this challenge for a certain class of functions. By the end of this live session you should be able to:
# MAGIC * __... define__ the loss function for OLS Regression and its gradient.
# MAGIC * __... explain__ the relationship between model space and parameter space.
# MAGIC * __... recognize__ convex optimization problems and why they are desirable.
# MAGIC * __... describe__ the process of Gradient Descent & how it can be parallelized.

# COMMAND ----------

# MAGIC %md # Introduction
# MAGIC 
# MAGIC In today's demo, we'll use Linear Regression on a simple example in order to explore key topics related to distributed learning of parametric models. Broadly speaking, the supervised learning of a parametric model can be split into to two components:
# MAGIC 
# MAGIC 1. **Optimization Task (a.k.a. Learning)**: Given a strategy for making a prediction, return the specific parameters which guarantee the optimal prediction.   
# MAGIC 2. **Prediction Task**: Given an input vector, return an output value.
# MAGIC 
# MAGIC 
# MAGIC > __DISCUSSION QUESTION:__ _In the case of Linear Regression, which of the two tasks above are we most likely to want to parallelize? Why?_
# MAGIC 
# MAGIC 
# MAGIC OK, Let's start with a quick review of some notation you will have seen in w207. 
# MAGIC 
# MAGIC ## Notation Review
# MAGIC 
# MAGIC Linear Regression tackles the __prediction task__ by assuming that we can compute our output variable, \\(y\\), using a linear combination of our input variables. That is we assume there exist a set of **weights**, \\(\mathbf{w}\\), and a **bias** term, \\(\mathbf{b}\\), such that for any input \\(\mathbf{x}_j \in \mathbb{R}^m\\):
# MAGIC 
# MAGIC $$
# MAGIC y\_j = \displaystyle\sum\_{i=1}^{m}{w\_i\cdot x\_{ji} + b} \ \ \ \ \ \ (Eq \ 1.1)
# MAGIC $$
# MAGIC 
# MAGIC 
# MAGIC In vector notation, this can be written:
# MAGIC $$
# MAGIC y\_j = \displaystyle{\mathbf{w}^T\mathbf{x}\_{j} + b}
# MAGIC $$
# MAGIC Of course, this perfect linear relationship never holds over a whole dataset **\\(X\\)**, so Linear Regression attempts to fit (i.e. **learn**) the best line (in 1 dimension) or hyperplane (in 2 or more dimensions) to the data.  In the case of **ordinary least squares (OLS)** linear regression, best fit is defined as minimizing the Euclidean distances of each point in the dataset to the line or hyperplane.  These distances are often referred to as **residuals**. 

# COMMAND ----------

# MAGIC %md
# MAGIC <img src="https://github.com/kyleiwaniec/w261_assets/blob/master/images/wk6-demo/residual.png?raw=true"/>

# COMMAND ----------

# MAGIC %md The calculation of the average residual (a.k.a.**mean squared error, MSE**) over our test or training set allows us to measure how good a fit we've achieved. We call this function the **loss** or **objective** or **cost** function because our goal in the **optimization task** is to find the parameters which minimize it. (Ok, yes, _technically_ MSE is _not actually equal_ to the average residual but it is conceptually equivalent & guaranteed to have the same minimum.)
# MAGIC 
# MAGIC $$
# MAGIC f(\mathbf{w}, b) = \frac{1}{n}\sum\_{j=1}^{n}\left[(\mathbf{w}^T\mathbf{x}\_j + b) - y\_i\right]^2  \ \ \ \ \ \ (Eq \ 1.2)
# MAGIC $$
# MAGIC 
# MAGIC $$
# MAGIC n = |X\_{train}|
# MAGIC $$
# MAGIC 
# MAGIC For convenience, we sometimes choose to think of the bias \\(b\\) as weight \\(w\_{m+1}\\). To operationalize this, we'll augment our input vectors by setting \\(x\_{m+1}=1\\). This gives us a simpler way to write the loss function:
# MAGIC 
# MAGIC $$
# MAGIC \mathbf{x}' := \begin{bmatrix} \mathbf{x} \\\ 1 \end{bmatrix} , \quad \boldsymbol{\theta} := \begin{bmatrix} \mathbf{w} \\\ b \end{bmatrix}
# MAGIC $$
# MAGIC 
# MAGIC $$
# MAGIC f(\boldsymbol{\theta}) = \frac{1}{n}\sum\_{i=1}^{n}\left[ \boldsymbol{\theta}^T\cdot\mathbf{x}'\_i - y\_i\right]^2  \ \ \ \ \ \ (Eq \ 1.3)
# MAGIC $$
# MAGIC 
# MAGIC Machine Learning packages like `sklearn` and `tensorflow` take this one step further by representing the entire training set in a single matrix were each row is an input vector and each column represents a feature:
# MAGIC 
# MAGIC $$
# MAGIC X' = \begin{bmatrix} \mathbf{x'}\_1^{T}\\\ \vdots\\\ \mathbf{x'}\_n^{T} \end{bmatrix},\quad \mathbf{y} =  \begin{bmatrix} y_1\\\ \vdots\\\ y_n \end{bmatrix}
# MAGIC $$
# MAGIC 
# MAGIC $$
# MAGIC f(\boldsymbol{\theta}) = \frac{1}{n}\left\Vert\text{X}'\cdot \boldsymbol{\theta} - \mathbf{y}\right\Vert\_2^2  \ \ \ \ \ \ (Eq \ 1.4)
# MAGIC $$
# MAGIC 
# MAGIC As you see here, it is customary to write loss as a function of the parameters \\(\theta\\) (or equivalently \\(\mathbf{w}\\) and \\(b\\). However it is important to note that the MSE loss depends on both the parameters/weights  _and_ the data \\(X\\), we'll talk more about that later.

# COMMAND ----------

# MAGIC %md > __DISCUSSION QUESTIONS:__ 
# MAGIC * In equation 1.1 what do \\(x_{ji}\\), \\(w_i\\), and \\(\mathbf{w}\\) each represent?  
# MAGIC * In the asynch's version of the loss function \\(\alpha\\) and \\(\beta\\) appear as parameters... what do they represent? How are they captured in equations 1.2 and 1.3 respectively? 
# MAGIC * If we were computing loss over a really large data set what might be the arguments in favor / against using the augmented version of the loss function calculation?

# COMMAND ----------

# MAGIC %md #### A warning about OLS before we start:
# MAGIC 
# MAGIC Supervised learning models, especially interpratable ones, and especially linear/logistic regression, tend to get used for two different kinds of tasks: prediction and inference -- it is important to remember the difference between these two use cases. While it is practically possible to fit a linear model to any dataset and then use that model to make predictions... it is _not_ always fair to use the coefficients of your model to infer relationships (causal or otherwise) between your features and outcome variable. As you will rememeber from w203 and w207 if you are going to perform inference using OLS, your data should satisfy the following conditions:
# MAGIC 1. Residuals are homoscedastic - they have constant variance    
# MAGIC 1. Residuals are normaly distributed
# MAGIC 1. There is no multicolinearity - features are not correlated
# MAGIC 
# MAGIC __For more info see the reading ISL 3.1.3__
# MAGIC [ISL Slides](https://docs.google.com/presentation/d/1FuUe3jrFoCwA8XTkoSZBwz8xZ0oGJmpUGOmjWqJJAIc/edit#slide=id.p)

# COMMAND ----------

# MAGIC %md ## Notebook Set Up

# COMMAND ----------

# RUN THIS CELL AS IS
# This code snippet reads the user directory name, and stores is in a python variable.
# Next, it creates a folder inside your home folder, which you will use for files which you save inside this notebook.
username = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().apply('user')
userhome = 'dbfs:/user/' + username
print(userhome)
demo6_path = userhome + "/demo6/" 
demo6_path_open = '/dbfs' + demo6_path.split(':')[-1] # for use with python open()
dbutils.fs.mkdirs(demo6_path)

# COMMAND ----------

# RUN THIS CELL AS IS
# Here we'll create a test file, and use databricks utils to makes usre everything works as expected.
# You should see a result like: dbfs:/user/<your email>@ischool.berkeley.edu/demo4/test.txt
dbutils.fs.put(demo6_path+'test6.txt',"hello world",True)
display(dbutils.fs.ls(demo6_path))

# COMMAND ----------

# general imports
import sys
import csv
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import Image

# COMMAND ----------

dbutils.fs.ls("/mnt/mids-w261/modules")

# COMMAND ----------

import sys

# Add the path to system, local or mounted S3 bucket, e.g. /dbfs/mnt/<path_to_bucket>
sys.path.append('/dbfs/mnt/mids-w261/')
sys.path.append('/dbfs/mnt/mids-w261/modules')
sys.path.append('/dbfs/mnt/mids-w261/modules/linRegFunc.py')
sys.path.append('/dbfs/mnt/mids-w261/modules/helperFunc.py')

# COMMAND ----------

# import helper modules
import helperFunc
import linRegFunc

# OPTIONAL - uncomment to print helper file docstrings
print(helperFunc.__doc__)
print(linRegFunc.__doc__)

# COMMAND ----------

# MAGIC %md # A Small Example
# MAGIC 
# MAGIC We'll start with a small example of 5 2-D points:

# COMMAND ----------

dbutils.fs.put(demo6_path+"fivePoints.csv", 
"""
1,2
2,3
3,4
4,3
5,5
""", True)


# COMMAND ----------

# load data from file
points = np.genfromtxt(demo6_path_open+"fivePoints.csv", delimiter=',')

# COMMAND ----------

# MAGIC %md Here's what they look like next to a the simplest possible linear model:  \\(y = x\\)

# COMMAND ----------

# easy plotting with a helper function
display(helperFunc.plot2DModels(points, [[1,0]],['model'], title = 'Small Example'))

# COMMAND ----------

points

# COMMAND ----------

# MAGIC %md Looks reasonable, but its hard to gauge exactly how good a fit we have just by looking.
# MAGIC 
# MAGIC > __A TASK FOR YOU:__ Fill in the calculations below to compute the "Training Loss" for our data. These are easy and intuitive calculations that you will know from long-ago math classes... but instead of relying on your visual intuition, challenge yourself to think through these numbers in the context of our matrix equation for loss. Here it is again for your reference:
# MAGIC $$
# MAGIC f(\boldsymbol{\theta}) = \frac{1}{n}\sum\_{i=1}^{n}\left[ \boldsymbol{\theta}^T\cdot\mathbf{x}'\_i - y\_i\right]^2  \ \ \ \ \ \ (Eq \ 1.3)
# MAGIC $$

# COMMAND ----------

# MAGIC %md The parameter vector \\(\theta\\) for our initial line \\(y=x \\) is \\(\begin{bmatrix} ? \ \quad ? \ \end{bmatrix} \\)
# MAGIC          
# MAGIC The (augmented) data points \\(x_j\\) are:
# MAGIC \\( \begin{bmatrix} ? \\\ ? \\\ \end{bmatrix} \ \begin{bmatrix} ? \\\ ? \\\ \end{bmatrix} \ \begin{bmatrix} ? \\\ ? \\\ \end{bmatrix} \ \begin{bmatrix} ? \\\ ? \\\ \end{bmatrix}\ \begin{bmatrix} ? \\\ ? \\\ \end{bmatrix} \\)
# MAGIC 
# MAGIC 
# MAGIC Our loss calculations will be:
# MAGIC 
# MAGIC | \\(i\\) | \\(x\_i\\) |    \\(y\_i\\)  |  \\(\boldsymbol{\theta}^T\cdot\mathbf{x}'\_i\\) | \\(\left[ \boldsymbol{\theta}^T\cdot\mathbf{x}'\_i - y_i\right]^2\\) |
# MAGIC |:---:|:--------:|:---------------:|:-----------------------:|
# MAGIC |  -  | x |     true y   | predicted y |  squared residual       |
# MAGIC | 1     |-     |     -     |     -     |         -               |
# MAGIC | 2     |-     |     -     |     -     |         -               |
# MAGIC | 3     |-     |     -     |     -     |         -               |
# MAGIC | 4     |-     |     -     |     -     |         -               | 
# MAGIC | 5     |-     |     -     |     -     |         -               | 
# MAGIC 
# MAGIC  The training loss \\(f(\boldsymbol{\theta})\\) for this data and these weights is: _______

# COMMAND ----------

# MAGIC %md 

# COMMAND ----------

# Run this cell to confirm your Hand Calculations
X = helperFunc.augment(points)[:,:-1]
y = points[:,-1]
print("Loss:", linRegFunc.OLSLoss(X, y,[1,0]))

# COMMAND ----------

# MAGIC %md  > __DISCUSSION QUESTIONS:__ 
# MAGIC  * _What parts of this computation could be parallelized? What, if any, aggregation has to happen at the end?_ 
# MAGIC  * _What key-value format, partitioning, sorting would help? Could you use a combiner?_ 
# MAGIC  * _In addition to the data stream, what other information would your map or reduce tasks need access to?_ 

# COMMAND ----------

# MAGIC %md ## Demo: Random Parameter Search.

# COMMAND ----------

# MAGIC %md Ok, so we know the model looks ok and we know its loss is \\(0.8\\) but is that any good? A naive approach to "learning" a Linear Model might be to randomly generate a few more models and then pick the model with the lowest loss. Let's try it.

# COMMAND ----------

import helperFunc, linRegFunc

#################### Demo Parameters #################### 
# TRY CHANGING THESE & SEE HOW IT AFFECTS OUR SEARCH
NUM_MODELS = 10
PARAM_RANGE = [-5,5]

#################### Random Search Demo ####################
# Load & pre-process data
points = np.genfromtxt(demo6_path_open+"fivePoints.csv", delimiter=',')
X = helperFunc.augment(points)[:,:2]
y = points[:,1]

# "Training"
models = [[0,1]]
names = ["INIT - Loss: 0.8"]
best = {'loss':0.8, 'W': [1,0]}
for idx in range(NUM_MODELS):
    # initialize a random weight vector w/ values in specified range
    W = np.random.uniform(PARAM_RANGE[0],PARAM_RANGE[1], size=(2))
    # compute loss & store for plotting
    loss = linRegFunc.OLSLoss(X, y, W)
    models.append(W)
    names.append("model%s - Loss: %.2f" % (idx, loss))
    # track best model
    if loss < best['loss']:
        best['loss'] = loss
        best['W'] = W
        
# Display Results
print(f"Best Random Model: {best['W']}, Loss: {best['loss']}")
display(helperFunc.plot2DModels(points, models, names, "A Random Approach."))

# COMMAND ----------

# MAGIC %md So, that was pretty poor. One idea would be to run a lot more iterations.
# MAGIC 
# MAGIC > __DISCUSSION QUESTION:__ 
# MAGIC * _To what extent could parallelization help us redeem this approach? What exactly would you parallelize?_

# COMMAND ----------

# MAGIC %md ## Demo: Systematic Brute Force.

# COMMAND ----------

# MAGIC %md For obvious reasons a more systematic approach is desirable. Instead of randomly guessing, let's use what we know to search an appropriate section of the the model space.
# MAGIC 
# MAGIC We can tell from the data that the linear model should probably have a fairly shallow positive slope and a positive intercept between 0 and 2. So lets initialize every possible combination of weights in that range up to a granularity of, say 0.2, and compute the loss for each one.

# COMMAND ----------

import helperFunc, linRegFunc

#################### Demo Parameters #################### 
# TRY CHANGING THESE & SEE HOW IT AFFECTS OUR SEARCH
W0_MIN = 0
W0_MAX = 2
W0_STEP = 0.2

W1_MIN = 0
W1_MAX = 2
W1_STEP = 0.2

#################### Grid Search Demo #################### 
### Load & Pre-process Data
points = np.genfromtxt(demo6_path_open+"fivePoints.csv", delimiter=',')
X = helperFunc.augment(points)[:,:2]
y = points[:,1]

### "Training" 
# create a model for each point in our grid
grid = np.mgrid[W0_MIN:W0_MAX:W0_STEP,W1_MIN:W1_MAX:W1_STEP]
size = int(np.product(grid.shape)/2)
models = grid.reshape(2,size).T
# compute loss for each model
loss = []
for W in models:
    loss.append(linRegFunc.OLSLoss(X,y,W))
    
### Display Results
print(f"Searched {size} models...")
print(f"Best model: {models[np.argmin(loss)]}, Loss: {min(loss)}")
display(helperFunc.plotErrorSurface(points,models,loss))

# COMMAND ----------

# MAGIC %md > __DISCUSSION QUESTIONS:__ 
# MAGIC * _When we think about scaling up, is this still a better approach than guessing? How could it be parallelized?_ 
# MAGIC * _What would change about this approach if we had higher dimension data?_
# MAGIC * _In practice, when we're training Linear Models why don't we just look at the error surface and identify the lowest point?_
# MAGIC * _What about if we're training other kinds of models?_  

# COMMAND ----------

# MAGIC %md # Parameter Space, Gradients, and Convexity
# MAGIC 
# MAGIC As suggested by the systematic search demo, when we train parametric models we tend to switch back and forth between two different ways of visualizing our goal.

# COMMAND ----------

# MAGIC %md
# MAGIC <img src="https://github.com/kyleiwaniec/w261_assets/blob/master/images/wk6-demo/Gradient_Descent.gif?raw=true"/>

# COMMAND ----------

# MAGIC %md * When we look at a model next to our data represented in the Problem Domain Space, it is natural to think about loss as a measure of _**how far off the data are from our model**_. In other words, this visual suggests loss is a function of the training data \\(X\\).
# MAGIC * By contrast, looking at an error surface plotted in Model Parameter Space, we intuitively see loss as an indicator of _**how far off the current model is from the optimal model**_. In other words, this view helps us think of loss as a function of the parameters \\(\theta\\).
# MAGIC 
# MAGIC Of course in one sense, this distinction is just a matter of sematics. As we saw in equations 1.2, 1.3 and 1.4, MSE loss depends on _both_ the data and the parameters. However, in the context of 'inventing' ways to train a model, this distinction is a useful one. If we think of the data as fixed and focus on how loss varies _with respect to the parameters_, then we can take advantage of a little theory to speed up our search for the optimal parameters.

# COMMAND ----------

# MAGIC %md ### Optimization Theory ... a short digression
# MAGIC 
# MAGIC Calculus gives us the simple solution to optimizing a real function. The **First Order Conditions** (a.k.a. 'first derivative rule') says that the maximum or minimum of an unconstrained function must occur at a point where the first derivative = 0. In higher dimensions we extend this rule to talk about a **gradient** vector of partial derivatives which all must equal 0. 
# MAGIC 
# MAGIC When the first order partial derivatives are equal to zero, then we know we are at a local maximum or minimum of the real function.  But which one is it?  In order to tell, we must take the second derivatives of the real function.  If the second derivatives are positive at that point, then we know we are at a minimum.  If the second derivatives are negative, then we know we are at a maximum.  These are the **second order conditions.**

# COMMAND ----------

# MAGIC %md **Convex Optimization** is the lucky case where we know that the second derivatives never change sign. There are lots of complicated loss functions for which we can't easily visualize the error surface but for which we _can_ prove mathematically that this 2nd order condition is met. If this is the case, then we can think of the suface as _always curving up_ or _always curving down_ which guarantees that any minimum we reach will be an absolute minimum. More powerfully still, this result can be shown to _also_ apply to a class of "pseudo-convex" functions - functions whose second derivative might not be well defined, but satisfy certain conditions that allow us to guarantee convergence.

# COMMAND ----------

# MAGIC %md > __DSICUSSION QUESTIONS:__ 
# MAGIC * _In the case of Linear Regression performed on data \\(X \in \mathbb{R}^m\\), how many dimensions does the gradient vector have? What do each of the values in this vector represent visually?_
# MAGIC * _If we are systematically searching the parameter space for a lowest point, why might it be useful to know that our loss function is convex?_ 
# MAGIC * _Condider the loss curves illustrated below -- do these illustrations represent problem space or parameter space? which ones are convex?_

# COMMAND ----------

# MAGIC %md 
# MAGIC <img src="https://github.com/kyleiwaniec/w261_assets/blob/master/images/wk6-demo/loss_01_fabianp.png?raw=true"  height="250" width="250"/>
# MAGIC <img src="https://github.com/kyleiwaniec/w261_assets/blob/master/images/wk6-demo/loss_02_algorithmia.png?raw=true"  height="250" width="250"/>
# MAGIC <img src="https://github.com/kyleiwaniec/w261_assets/blob/master/images/wk6-demo/loss_03_mathworks.png?raw=true"  height="250" width="250"/>
# MAGIC 
# MAGIC Sources: [first image](http://fa.bianp.net/blog/2014/surrogate-loss-functions-in-machine-learning/) | [second image](https://blog.algorithmia.com/introduction-to-loss-functions/) | [third image](https://fr.mathworks.com/help/gads/example-finding-the-minimum-of-a-function-using-the-gps-algorithm.html)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Closed form (analytical) solution
# MAGIC 
# MAGIC To take advantage of these lessons from Optimization Theory, we'll start by taking the derivative of the loss function with respect to the parameters \\(\boldsymbol{\theta}\\). Recall the matrix formulation of our loss function:
# MAGIC 
# MAGIC $$
# MAGIC f(\boldsymbol{\theta}) = \frac{1}{n}\sum\_{i=1}^{n}\left[ \boldsymbol{\theta}^T\cdot\mathbf{x}'\_i - y\_i\right]^2  \ \ \ \ \ \ (Eq \ 1.3)
# MAGIC $$
# MAGIC 
# MAGIC We can apply the sum and chain derivation rules to compute the gradient:
# MAGIC 
# MAGIC $$
# MAGIC \nabla\_{\boldsymbol{\theta}} f(\boldsymbol{\theta}) = \frac{2}{n}\,\sum\_{i=1}^{n}\left[ \boldsymbol{\theta}^T\cdot\mathbf{x}'\_i - y_i\right] \cdot \mathbf{x}'\_i  \ \ \ \ \ \ (Eq \ 3.1)
# MAGIC $$
# MAGIC 
# MAGIC 
# MAGIC We _could_ now set this equation equal to \\(0\\) and then solve for \\(\boldsymbol{\theta}\\)... 
# MAGIC 
# MAGIC Rewriting this in vector notation we have:
# MAGIC 
# MAGIC $$
# MAGIC 0 = \frac{2}{n}\left\Vert\text{X}'\cdot \boldsymbol{\theta} - \mathbf{y}\right\Vert\cdot x 
# MAGIC $$
# MAGIC 
# MAGIC We can then solve for \\(\theta\\):
# MAGIC $$
# MAGIC \theta^* = (X^TX)^{-1}X^Ty
# MAGIC $$
# MAGIC 
# MAGIC _NOTE: for the sake of time, we'll omit the derivation here, but there are lots of useful resources for this online - here's one from Khan Academy: https://www.youtube.com/watch?v=MC7l96tW8V8)_

# COMMAND ----------

# MAGIC %md > __DSICUSSION QUESTIONS:__ 
# MAGIC * _In general (i.e. beyond Linear Regression) if finding the ideal parameters \\(\theta\\), is as simple as solving the equation \\(f'(\theta)=0\\), why don't we always train our models by solving that equation?_ 

# COMMAND ----------

# MAGIC %md                                                                                   
# MAGIC ## Numerical Approximation - Newton Raphson
# MAGIC 
# MAGIC <img src="https://github.com/kyleiwaniec/w261_assets/blob/master/images/wk6-demo/newton-raphson.png?raw=true">   
# MAGIC source: https://www.youtube.com/watch?v=cOmAk82cr9M
# MAGIC 
# MAGIC 
# MAGIC Now we want to find the root of our derivative of the loss function with respect to \\(\theta\\). We can use Newton's method to iteratively approximate:
# MAGIC 
# MAGIC $$
# MAGIC f'(\theta) = 0
# MAGIC $$
# MAGIC 
# MAGIC Using the formula:
# MAGIC 
# MAGIC $$
# MAGIC \theta\_n = \theta\_{n-1} - \frac{f'(\theta)}{f''(\theta)}
# MAGIC $$
# MAGIC 
# MAGIC And in higher dimetions:
# MAGIC $$
# MAGIC \theta\_n = \theta\_{n-1} - \frac{\nabla\_{\boldsymbol{\theta}} f(\boldsymbol{\theta})}{H(\theta)}
# MAGIC $$
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC __DISCUSSION QUESTIONS:__
# MAGIC * What's the problem with this method?

# COMMAND ----------

# MAGIC %md ## Demo: Gradient Descent

# COMMAND ----------

# MAGIC %md
# MAGIC $$
# MAGIC \theta\_n = \theta\_{n-1} - \frac{1} {H(\theta)} \cdot \nabla\_{\boldsymbol{\theta}}f(\boldsymbol{\theta})
# MAGIC $$
# MAGIC 
# MAGIC We can substitue \\(\frac{1} {H(\theta)}\\) for \\(\eta\\) and VOILA! smells like Gradient Descent.

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC The strategy of **Gradient Descent** is to start somewhere random in the Model Parameter Space and then move down the error surface to find a minimum point with the optimal parameters for our training data. Its ingeniousness is that we can do this without actually knowing the full shape of the error surface. Think of it like walking down a hill while blindfolded. You test each direction to see which way is down, then take a little step in that direction and repeat the process until you can't feel any more 'down' to go. The 'size' of our steps is controled by a hyperparameter, \\(\eta\\), the **learning rate**. The whole process can be summarized in 3 steps:
# MAGIC 1. Initialize the parameters \\(\theta\\).
# MAGIC 2. Compute the gradient \\(\nabla\_{\boldsymbol{\theta}} f(\boldsymbol{\theta})\\).
# MAGIC 3. Update the parameters: \\(\theta\_{\text{new}} = \theta\_{\text{old}} - \eta \cdot \nabla\_{\boldsymbol{\theta}} f(\boldsymbol{\theta}) \\) .  
# MAGIC 
# MAGIC We repeat these steps until we reach a stopping criteria.

# COMMAND ----------

# MAGIC %md > __A TASK FOR YOU:__ Compute one Gradent Descent update step for the small example from Part 2. 
# MAGIC Recall that our initial parameters were:
# MAGIC $$ \boldsymbol{\theta} = \begin{bmatrix} 1 \ \quad 0 \ \end{bmatrix}$$  
# MAGIC > For your convenience the augmented input data vectors are already entered in the table below:
# MAGIC 
# MAGIC Hand Calculations:
# MAGIC 
# MAGIC |  \\(x\_j'\\)  | \\(y\_j\\) |   \\(\boldsymbol{\theta}^T\cdot\mathbf{x}'\_j\\) | \\(\left[ \boldsymbol{\theta}^T\cdot\mathbf{x}'\_j - y\_j\right]\cdot\mathbf{x}'\_j\\) |
# MAGIC |:----:|:-----:|:----------------:|:------------------------:|
# MAGIC |  input   | true \\(y\\)   |   predicted \\(y\\)   |  gradient  component for \\(x\_j\\)       |
# MAGIC | \\( \begin{bmatrix} 1 \\\ 1 \\\ \end{bmatrix}\\)   |  2   |     _             |    _        
# MAGIC | \\( \begin{bmatrix} 2 \\\ 1 \\\ \end{bmatrix}\\)   |  3   |     _             |    _
# MAGIC | \\( \begin{bmatrix} 3 \\\ 1 \\\ \end{bmatrix}\\)   |  4   |     _             |    _
# MAGIC | \\( \begin{bmatrix} 4 \\\ 1 \\\ \end{bmatrix}\\)   |  3   |     _             |    _
# MAGIC | \\( \begin{bmatrix} 5 \\\ 1 \\\ \end{bmatrix}\\)   |  5   |     _             |    _

# COMMAND ----------

# MAGIC %md The gradient \\(\nabla_{\boldsymbol{\theta}} f(\boldsymbol{\theta})\\) for this data and these weights is: [? ?]
# MAGIC 
# MAGIC If \\(\eta = 0.1\\) the update for this step will be: [? ?]
# MAGIC 
# MAGIC The new parameters will be \\(\theta_{\text{new}}=\\) [? ?]  

# COMMAND ----------

# MAGIC %md  > __DISCUSSION QUESTIONS:__  
# MAGIC  * _How would you go about parallelizing this calculation? What would the mapper do, what would the reducers do? What key-value structure, sorting, partitioning, etc would you use?_ 
# MAGIC  * _How do the computational demands of performing GD compare to the task of computing the loss?_ 

# COMMAND ----------

# MAGIC %md __Run this demo to confirm your hand calculations & examine a few more GD steps.__

# COMMAND ----------

import numpy as np
import helperFunc, linRegFunc

#################### Demo Parameters #################### 
# TRY CHANGING THESE & SEE HOW IT AFFECTS OUR SEARCH
N_STEPS = 5
LEARNING_RATE = 0.1
ORIGINAL_MODEL = [1,0]
SHOW_CONTOURS = True

################### Gradient Update Demo #################### 
### Load & Pre-process Data
points = np.genfromtxt(demo6_path_open+"fivePoints.csv", delimiter=',')
X = helperFunc.augment(points)[:,:2]
y = points[:,1]

### Perform GD Update & save intermediate model performance
models, loss = linRegFunc.GDUpdate(X, y, N_STEPS,
                                   ORIGINAL_MODEL, 
                                   LEARNING_RATE, 
                                   verbose = True)

### Display Results
print(f"\nSearched {len(models)} models...")
print(f"Best model: {models[np.argmin(loss)]}, Loss: {loss[np.argmin(loss)]}")
display(linRegFunc.plotGDProgress(points, models, loss, show_contours = SHOW_CONTOURS))

# COMMAND ----------

# MAGIC %md  > __DISCUSSION QUESTIONS:__  
# MAGIC  * _Look closely at the loss for each model, what problem do you notice?_ 
# MAGIC  * _Use the Model Parameter Space view to explain why this problem might be occurring._ __HINT:__ Try `SHOW_CONTOURS = True`. _Based upon your insights, propose a solution to this problem._
# MAGIC  * _When performing GD 'in the wild' will we be able to visualize the error surface (eg. using contour lines, heatmaps or 3D plots)?_

# COMMAND ----------

# MAGIC %md ## Demo : Stochastic Gradient Descent
# MAGIC 
# MAGIC 
# MAGIC In full Gradient Descent (what we did above) we do a descent step only after the calculation of the gradient over the whole set of data. That means we only update the weight vector once each **epoch** (pass over the data) thus making one small but “good” step towards the minimum. However since gradient descent is an interative algorithm that requires many updates to find the minimum, with large datasets, waiting to process every record before performing an update can result in a slow and computationaly costly training process. 
# MAGIC 
# MAGIC The alternatives are:
# MAGIC 1. **Stochastic GD** -- compute the gradient _with respect to a single point at a time_ and update the entire weight vector after each record. By the time we have seen the whole data set, we will have made N (num of observations), perhaps “not so good”, steps with a general trend towards the minimum. SGD will “zig-zag” towards the minimum and eventually oscillate around the minimum but never converge. The advantage of SGD is that we can make progress at every example - if the data is very large, we may only need 1 pass over the whole dataset.
# MAGIC 2. **Mini-batch GD** -- compute the gradient _with respect to a small **batch** (size of \\(B\\)) of points at a time_ and update the entire weight vector after each batch. If we are smart about shuffling the data, this can reduce the “zig-zaging” because the points in a batch will temper each other's influence. This is especially advantageous for noisy data where a single point might result in a gradient update that is dramatically in the wrong direction for the rest of the data. For this reason, MBGD can potentially finish even faster than SGD.
# MAGIC 
# MAGIC 
# MAGIC Other than the denominator in front, the loss function for SGD/MBGD should look very familiar (note that SGD is basically just the special case where \\(B = 1\\)):
# MAGIC 
# MAGIC 
# MAGIC $$
# MAGIC \nabla f(\boldsymbol{\theta}) \approx \nabla\_{batch}\,\, f(\boldsymbol{\theta}) = \frac{2}{B}\sum\_{i=1}^{B}\left(\boldsymbol{\theta}^T \cdot\mathbf{x}'\_{a\_i} - y\_{a\_i}\right)\cdot \mathbf{x}'\_{a\_i} \ \ \ \ \ \ \ \ \ \ \ (Eq \ 3.2)
# MAGIC $$
# MAGIC 
# MAGIC 
# MAGIC where \\(a_i\\) is an array of indices of objects which are in this batch. After obtaining this gradient we do a descent step in this approximate direction and proceed to the next stage of batch descent.
# MAGIC 
# MAGIC > __A TASK FOR YOU:__ Perform 5 update steps of Stochastic Gradient Descent with batchsize = \\(1\\) on our small data set. 
# MAGIC Recall that our initial parameters were:
# MAGIC $$ \boldsymbol{\theta} = \begin{bmatrix} 1 \ \quad 0 \ \end{bmatrix}$$  
# MAGIC > ... and we used a learning rate of \\(\boldsymbol{\eta} = 0.1\\)
# MAGIC 
# MAGIC (\\(\eta\\) is pronounced 'eh-ta', sometimes we also use \\(\alpha\\), "apha" to denote learning rate, the two are equivalent)
# MAGIC 
# MAGIC Hand Calculations:
# MAGIC 
# MAGIC |  \\(x_j'\\)  | \\(y\_j\\) |   \\(\boldsymbol{\theta}\cdot\mathbf{x}'\_j\\) | \\(\frac{2}{B}\left[ \boldsymbol{\theta}^T\cdot\mathbf{x}'\_j - y\_j\right]\cdot\mathbf{x}'\_j\\) | \\(\eta \nabla\_{\boldsymbol{\theta}} f\\) | \\(\boldsymbol(\theta) - \eta \nabla\_{\boldsymbol{\theta}} f \\) |
# MAGIC |:----:|:-----:|:----------------:|:------------------------:|:--------------:|:-----------:|
# MAGIC |  input   | true \\(y   |   predicted \\(y   | gradient for this 'batch' | update | new parameters|
# MAGIC | \\( \begin{bmatrix} 1 \\\ 1 \\\ \end{bmatrix}\\)   |  2   |        -          |     -|     -|     -    
# MAGIC | \\( \begin{bmatrix} 3 \\\ 1 \\\ \end{bmatrix}\\)   |  4   |        -          |     -|     -|     -
# MAGIC | \\( \begin{bmatrix} 5 \\\ 1 \\\ \end{bmatrix}\\)   |  5   |        -          |     -|     -|     -
# MAGIC | \\( \begin{bmatrix} 4 \\\ 1 \\\ \end{bmatrix}\\)   |  3   |        -          |     -|     -|     -
# MAGIC | \\( \begin{bmatrix} 2 \\\ 1 \\\ \end{bmatrix}\\)   |  3   |        -          |     -|     -|     -

# COMMAND ----------

# MAGIC %md  > __DISCUSSION QUESTIONS:__  
# MAGIC  * _How does this result compare to our result from the hand calculations in the last section? What implications does this have for our quest to find the optimal paramters?_ 
# MAGIC  * _How will parallelizing Stoichastic Gradient Descent be similar/different to parallelizing regular GD?_ 

# COMMAND ----------

import helperFunc, linRegFunc

#################### Demo Parameters #################### 
# TRY CHANGING THESE & SEE HOW IT AFFECTS OUR SEARCH
N_STEPS = 5
BATCHSIZE = 1
LEARNING_RATE = 0.1
ORIGINAL_MODEL = [1,0]
SHOW_CONTOURS = True

################### Stoichastic GD Demo #################### 
### Load & Pre-process Data
points = np.genfromtxt(demo6_path_open+"fivePoints.csv", delimiter=',')
X = helperFunc.augment(points)[:,:2]
y = points[:,1]

### Perform SGD Updates & save intermediate model performance
models, loss = linRegFunc.SGDUpdate(X, y, N_STEPS,
                                    BATCHSIZE,
                                    ORIGINAL_MODEL, 
                                    LEARNING_RATE, 
                                    verbose = False)

### Display Results
print(f"\nSearched {len(models)} models..." %())
print(f"Best model: {models[np.argmin(loss)]}, Loss: {loss[np.argmin(loss)]}")
display(linRegFunc.plotGDProgress(points, models, loss, show_contours = SHOW_CONTOURS))

# COMMAND ----------

# MAGIC %md > __DISCUSSION QUESTIONS:__ 
# MAGIC * _At first glance does this seem to work as well as regular gradient descent? Why might our initial impression be deceiving?_ 
# MAGIC * _Does adjusting the batchsize and/or learning rate fix the problem that we're seeing?_
# MAGIC * _What do you notice about the direction of the first 3 updates? From the perspective of the first three points, what should our line look like?_
# MAGIC * _How does the scale of our data can impact the direction of our updates & time to convergence?_

# COMMAND ----------

# MAGIC %md ## Selecting the learning rate
# MAGIC 
# MAGIC As you saw in the earlier eaxmples, increasing the learning rate reduces the number of iterations with which GD converges. However, care must be taken not to select a learning rate so high that the algorithm ends up overshooting the minimum and never converging at all!
# MAGIC 
# MAGIC A standard approach to selecting the "best" learning rate is to do a grid search using a range of rates, say, `[0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3]`, then plotting the losses at each iteration to see which converges fastest.
# MAGIC <img src="https://github.com/kyleiwaniec/w261_assets/blob/master/images/wk6-demo/loss-iterations.png?raw=true"/>

# COMMAND ----------

# MAGIC %md Taking this a step further (or even a few steps), a lot of work has been done on ways to "select" the learning rate to get GD to both converge faster, and perform better on unseen data. These variants of the GD algorithm include ideas like ADAM, AMSGrad, etc.. They work by dynamically adjusting the learning rate based on both recent gradients as well as time steps. A detailed discussion of these efforts are beyond the scope of this course. Below is an excellent overview of the most popular GD optimizers.

# COMMAND ----------

# MAGIC %md __An overview of gradient descent optimization algorithms__ by Sebastian Ruder     
# MAGIC http://ruder.io/optimizing-gradient-descent/

# COMMAND ----------

# MAGIC %md
# MAGIC <img src="https://github.com/kyleiwaniec/w261_assets/blob/master/images/wk6-demo/saddle_point_evaluation_optimizers.gif?raw=true"/>
# MAGIC <img src="https://github.com/kyleiwaniec/w261_assets/blob/master/images/wk6-demo/contours_evaluation_optimizers.gif?raw=true"/>

# COMMAND ----------

# MAGIC %md See also: https://towardsdatascience.com/10-gradient-descent-optimisation-algorithms-86989510b5e9

# COMMAND ----------

# MAGIC %md __For more info, here are a few of rabbit holes:__
# MAGIC > https://arxiv.org/pdf/1707.00424.pdf      
# MAGIC > https://papers.nips.cc/paper/4390-hogwild-a-lock-free-approach-to-parallelizing-stochastic-gradient-descent.pdf   
# MAGIC > http://papers.nips.cc/paper/4006-parallelized-stochastic-gradient-descent.pdf     
# MAGIC > https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/distr_mini_batch.pdf   

# COMMAND ----------

# MAGIC %md ### That's it for today! 
# MAGIC 
# MAGIC #### Next week we will discuss...
# MAGIC * __L1 and L2 Regularization__ 
# MAGIC * __Common GD variants__
# MAGIC * __What to do if you can't compute a gradient for your loss function.__
# MAGIC * __Logistic Regression & classification__

# COMMAND ----------

