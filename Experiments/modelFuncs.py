#!/usr/bin/env python

# modelFuncs.py: functions to train and evaluate LVMs
# Author: Nishanth Koganti
# Date: 2018/06/04


# import the modules
import sys
import GPy
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

################################################################################
# Functions for model training
################################################################################

# function to train pca model
def pcaModel(data, expName, qDim=10):
    # apply PCA
    model = PCA(n_components=qDim)
    model.fit(data)

    # return model variables
    return model

# function to train gplvm model
def gplvmModel(data, expName, qDim=10, initIters=500, trainIters=2000, 
               init='pca', ard=True):
    # model train variables
    SNR = 100

    # initialize latent space
    if init == 'pca':
        pca = PCA(n_components=qDim)
        pca.fit(data)

        # get latent points and scales
        xData = pca.transform(data)
        scales = pca.explained_variance_ratio_
        scales = scales/scales.max()
    elif init == 'rand':
        xData = np.random.normal(0, 1, (data.shape[0],qDim))
        scales = np.ones(qDim)
    
    # setting up the kernel
    if ard:
        lScales = 1.0/scales
    else:
        lScales = 1.0
    kernel = GPy.kern.RBF(qDim, variance=1., lengthscale=lScales, ARD=ard)

    # initialize BGPLVM model
    model = GPy.models.GPLVM(data, input_dim=qDim, kernel=kernel,
                             X=xData, name=expName)

    # train with constriants
    if initIters > 0:
        var = model.Y.var()
        model.kern.variance.fix(var)
        #model.kern.lengthscale.fix(lScales)
        model.Gaussian_noise.variance.fix(var/SNR)
        model.optimize(messages = True, max_iters = initIters)
        model.unconstrain_fixed()
        
    # train without constraints
    if trainIters > 0:
        model.optimize(messages = True, max_iters = trainIters)

    # return model
    return model

# function to train bgplvm model
def bgplvmModel(data, expName, qDim=10, nInducing=50, initIters=500, 
                trainIters=2000, init='pca', ard=True):
    # model train variables
    SNR = 100

    # initialize latent space
    if init == 'pca':
        pca = PCA(n_components=qDim)
        pca.fit(data)

        # get latent points and scales
        xData = pca.transform(data)
        scales = pca.explained_variance_ratio_
        scales = scales/scales.max()
    elif init == 'rand':
        xData = np.random.normal(0, 1, (data.shape[0],qDim))
        scales = np.ones(qDim)
        
    # setting up the kernel
    if ard:
        lScales = 1.0/scales
    else:
        lScales = 1.0

    kernel = GPy.kern.RBF(qDim, variance=1., lengthscale=lScales, ARD=ard)
        
    # initialize BGPLVM model
    model = GPy.models.BayesianGPLVM(data, input_dim=qDim, num_inducing=nInducing,
                                     kernel=kernel, X=xData)

    # train with constriants
    if initIters > 0:
        var = model.Y.var()
        model.kern.variance.fix(var)
        #model.kern.lengthscale.fix(lScales)
        model.Gaussian_noise.variance.fix(var/SNR)
        model.optimize(messages = True, max_iters = initIters)
        model.unconstrain_fixed()
    
    # train without constraints
    if trainIters > 0:
        model.optimize(messages = True, max_iters = trainIters)

    # return model
    return model

################################################################################
# Functions for test inference
################################################################################

def pcaInference(model, trainInput, testInput):
    # get latent space plot parameters
    testData = model.transform(testInput)
    trainData = model.transform(trainInput)
    return trainData, testData

def gplvmInference(model, trainInput, testInput):
    # get latent space plot parameters
    qDim = model.X.shape[1]
    testData = np.zeros((testInput.shape[0], qDim))
    trainData = np.zeros((trainInput.shape[0], qDim))

    for n in range(trainInput.shape[0]):
        # infer latent position
        xTrain, _ = model.infer_newX(np.atleast_2d(trainInput[n,:]), optimize=True)

        # update parameter
        trainData[n,:] = xTrain
        sys.stdout.write('.')
    sys.stdout.write('\n')

    for n in range(testInput.shape[0]):
        # infer latent position
        xTest, _ = model.infer_newX(np.atleast_2d(testInput[n,:]), optimize=True)

        # update parameter
        testData[n,:] = xTest
        sys.stdout.write('.')
    sys.stdout.write('\n')

    return trainData, testData

################################################################################
# Function to evaluate trained models
################################################################################

# function to compute test and training reconstruction error
def pcaError(model, trainData, testData):
    # get number of test and training samples
    nTest = testData.shape[0]
    nTrain = trainData.shape[0]
    nDims = trainData.shape[1]

    # create output variables
    testLatent = model.transform(testData)
    trainLatent = model.transform(trainData)

    testOut = model.inverse_transform(testLatent)
    trainOut = model.inverse_transform(trainLatent)

    testError = np.zeros((nTest,1))
    trainError = np.zeros((nTrain,1))

    # compute reconstruction error
    testError = np.sqrt(((testData - testOut)**2).sum(axis=1)/nDims)
    trainError = np.sqrt(((trainData - trainOut)**2).sum(axis=1)/nDims)

    # return the variables
    return (trainLatent, testLatent, trainOut, testOut, trainError, testError)

# function to compute test and training reconstruction error
def gplvmError(model, trainData, testData, wThresh=0.05, minDim=3, ard=True):
    # get number of test and training samples
    nTest = testData.shape[0]
    nTrain = trainData.shape[0]

    nDims = trainData.shape[1]
    qDims = model.X.shape[1]

    # get active dimensions
    if ard:
        scales = model.kern.input_sensitivity(summarize=False)
        scales = scales/scales.max()
        nActDim = max((scales >= wThresh).sum(),minDim)
        inactiveDims = np.argsort(scales)[:-nActDim]
    else:
        inactiveDims = []

    # create output variables
    testOut = np.zeros((nTest,nDims))
    trainOut = np.zeros((nTrain,nDims))

    testLatent = np.zeros((nTest,qDims))
    trainLatent = np.zeros((nTrain,qDims))

    testError = np.zeros((nTest,1))
    trainError = np.zeros((nTrain,1))

    # loop over training data
    for n in range(nTrain):
        # infer latent position
        xTrain, _ = model.infer_newX(np.atleast_2d(trainData[n,:]),optimize=False)

        # update parameter
        xTrain[0,inactiveDims] = 0.0
        trainLatent[n,:] = xTrain
        

        # infer high dimensional output
        yOut = model.predict(xTrain)
        trainOut[n,:] = yOut[0]
        sys.stdout.write('.')
    sys.stdout.write('\n')

    # loop over test data
    for n in range(nTest):
        # infer latent position
        xTest, _ = model.infer_newX(np.atleast_2d(testData[n,:]),optimize=True)

        # update parameter
        xTest[0,inactiveDims] = 0.0
        testLatent[n,:] = xTest

        # infer high dimensional output
        yOut = model.predict(xTest)
        testOut[n,:] = yOut[0]
        sys.stdout.write('.')
    sys.stdout.write('\n')

    # compute reconstruction error
    testError = np.sqrt(((testData - testOut)**2).sum(axis=1)/nDims)
    trainError = np.sqrt(((trainData - trainOut)**2).sum(axis=1)/nDims)

    # return the variables
    return (trainLatent, testLatent, trainOut, testOut, trainError, testError)

# function to compute test and training reconstruction error
def bgplvmError(model, trainData, testData, wThresh=0.05, minDim=3, ard=True):
    # get number of test and training samples
    nTest = testData.shape[0]
    nTrain = trainData.shape[0]

    nDims = trainData.shape[1]
    qDims = model.X.mean.shape[1]

    # get active dimensions
    if ard:
        scales = model.kern.input_sensitivity(summarize=False)
        scales = scales/scales.max()
        nActDim = max((scales >= wThresh).sum(),minDim)
        inactiveDims = np.argsort(scales)[:-nActDim]
    else:
        inactiveDims = []

    # create output variables
    testOut = np.zeros((nTest,nDims))
    trainOut = np.zeros((nTrain,nDims))

    testLatent = np.zeros((nTest,qDims))
    trainLatent = np.zeros((nTrain,qDims))

    testError = np.zeros((nTest,1))
    trainError = np.zeros((nTrain,1))

    # loop over training data
    for n in range(nTrain):
        # infer latent position
        xTrain, _ = model.infer_newX(np.atleast_2d(trainData[n,:]),optimize=False)

        # update parameter
        xTrain.mean[0,inactiveDims] = 0.0
        trainLatent[n,:] = xTrain.mean

        # infer high dimensional output
        yOut = model.predict(xTrain.mean)
        trainOut[n,:] = yOut[0]
        sys.stdout.write('.')
    sys.stdout.write('\n')

    # loop over test data
    for n in range(nTest):
        # infer latent position
        xTest, _ = model.infer_newX(np.atleast_2d(testData[n,:]),optimize=True)

        # update parameter
        xTest.mean[0,inactiveDims] = 0.0
        testLatent[n,:] = xTest.mean

        # infer high dimensional output
        yOut = model.predict(xTest.mean)
        testOut[n,:] = yOut[0]
        sys.stdout.write('.')
    sys.stdout.write('\n')

    # compute reconstruction error
    testError = np.sqrt(((testData - testOut)**2).sum(axis=1)/nDims)
    trainError = np.sqrt(((trainData - trainOut)**2).sum(axis=1)/nDims)

    # return the variables
    return (trainLatent, testLatent, trainOut, testOut, trainError, testError)
