#!/usr/bin/env python

# plotFuncs.py: plot functions for data inspection
# Author: Nishanth Koganti
# Date: 2018/06/04

import sys
import GPy
import matplotlib
import numpy as np
from matplotlib import cm
from matplotlib import pyplot as plt

################################################################################
# Plot properties fixed through the functions
################################################################################
fontSize = 15

################################################################################
# Functions to visualize trajectory data
################################################################################

def plotTraj(Dataset, plotType = 0, jointIndex = np.arange(6),
             labels = ['train','test'], colors=['b','r']):
    """function to plot multiple joint tracks."""
    timeData = []
    leftData = []
    rightData = []

    LEFT_ANGLE_OFFSET = 1
    RIGHT_ANGLE_OFFSET = 7

    # loop over first plotNum files
    for data in Dataset.values():
        timeData.append(data[:, 0])
        leftData.append(data[:, LEFT_ANGLE_OFFSET+jointIndex])
        rightData.append(data[:, RIGHT_ANGLE_OFFSET+jointIndex])

    jointData = [leftData, rightData]

    # number of joints to plot
    xlabel = 'Time(sec)'
    arms = ['Left', 'Right']
    nJoints = jointIndex.size
    if plotType == 0:
        ylabels = 7*['Joint Angle (rad)']
    else:
        ylabels = 3*['Position (m)']+4*['Angle (rad)']

    # plot all the joint data
    for ind in range(2):
        fig = plt.figure(figsize=(8, 2*nJoints))
        for i, jI in enumerate(jointIndex):
            plt.subplot(nJoints, 1, i+1)

            # plot all the tracks
            for n in range(len(Dataset.values())):
                timeDat = timeData[n]
                plt.plot(timeDat, jointData[ind][n][:, i], label=labels[n],
                         color=colors[n], linewidth=2)

            plt.xlabel(xlabel, fontsize=10, fontweight='bold')
            plt.ylabel(ylabels[i], fontsize=10, fontweight='bold')

            if plotType == 0:
                plt.title('%s Joint %d' % (arms[ind], jI+1), fontsize=fontSize,
                          fontweight='bold')
            else:
                plt.title('%s Pose %d' % (arms[ind], jI+1), fontsize=fontSize,
                          fontweight='bold')

            # plot legend only for 1st sub plot
            if i == 0:
                plt.legend(frameon=True)

        # adjust subplots for legend
        fig.subplots_adjust(top=0.96, right=0.8)
        plt.tight_layout()

    # show all the plots
    plt.show()

def plotLatentTraj(Dataset, nDim, points=None, colors={'train':'b','test':'r'}, 
                   ylabel='Latent Pos.', title='Dim'):
    """function to plot multiple joint tracks."""
    timeData = {}
    latentData = {}

    # loop over first plotNum files
    for key,data in Dataset.items():
        timeData[key] = data[:, 0]
        latentData[key] = data[:, 1:]

    # number of latent dims to plot
    if nDim == None:
        nDim = latentData[key].shape[1]

    xlabel = 'Time(sec)'
    ylabels = nDim*[ylabel]

    # plot all the latent data
    _ = plt.figure(figsize=(8, 4*nDim))
    for i in range(nDim):
        plt.subplot(nDim, 1, i+1)

        # plot all the tracks
        for _,key in enumerate(Dataset.keys()):
            timeDat = timeData[key]
            plt.plot(timeDat, latentData[key][:, i], label=key,
                     color=colors[key], linewidth=2)

        if points:
            plt.plot(points[i][:, 0], points[i][:, 1], 'ob', markersize=15,
                     label='viapoints')
        
        plt.xlabel(xlabel, fontsize=fontSize+5)
        plt.ylabel(ylabels[i], fontsize=fontSize+5)
        plt.title('%s %d' % (title, i+1), fontsize=fontSize+5)

    # adjust subplots for legend
    plt.tight_layout()

    # show all the plots
    plt.show()

################################################################################
# Functions to visualize model parameters and latent spaces
################################################################################

def plotScales(model, yThresh=0.05, lvm='bgplvm', ax=None):
    # get ARD weight parameters
    if lvm == 'pca':
        ylabel = 'Eigen Values'
        scales = model.explained_variance_ratio_
    else:
        ylabel = 'ARD Weight'
        scales = model.kern.input_sensitivity(summarize=False)
    scales =  scales/scales.max()

    if ax == None:
        fig = plt.figure(figsize=(10,8))
        ax = fig.add_subplot(111)

    x = np.arange(1,scales.shape[0]+1)
    ax.bar(x, height=scales, width=0.8, align='center',
           color='b', edgecolor='k', linewidth=1.3)
    ax.plot([0.4, scales.shape[0]+0.6], [yThresh, yThresh],
            '--', linewidth=3, color='r')

    # setting the bar plot parameters
    ax.set_xlim(.4, scales.shape[0]+.6)
    ax.set_ylabel(ylabel, fontsize=fontSize, fontweight='bold')
    ax.tick_params(axis='both', labelsize=fontSize)
    ax.set_xticks(np.arange(1,scales.shape[0]+1))
    ax.set_xlabel('Latent Dimensions', fontsize=fontSize, fontweight='bold')

    plt.show()
    return ax

def plotLatent(model, trainData, testData, lvm='bgplvm', plotIndices=None, ax=None):
    sTest = 200
    sTrain = 150
    resolution = 50

    testMarker = 's'
    trainMarker = 'o'

    qDim = testData.shape[1]
    nTest = testData.shape[0]
    nTrain = trainData.shape[0]
    testLabels = [(1,0,0)]*nTest
    trainLabels = [(0,0,1)]*nTrain

    # variables for plotting
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111)

    if plotIndices == None:
        if lvm != 'pca':
            scales = model.kern.input_sensitivity(summarize=False)
        else:
            scales = model.explained_variance_ratio_
        plotIndices = np.argsort(scales)[-2:]
    
    input1, input2 = plotIndices
    print(input1, input2)

    # compute plot limits
    xmin = min(testData[:, input1].min(0), trainData[:, input1].min(0)) 
    ymin = min(testData[:, input2].min(0), trainData[:, input2].min(0)) 
    xmax = max(testData[:, input1].max(0), trainData[:, input1].max(0)) 
    ymax = max(testData[:, input2].max(0), trainData[:, input2].max(0)) 
    x_r, y_r = xmax-xmin, ymax-ymin
    xmin -= .1*x_r
    xmax += .1*x_r
    ymin -= .1*y_r
    ymax += .1*y_r

    if lvm != 'pca':
        # plot the variance for the model
        def plotFunction(x):
            Xtest_full = np.zeros((x.shape[0], qDim))
            Xtest_full[:, [input1, input2]] = x
            _, var = model.predict(np.atleast_2d(Xtest_full))
            var = var[:, :1]
            return -np.log(var)

        x, y = np.mgrid[xmin:xmax:1j*resolution, ymin:ymax:1j*resolution]
        gridData = np.hstack((x.flatten()[:, None], y.flatten()[:, None]))
        gridVariance = (plotFunction(gridData)).reshape((resolution, resolution))

        _ = plt.imshow(gridVariance.T, interpolation='bilinear',
                                    origin='lower', cmap=cm.gray,
                                    extent=(xmin, xmax, ymin, ymax))

    testHandle = ax.scatter(testData[:, input1], testData[:, input2],
                            marker=testMarker, s=sTest, c=testLabels,
                            linewidth=.2, edgecolor='k', alpha=1.)
    trainHandle = ax.scatter(trainData[:, input1], trainData[:, input2],
                             marker=trainMarker, s=sTrain, c=trainLabels,
                             linewidth=.2, edgecolor='k', alpha=1.)

    ax.grid(b=False)
    ax.set_aspect('auto')
    ax.tick_params(axis='both', labelsize=fontSize)
    ax.set_xlabel('Latent Dimension %i' % (input1+1), fontsize=fontSize, fontweight='bold')
    ax.set_ylabel('Latent Dimension %i' % (input2+1), fontsize=fontSize, fontweight='bold')

    ax.set_xlim((xmin, xmax))
    ax.set_ylim((ymin, ymax))

    properties = {'size':fontSize, 'weight':'bold'}
    plt.legend([trainHandle, testHandle], ['Train', 'Test'], prop=properties, 
               loc=2, frameon=True)

    fig.canvas.draw()
    fig.tight_layout()
    fig.canvas.draw()
    plt.show()

    return ax

################################################################################
# Functions to visualize results
################################################################################

# function to plot error bars
def plotErrorBars(mE, vE, xLabels, legend, colors, ylabel='NRMSE',
                  legendLoc=3, title='Comparison', ylimit=[0.,1.],
                  xlimit=[-0.1,2.1]):

    N = mE.shape[1]

    widthFull = 0.8
    width = widthFull/N
    buffer = (1.0 - widthFull)/2.0

    ind = np.arange(mE.shape[0])
    _, ax = plt.subplots(figsize=(10,8))

    for i in range(mE.shape[1]):
        _ = ax.bar(buffer+ind+i*width, mE[:,i], yerr=vE[:,i], width=width,
                     color=colors[i], ecolor='k')

    ax.set_ylim(ylimit)
    ax.set_xlim(xlimit)
    ax.set_xticks(ind + 0.5)
    ax.set_ylabel(ylabel, fontsize=fontSize, fontweight='bold')
    ax.set_xticklabels(xLabels, fontsize=fontSize, fontweight='bold')
    ax.legend(legend, loc=legendLoc, fontsize=fontSize,
              prop = {'size':fontSize, 'weight':'bold'})

    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(fontSize)

    plt.tight_layout()
    plt.show()
    return ax
