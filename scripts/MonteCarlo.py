"""
This is for the Monte Carlo simulation of the configuration of many rafts
The maximum characters per line is set to be 120.

"""
# import glob
import os
import sys
import shelve
import platform
import datetime

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import progressbar
# from scipy.integrate import RK45
from scipy.integrate import solve_ivp
from scipy.spatial import Voronoi as scipyVoronoi
# import scipy.io
from scipy.spatial import distance as scipy_distance

parallel_mode = 0

if platform.node() == 'NOTESIT43' and platform.system() == 'Windows':
    projectDir = "D:\\simulationFolder\\spinning_rafts_sim2"
elif platform.node() == 'NOTESIT71' and platform.system() == 'Linux':
    projectDir = r'/media/wwang/shared/spinning_rafts_simulation/spinning_rafts_sim2'
else:
    projectDir = os.getcwd()

if projectDir != os.getcwd():
    os.chdir(projectDir)

if parallel_mode == 1:
    import functions_spinning_rafts as fsr
else:
    import scripts.functions_spinning_rafts as fsr

scriptDir = os.path.join(projectDir, "scripts")
# capSym6Dir = os.path.join(projectDir, '2019-05-13_capillaryForceCalculations-sym6')
# capSym4Dir = os.path.join(projectDir, '2019-03-29_capillaryForceCalculations')
dataDir = os.path.join(projectDir, 'data')
if not os.path.isdir(dataDir):
    os.mkdir('data')


# %% Monte Carlo simulation
# key parameters
os.chdir(dataDir)
if parallel_mode == 1:
    numOfRafts = int(sys.argv[1])
    spinSpeed = int(sys.argv[2])
else:
    numOfRafts = 218
    spinSpeed = 30
numOfTimeSteps = 50000  # 80000
arenaSize = 1.5e4  # unit: micron
centerOfArena = np.array([arenaSize / 2, arenaSize / 2])
R = raftRadius = 1.5e2  # unit: micron

batchSize = 2  # how many rafts are moved together
incrementSize = 50  # unit: radius, initial increment size
finalIncrementSize = 5  # unit: radius
incrementSwitchStep = 0  # step at which increment size is decreased
rejectionThreshold = 0.9  # threshold below which we want to keep the rejection rate
adaptiveIncrement = 0.7  # factor by which we decrease the increment step size each time
numOfTestSteps = 100  # number of steps to run before testing the rejection rate
incrementSizeArray = np.zeros(numOfTimeSteps)
countIncrement2 = 0

binSize_NDist = 0.5  # unit: radius
binStart_NDist = 2  # unit: radius
binEnd_NDist = 50  # unit: radius
binEdgesNeighborDistances = list(np.arange(binStart_NDist, binEnd_NDist, binSize_NDist)) + [100]
binSize_NAngles = 10  # unit: deg
binStart_NAngles = 0  # unit: deg
binEnd_NAngles = 360  # unit: deg
binEdgesNeighborAngles = list(np.arange(binStart_NAngles, binEnd_NAngles, binSize_NAngles)) + [360]
binSize_ODist = 0.5  # unit: radius
binStart_ODist = 2  # unit: radius
binEnd_ODist = 80  # unit: radius
binEdgesOrbitingDistances = list(np.arange(binStart_ODist, binEnd_ODist, binSize_ODist)) + [100]
binSize_XY = 2  # unit: radius
binEdgesX = list(np.arange(0, arenaSize/R + binSize_XY, binSize_XY))
binEdgesY = list(np.arange(0, arenaSize/R + binSize_XY, binSize_XY))

# load target distributions
expDuration = 20
tempShelf = shelve.open('target_{}s_{}Rafts_{}rps_reprocessed'.format(expDuration, numOfRafts, spinSpeed))
variableListOfTargetDistributions = list(tempShelf.keys())
target = {}
for key in tempShelf:
    try:
        target[key] = tempShelf[key]
    except TypeError:
        pass
tempShelf.close()
# readjust parameters according to the target distributions
binEdgesX = target['binEdgesX']
binEdgesY = target['binEdgesY']
arenaSize = target['arenaSizeInR'] * R
centerOfArena = np.array([arenaSize / 2, arenaSize / 2])

# make folder for the current dataset
now = datetime.datetime.now()
if parallel_mode == 1:
    outputFolderName = now.strftime("%Y-%m-%d") + '_{}Rafts_totalSteps{}_{}rps_incre{}R_batchSize{}'.format(
        numOfRafts, numOfTimeSteps, spinSpeed, finalIncrementSize, batchSize)
else:
    outputFolderName = now.strftime("%Y-%m-%d_%H-%M-%S") + \
                       '_{}Rafts_totalSteps{}_{}rps_incre{}R_batchSize{}'.format(
                           numOfRafts, numOfTimeSteps, spinSpeed, finalIncrementSize, batchSize)

if not os.path.isdir(outputFolderName):
    os.mkdir(outputFolderName)
os.chdir(outputFolderName)

# Drawing related parameters (in pixel unit)
canvasSizeInPixel = int(1000)  # unit: pixel
scaleBar = arenaSize / canvasSizeInPixel  # unit: micron/pixel
blankFrameBGR = np.ones((canvasSizeInPixel, canvasSizeInPixel, 3), dtype='int32') * 255

# key data variables
raftLocations = np.zeros((numOfRafts, numOfTimeSteps, 2))  # in microns
raftRadii = np.ones(numOfRafts) * raftRadius  # in micron
count_NDist = np.zeros((len(binEdgesNeighborDistances)-1, numOfTimeSteps))
count_NAngles = np.zeros((len(binEdgesNeighborAngles)-1, numOfTimeSteps))
count_ODist = np.zeros((len(binEdgesOrbitingDistances)-1, numOfTimeSteps))
count_X = np.zeros((len(binEdgesX)-1, numOfTimeSteps))
count_Y = np.zeros((len(binEdgesY)-1, numOfTimeSteps))
klDiv_NDist = np.ones(numOfTimeSteps)
klDiv_NAngles = np.ones(numOfTimeSteps)
klDiv_ODist = np.ones(numOfTimeSteps)
klDiv_X = np.ones(numOfTimeSteps)
klDiv_Y = np.ones(numOfTimeSteps)
entropy_NDist = np.zeros(numOfTimeSteps)
entropy_NAngles = np.zeros(numOfTimeSteps)
entropy_ODist = np.zeros(numOfTimeSteps)
entropy_X = np.zeros(numOfTimeSteps)
entropy_Y = np.zeros(numOfTimeSteps)
hexOrderParas = np.zeros((numOfRafts, numOfTimeSteps), dtype="complex")
rejectionRates = np.zeros(numOfTimeSteps)

# initialize rafts positions: 1 - random positions, 2 - fixed initial position,
# 3 - hexagonal fixed position
initialPositionMethod = 1
currStepNum = 0
paddingAroundArena = 5  # unit: R
if initialPositionMethod == 1:
    ccDistanceMin = 2.5  # unit: R
    raftLocations[:, currStepNum, :] = np.random.uniform(0 + raftRadius * paddingAroundArena,
                                                         arenaSize - raftRadius * paddingAroundArena,
                                                         (numOfRafts, 2))
    raftsToRelocate = np.arange(numOfRafts)
    while len(raftsToRelocate) > 0:
        raftLocations[raftsToRelocate, currStepNum, :] = np.random.uniform(
            0 + raftRadius * paddingAroundArena,
            arenaSize - raftRadius * paddingAroundArena, (len(raftsToRelocate), 2))
        pairwiseDistances = scipy_distance.cdist(raftLocations[:, currStepNum, :],
                                                 raftLocations[:, currStepNum, :], 'euclidean')
        np.fill_diagonal(pairwiseDistances, raftRadius * ccDistanceMin + 1)
        raftsToRelocate, _ = np.nonzero(pairwiseDistances < raftRadius * ccDistanceMin)
        raftsToRelocate = np.unique(raftsToRelocate)
elif initialPositionMethod == 2:
    raftLocations[:, currStepNum, :] = fsr.square_spiral(numOfRafts, raftRadius * 2 + 100, centerOfArena)
elif initialPositionMethod == 3:
    raftLocations[:, currStepNum, :] = fsr.hexagonal_spiral(numOfRafts, raftRadius * 2 + 200, centerOfArena)

# drawing rafts
currentFrameBGR = fsr.draw_rafts_rh_coord(blankFrameBGR.copy(),
                                          np.int32(raftLocations[:, currStepNum, :] / scaleBar),
                                          np.int64(raftRadii / scaleBar), numOfRafts)
currentFrameBGR = fsr.draw_raft_num_rh_coord(currentFrameBGR,
                                             np.int64(raftLocations[:, currStepNum, :] / scaleBar),
                                             numOfRafts)

outputFileName = 'MonteCarlo_' + str(numOfRafts) + 'Rafts_' + 'startPosMeth' + str(initialPositionMethod) \
                 + '_numOfSteps' + str(numOfTimeSteps) + '_currStep'

outputImageName = outputFileName + str(currStepNum).zfill(7) + '.jpg'
cv.imwrite(outputImageName, currentFrameBGR)

# try run optimization on x and y distribution first, once they are below a certain threshold, start optimizing NDist
masterSwitch = 1  # 1: switch runNDist on after 100 step, 2: switch runNDist_NAngles on after 100 step
runNDist = 0  # switch for running NDist or not
runNDist_NAngles = 0
beta = 1000  # inverse of effective temperature
switchThreshold = (1/numOfRafts) * np.log2(1e9/numOfRafts)  # penalty for rafts in the probability zero region
for currStepNum in progressbar.progressbar(np.arange(0, numOfTimeSteps - 1)):
    dict_X = fsr.count_kldiv_entropy_x(raftLocations[:, currStepNum, :], raftRadius, binEdgesX, target)
    dict_Y = fsr.count_kldiv_entropy_y(raftLocations[:, currStepNum, :], raftRadius, binEdgesY, target)
    # assignments
    count_X[:, currStepNum], klDiv_X[currStepNum] = dict_X['count_X'], dict_X['klDiv_X'],
    count_Y[:, currStepNum], klDiv_Y[currStepNum] = dict_Y['count_Y'], dict_Y['klDiv_Y']
    entropy_X[currStepNum], entropy_Y[currStepNum] = dict_X['entropy_X'], dict_Y['entropy_Y']

    # if runNDist == 1:
    #     dict_NDist = fsr.count_kldiv_entropy_ndist(raftLocations[:, currStepNum, :], raftRadius,
    #                                                binEdgesNeighborDistances, target)
    #     count_NDist[:, currStepNum], klDiv_NDist[currStepNum] = dict_NDist['count_NDist'], dict_NDist['klDiv_NDist']
    #     entropy_NDist[currStepNum] = dict_NDist['entropy_NDist']

    if runNDist == 1 or runNDist_NAngles == 1:
        dict_NDist_NAngles = fsr.count_kldiv_entropy_ndist_nangles(raftLocations[:, currStepNum, :], raftRadius,
                                                                   binEdgesNeighborDistances, binEdgesNeighborAngles,
                                                                   target)
        count_NDist[:, currStepNum], klDiv_NDist[currStepNum] = \
            dict_NDist_NAngles['count_NDist'], dict_NDist_NAngles['klDiv_NDist']
        count_NAngles[:, currStepNum], klDiv_NAngles[currStepNum] = \
            dict_NDist_NAngles['count_NAngles'], dict_NDist_NAngles['klDiv_NAngles']
        entropy_NDist[currStepNum], entropy_NAngles[currStepNum] = \
            dict_NDist_NAngles['entropy_NDist'], dict_NDist_NAngles['entropy_NAngles']
        hexOrderParas[:, currStepNum] = dict_NDist_NAngles['hexOrderParas']

    newLocations = raftLocations[:, currStepNum, :].copy()
    permuatedRaftIDs = list(np.random.permutation(numOfRafts))
    for batchNum in np.arange(int(numOfRafts/batchSize)):
        # raftID = 0
        # batchNum = 0
        firstRaftInBatch = batchSize * batchNum
        lastRaftInBatch = batchSize * (batchNum + 1) if (batchNum + 1) * batchSize <= numOfRafts else numOfRafts
        raftIDs = permuatedRaftIDs[firstRaftInBatch:lastRaftInBatch]
        restRaftIDS = list(set(permuatedRaftIDs) - set(raftIDs))

        incrementInXY = np.random.uniform(low=-1, high=1, size=(batchSize, 2)) * incrementSize * R
        # take care of the cases where moving the rafts outside the arena or overlapping with another raft.
        newXY = newLocations[raftIDs, :] + incrementInXY
        while newXY.max() > arenaSize - paddingAroundArena * R or newXY.min() < 0 + paddingAroundArena * R or \
              scipy_distance.cdist(newLocations[restRaftIDS, :], newXY).min() < 2 * R or \
                scipy_distance.cdist(newXY, newXY)[np.nonzero(scipy_distance.cdist(newXY, newXY))].min() < 2 * R:
            incrementInXY = np.random.uniform(low=-1, high=1, size=(batchSize, 2)) * incrementSize * R
            newXY = newLocations[raftIDs, :] + incrementInXY
        newLocations[raftIDs, :] = newXY

        dict_X = fsr.count_kldiv_entropy_x(newLocations, raftRadius, binEdgesX, target)
        dict_Y = fsr.count_kldiv_entropy_y(newLocations, raftRadius, binEdgesY, target)
        if runNDist == 1:
            dict_NDist = fsr.count_kldiv_entropy_ndist(newLocations, raftRadius, binEdgesNeighborDistances, target)
        if runNDist_NAngles == 1:
            dict_NDist_NAngles = fsr.count_kldiv_entropy_ndist_nangles(newLocations, raftRadius,
                                                                       binEdgesNeighborDistances,
                                                                       binEdgesNeighborAngles, target)

        # calculate the difference in divergences
        diff_klDiv_X = dict_X["klDiv_X"] - klDiv_X[currStepNum]
        diff_klDiv_Y = dict_Y["klDiv_Y"] - klDiv_Y[currStepNum]
        if runNDist == 1:
            diff_klDiv_NDist = dict_NDist["klDiv_NDist"] - klDiv_NDist[currStepNum]
        if runNDist_NAngles == 1:
            diff_klDiv_NDist = dict_NDist_NAngles["klDiv_NDist"] - klDiv_NDist[currStepNum]
            diff_klDiv_NAngles = dict_NDist_NAngles["klDiv_NAngles"] - klDiv_NDist[currStepNum]
        # accept the move if the divergences decrease, otherwise accept/reject according to probability
        if runNDist == 0 and runNDist_NAngles == 0:
            if (diff_klDiv_X <= 0) and (diff_klDiv_Y <= 0):
                continue
            else:
                newLocations[raftIDs, :] = newLocations[raftIDs, :] - incrementInXY
                rejectionRates[currStepNum] += batchSize
        elif runNDist == 1:
            if (diff_klDiv_X <= 0) and (diff_klDiv_Y <= 0) and (diff_klDiv_NDist <= 0):
                continue
            else:
                randomProb = np.random.uniform(low=0, high=1, size=1)
                diff_max = np.array((diff_klDiv_X, diff_klDiv_Y, diff_klDiv_NDist)).max()
                # higher diff or higher beta means less likely to jump
                jumpThresholdProb = np.exp(- diff_max * beta)
                if randomProb < np.exp(- diff_max * beta):
                    continue
                else:
                    newLocations[raftIDs, :] = newLocations[raftIDs, :] - incrementInXY
                    rejectionRates[currStepNum] += batchSize
        elif runNDist_NAngles == 1:
            if (diff_klDiv_X <= 0) and (diff_klDiv_Y <= 0) and (diff_klDiv_NDist <= 0) and (diff_klDiv_NAngles <= 0):
                continue
            else:
                randomProb = np.random.uniform(low=0, high=1, size=1)
                diff_max = np.array((diff_klDiv_X, diff_klDiv_Y, diff_klDiv_NDist, diff_klDiv_NAngles)).max()
                # higher diff or higher beta means less likely to jump
                jumpThresholdProb = np.exp(- diff_max * beta)
                if randomProb < np.exp(- diff_max * beta):
                    continue
                else:
                    newLocations[raftIDs, :] = newLocations[raftIDs, :] - incrementInXY
                    rejectionRates[currStepNum] += batchSize

    # if the KL divergences of the global distributions are good, then switch on runNDist
    if currStepNum > 100 and np.all(klDiv_X[currStepNum - 5: currStepNum] < switchThreshold) and \
            np.all(klDiv_Y[currStepNum - 5: currStepNum] < switchThreshold):
        if masterSwitch == 1:
            runNDist = 1
        elif masterSwitch == 2:
            runNDist_NAngles = 1
        #
        incrementSize = finalIncrementSize  # unit: radius
        incrementSwitchStep = currStepNum

    # After running for "NumOfTestSteps" if the minimum of rejection rate (in last 100 frames) is higher
    # than the "rejectionThreshold" we decrease the increment size
    # if currStepNum > incrementSwitchStep + numOfTestSteps and \
    #         rejectionRates[incrementSwitchStep: currStepNum].mean() > rejectionThreshold * numOfRafts:
    #     incrementSize = int(incrementSize * adaptiveIncrement)
    #     if incrementSize < 5:
    #         incrementSize = 5
    #         countIncrement2 += 1
    #     if countIncrement2 > 3:
    #         incrementSize = 30
    #         countIncrement2 = 0
    #     incrementSwitchStep = currStepNum

    # incrementSizeArray[currStepNum] = incrementSize

    raftLocations[:, currStepNum + 1, :] = newLocations

# %% plotting simulation results
# KL divergence of neighbor distances vs time steps
fig, ax = plt.subplots(ncols=1, nrows=1)
ax.plot(np.arange(numOfTimeSteps - 1), klDiv_NDist[:-1], label='kldiv_NDist vs steps')
# ax.plot(np.arange(900, numOfTimeSteps - 2), np.diff(klDiv_NDist[:-1])[900:], label='kldiv_NDist vs steps')
ax.set_xlabel('time steps', size=20)
ax.set_ylabel('KL divergence of NDist', size=20)
ax.set_title('KL divergence of NDist')
ax.set_yscale("log")
ax.legend()
plt.show()
figName = 'KL divergence of NDist'
fig.savefig(figName)

fig, ax = plt.subplots(ncols=1, nrows=1)
ax.plot(np.arange(numOfTimeSteps - 1), klDiv_NAngles[:-1], label='kldiv_NAngles vs steps')
ax.set_xlabel('time steps', size=20)
ax.set_ylabel('KL divergence of NAngles', size=20)
ax.set_title('KL divergence of NAngles')
ax.legend()
plt.show()
figName = 'KL divergence of NAngles'
fig.savefig(figName)

# KL divergence of x vs time steps
fig, ax = plt.subplots(ncols=1, nrows=1)
ax.plot(np.arange(numOfTimeSteps - 1), klDiv_X[:-1], label='kldiv_X vs steps')
ax.set_xlabel('time steps', size=20)
ax.set_ylabel('KL divergence of X', size=20)
ax.set_title('KL divergence of X')
ax.set_yscale("log")
ax.legend()
plt.show()
figName = 'KL divergence of X'
fig.savefig(figName)

# KL divergence of y vs time steps
fig, ax = plt.subplots(ncols=1, nrows=1)
ax.plot(np.arange(numOfTimeSteps - 1), klDiv_Y[:-1], label='kldiv_Y vs steps')
ax.set_xlabel('time steps', size=20)
ax.set_ylabel('KL divergence of Y', size=20)
ax.set_title('KL divergence of Y')
ax.set_yscale("log")
ax.legend()
plt.show()
figName = 'KL divergence of Y'
fig.savefig(figName)

# rejection rate vs time steps
fig, ax = plt.subplots(ncols=1, nrows=1)
ax.plot(np.arange(numOfTimeSteps - 1), rejectionRates[:-1], label='total number of rafts {}'.format(numOfRafts))
ax.set_xlabel('time steps', size=20)
ax.set_ylabel('rejection rate', size=20)
ax.set_title('rejection rate')
ax.legend()
plt.show()
figName = 'rejection rate'
fig.savefig(figName)

# Histogram of neighbor distances
fig, ax = plt.subplots(ncols=1, nrows=1)
ax.plot(binEdgesNeighborDistances[:-1],
        count_NDist[:, currStepNum] / count_NDist[:, currStepNum].sum(),
        label='NDist distribution of the last step')
ax.plot(binEdgesNeighborDistances[:-1], target['count_NDist']/target['count_NDist'].sum(),
        label='target NDist distribution')
ax.set_xlabel('edge-edge distance', size=20)
ax.set_ylabel('probability', size=20)
ax.set_title('histogram of neighbor distances')
ax.legend()
ax.set_yscale("log")
plt.show()
figName = 'Histogram of neighbor distances'
fig.savefig(figName)

# Histogram of neighbor angles
if runNDist_NAngles == 0:
    dict_NDist_NAngles = fsr.count_kldiv_entropy_ndist_nangles(raftLocations[:, -2, :], raftRadius,
                                                               binEdgesNeighborDistances,
                                                               binEdgesNeighborAngles, target)
    count_NAngles[:, currStepNum] = dict_NDist_NAngles['count_NAngles']
    hexOrderParas[:, -2] = dict_NDist_NAngles['hexOrderParas']
fig, ax = plt.subplots(ncols=1, nrows=1)
ax.plot(binEdgesNeighborAngles[:-1],
        count_NAngles[:, currStepNum] / count_NAngles[:, currStepNum].sum(),
        label='NAngles distribution')
ax.set_xlabel('edge-edge distance', size=20)
ax.set_ylabel('probability', size=20)
ax.set_title('histogram of neighbor angles')
ax.legend()
plt.show()
figName = 'Histogram of neighbor angles'
fig.savefig(figName)

# Histogram of x
fig, ax = plt.subplots(ncols=1, nrows=1)
ax.plot(binEdgesX[:-1], count_X[:, currStepNum] / count_X[:, currStepNum].sum(), label='marginal distribution of x')
ax.plot(binEdgesX[:-1], target['count_X'] / target['count_X'].sum(), label='target marginal distribution of x')
ax.set_xlabel('x', size=20)
ax.set_ylabel('probability', size=20)
ax.set_title('histogram of marginal distribution of X')
ax.legend()
ax.set_yscale('log')
plt.show()
figName = 'Histogram of marginal distribution of X'
fig.savefig(figName)

# Histogram of y
fig, ax = plt.subplots(ncols=1, nrows=1)
ax.plot(binEdgesY[:-1], count_Y[:, currStepNum] / count_Y[:, currStepNum].sum(), label='marginal distribution of y')
ax.plot(binEdgesX[:-1], target['count_Y'] / target['count_Y'].sum(), label='target marginal distribution of y')
ax.set_xlabel('y', size=20)
ax.set_ylabel('probability', size=20)
ax.set_title('histogram of marginal distribution of y')
ax.legend()
ax.set_yscale('log')
plt.show()
figName = 'Histogram of marginal distribution of y'
fig.savefig(figName)

# hexatic order parameters plots
# some data treatment
hexaticOrderParameterAvgs = hexOrderParas.mean(axis=0)
hexaticOrderParameterAvgNorms = np.sqrt(hexaticOrderParameterAvgs.real ** 2 + hexaticOrderParameterAvgs.imag ** 2)
hexaticOrderParameterModulii = np.absolute(hexOrderParas)
hexaticOrderParameterModuliiAvgs = hexaticOrderParameterModulii.mean(axis=0)
hexaticOrderParameterModuliiStds = hexaticOrderParameterModulii.std(axis=0)
# plot
fig, ax = plt.subplots(ncols=1, nrows=1)
ax.plot(np.arange(numOfTimeSteps-1), hexaticOrderParameterAvgNorms[:-1], label='psi6 averages and norms')
ax.errorbar(np.arange(numOfTimeSteps-1), hexaticOrderParameterModuliiAvgs[:-1],
            yerr=hexaticOrderParameterModuliiStds[:-1], errorevery=int(numOfTimeSteps/200),
            label='psi6 norms and averages')
ax.set_xlabel('y', size=20)
ax.set_ylabel('order parameter', size=20)
ax.set_title('hexatic order parameters over time')
ax.legend()
plt.show()
figName = 'hexatic order parameters over time'
fig.savefig(figName)

# save last frame
currentFrameBGR = fsr.draw_rafts_rh_coord(blankFrameBGR.copy(),
                                          np.int32(raftLocations[:, currStepNum, :] / scaleBar),
                                          np.int64(raftRadii / scaleBar), numOfRafts)
currentFrameBGR = fsr.draw_raft_num_rh_coord(currentFrameBGR,
                                             np.int64(raftLocations[:, currStepNum, :] / scaleBar),
                                             numOfRafts)

outputFileName = 'MonteCarlo_' + str(numOfRafts) + 'Rafts_' + 'startPosMeth' + str(initialPositionMethod) \
                 + '_numOfSteps' + str(numOfTimeSteps) + '_currStep'

outputImageName = outputFileName + str(currStepNum).zfill(7) + '.jpg'
cv.imwrite(outputImageName, currentFrameBGR)

# save the frame with maximum hex order parameter:
stepNumOfMaxHex = hexaticOrderParameterModuliiAvgs.argmax()
currentFrameBGR = fsr.draw_rafts_rh_coord(blankFrameBGR.copy(),
                                          np.int32(raftLocations[:, stepNumOfMaxHex, :] / scaleBar),
                                          np.int64(raftRadii / scaleBar), numOfRafts)
currentFrameBGR = fsr.draw_raft_num_rh_coord(currentFrameBGR,
                                             np.int64(raftLocations[:, stepNumOfMaxHex, :] / scaleBar),
                                             numOfRafts)

outputFileName = 'MonteCarlo_' + str(numOfRafts) + 'Rafts_' + 'startPosMeth' + str(initialPositionMethod) \
                 + '_numOfSteps' + str(numOfTimeSteps) + '_currStep'

outputImageName = outputFileName + str(stepNumOfMaxHex).zfill(7) + '.jpg'
cv.imwrite(outputImageName, currentFrameBGR)

# save files
listOfVariablesToSave = ['numOfRafts', 'arenaSize', 'spinSpeed',
                         'raftLocations',
                         'binEdgesNeighborDistances', 'binEdgesOrbitingDistances',
                         'binEdgesX', 'binEdgesX',
                         'entropy_NDist', 'count_NDist',
                         'entropy_ODist', 'count_ODist',
                         'entropy_X', 'count_X',
                         'entropy_Y', 'count_Y']
tempShelf = shelve.open('target_' + str(numOfRafts) + "Rafts_" + str(spinSpeed) + 'rps')
for key in listOfVariablesToSave:
    try:
        tempShelf[key] = globals()[key]
    except TypeError:
        #
        # __builtins__, tempShelf, and imported modules can not be shelved.
        #
        # print('ERROR shelving: {0}'.format(key))
        pass
tempShelf.close()


