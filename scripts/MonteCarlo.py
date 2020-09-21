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
# import pandas as pd
import progressbar
# from scipy.integrate import RK45
# from scipy.integrate import solve_ivp
# from scipy.spatial import Voronoi as scipyVoronoi
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
if parallel_mode == 1:
    numOfRafts = int(sys.argv[1])
    spinSpeed = int(sys.argv[2])
else:
    numOfRafts = 218
    spinSpeed = 30
numOfTimeSteps = 25000  # 80000
arenaSize = 1.5e4  # unit: micron
centerOfArena = np.array([arenaSize / 2, arenaSize / 2])
R = raftRadius = 1.5e2  # unit: micron

masterSwitch = 1  # 1: switch runNDist on after 100 step, 2: switch runNDist_NAngles on after 100 step
XY_or_ODist = 0   # 0 - XY, 1 - ODist
ifLastFrameCount = 1  # 0 - using counts from all frames, 1- using counts from the last frame only
initial_klDiv_avg = 0.01  # for both global and local klDiv
initial_klDiv_std = initial_klDiv_avg/2  # for both global and local klDiv
beta = 5 / initial_klDiv_std  # inverse of effective temperature ~ (5 / target_klDiv_NDist_std)
batchSize = 1  # how many rafts are moved together
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
experimental_generated = 0  # 0: experimental 20s data, 1: generated hexagonal patterns
if experimental_generated == 0:
    expDuration = 20
    expDataDir = os.path.join(projectDir, '2020-09-14_exp patterns', '{}s'.format(expDuration))
    os.chdir(expDataDir)
    shelfOfTarget = shelve.open('target_{}s_{}Rafts_{}rps_reprocessed'.format(expDuration, numOfRafts, spinSpeed))
elif experimental_generated == 1:
    genPatternDir = os.path.join(projectDir, '2020-09-17_generated patterns')
    os.chdir(genPatternDir)
    shelfOfTarget = shelve.open('generatedHexPattern_{}Rafts_processed'.format(numOfRafts))
variablesInTarget = list(shelfOfTarget.keys())
target = {}
for key in shelfOfTarget:
    try:
        target[key] = shelfOfTarget[key]
    except TypeError:
        pass
shelfOfTarget.close()
# readjust parameters according to the target distributions
binEdgesX = target['binEdgesX']
binEdgesY = target['binEdgesY']
arenaSizeInR = target['arenaSizeInR']
if ifLastFrameCount == 1:
    target['count_NDist'] = target['count_NDist_lastFrame']
    target['count_NAngles'] = target['count_NAngles_lastFrame']
    target['count_ODist'] = target['count_ODist_lastFrame']
    target['count_X'] = target['count_X_lastFrame']
    target['count_Y'] = target['count_Y_lastFrame']
arenaSize = arenaSizeInR * R
centerOfArena = np.array([arenaSize / 2, arenaSize / 2])

# make folder for the current dataset
os.chdir(dataDir)
now = datetime.datetime.now()
outputFolderName = now.strftime("%Y-%m-%d_%H-%M-%S") + \
                   '_{}Rafts_totalSteps{}_{}rps_incre{}R_batchSize{}_lastFrame{}_XYorODist{}_initialBeta{}'.format(
                       numOfRafts, numOfTimeSteps, spinSpeed, finalIncrementSize, batchSize, ifLastFrameCount,
                       XY_or_ODist, beta)

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
# 3 - hexagonal fixed position, 4 - load a particular frame from a previous run
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
elif initialPositionMethod == 4:
    os.chdir(dataDir)
    resultFolders = next(os.walk(dataDir))[1]
    resultFolders.sort()
    resultFolderID = 43  # the folder that contains the desired result
    os.chdir(resultFolders[resultFolderID])
    shelfToLoadFrame = shelve.open('simulation_{}Rafts_{}rps'.format(numOfRafts, spinSpeed), flag='r')
    frameNumToLoad = 652  # the frame number to load
    raftLocations[:, currStepNum, :] = shelfToLoadFrame['raftLocations'][:, frameNumToLoad, :]
    shelfToLoadFrame.close()
    os.chdir(dataDir)
    os.chdir(outputFolderName)

# drawing rafts
currentFrameBGR = fsr.draw_rafts_rh_coord(blankFrameBGR.copy(),
                                          np.int32(raftLocations[:, currStepNum, :] / scaleBar),
                                          np.int64(raftRadii / scaleBar), numOfRafts)
currentFrameBGR = fsr.draw_raft_num_rh_coord(currentFrameBGR,
                                             np.int64(raftLocations[:, currStepNum, :] / scaleBar),
                                             numOfRafts)

outputImageName = 'MonteCarlo_{}Rafts_startPosMeth{}_numOfSteps{}_currStep{}.jpg'.format(
    numOfRafts, initialPositionMethod, numOfTimeSteps, currStepNum)

cv.imwrite(outputImageName, currentFrameBGR)

# try run optimization on x and y distribution first, once they are below a certain threshold, start optimizing NDist
runNDist = 0  # switch for running NDist or not
runNDist_NAngles = 0
annealingSwitch = 0
if initialPositionMethod == 4:
    if masterSwitch == 1:
        runNDist = 1
    elif masterSwitch == 2:
        runNDist_NAngles = 1
    incrementSize = finalIncrementSize

target_klDiv_NDist_avg = initial_klDiv_avg  # 0.036
target_klDiv_NDist_std = initial_klDiv_std  # 0.007
target_klDiv_NAngles_avg = initial_klDiv_avg
target_klDiv_NAngles_std = initial_klDiv_std
target_klDiv_global_avg = initial_klDiv_avg  # 0.072  # for X, Y, ODist
target_klDiv_global_std = initial_klDiv_std  # 0.026

acceptedBasedOnProbs = []

switchThreshold = (1/numOfRafts) * np.log2(1e9/numOfRafts)  # penalty for rafts in the probability zero region
for currStepNum in progressbar.progressbar(np.arange(0, numOfTimeSteps - 1)):
    dict_X = fsr.count_kldiv_entropy_x(raftLocations[:, currStepNum, :], raftRadius, binEdgesX, target)
    dict_Y = fsr.count_kldiv_entropy_y(raftLocations[:, currStepNum, :], raftRadius, binEdgesY, target)
    dict_ODist = fsr.count_kldiv_entropy_odist(raftLocations[:, currStepNum, :], raftRadius, binEdgesOrbitingDistances,
                                               target, centerOfArena)
    # assignments
    count_X[:, currStepNum], klDiv_X[currStepNum] = dict_X['count_X'], dict_X['klDiv_X'],
    count_Y[:, currStepNum], klDiv_Y[currStepNum] = dict_Y['count_Y'], dict_Y['klDiv_Y']
    count_ODist[:, currStepNum], klDiv_ODist[currStepNum] = dict_ODist['count_ODist'], dict_ODist['klDiv_ODist']
    entropy_X[currStepNum], entropy_Y[currStepNum] = dict_X['entropy_X'], dict_Y['entropy_Y']
    entropy_ODist[currStepNum] = dict_ODist['entropy_ODist']

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
    permutatedRaftIDs = list(np.random.permutation(numOfRafts))
    numOfBatches = int((numOfRafts + batchSize - 1)/batchSize)
    for batchNum in np.arange(numOfBatches):
        # raftID = 0
        # batchNum = 0
        firstRaftInBatch = batchSize * batchNum
        lastRaftInBatch = batchSize * (batchNum + 1) if (batchNum + 1) * batchSize <= numOfRafts else numOfRafts
        raftIDs = permutatedRaftIDs[firstRaftInBatch:lastRaftInBatch]
        restRaftIDS = list(set(permutatedRaftIDs) - set(raftIDs))

        incrementInXY = np.random.uniform(low=-1, high=1, size=(len(raftIDs), 2)) * incrementSize * R
        # take care of the cases where moving the rafts outside the arena or overlapping with another raft.
        newXY = newLocations[raftIDs, :] + incrementInXY
        while newXY.max() > arenaSize - paddingAroundArena * R or newXY.min() < 0 + paddingAroundArena * R or \
              scipy_distance.cdist(newLocations[restRaftIDS, :], newXY).min() < 2 * R or \
                (batchSize > 1 and
                 scipy_distance.cdist(newXY, newXY)[np.nonzero(scipy_distance.cdist(newXY, newXY))].min() < 2 * R):
            incrementInXY = np.random.uniform(low=-1, high=1, size=(len(raftIDs), 2)) * incrementSize * R
            newXY = newLocations[raftIDs, :] + incrementInXY
        newLocations[raftIDs, :] = newXY

        if XY_or_ODist == 0:
            dict_X = fsr.count_kldiv_entropy_x(newLocations, raftRadius, binEdgesX, target)
            dict_Y = fsr.count_kldiv_entropy_y(newLocations, raftRadius, binEdgesY, target)
            # assign klDiv_global
            diff_klDiv_global = max(dict_X["klDiv_X"] - klDiv_X[currStepNum], dict_Y["klDiv_Y"] - klDiv_Y[currStepNum])
            diffToTarget_klDiv_global = max(dict_X["klDiv_X"], dict_Y["klDiv_Y"]) - target_klDiv_global_avg
        elif XY_or_ODist == 1:
            dict_ODist = fsr.count_kldiv_entropy_odist(newLocations, raftRadius, binEdgesOrbitingDistances, target,
                                                       centerOfArena)
            diff_klDiv_global = dict_ODist['klDiv_ODist'] - klDiv_ODist[currStepNum]
            diffToTarget_klDiv_global = dict_ODist['klDiv_ODist'] - target_klDiv_global_avg

        if runNDist == 1:
            dict_NDist = fsr.count_kldiv_entropy_ndist(newLocations, raftRadius, binEdgesNeighborDistances, target)
            diff_klDiv_NDist = dict_NDist["klDiv_NDist"] - klDiv_NDist[currStepNum]
            diffToTarget_klDiv_NDist = dict_NDist["klDiv_NDist"] - target_klDiv_NDist_avg

        if runNDist_NAngles == 1:
            dict_NDist_NAngles = fsr.count_kldiv_entropy_ndist_nangles(newLocations, raftRadius,
                                                                       binEdgesNeighborDistances,
                                                                       binEdgesNeighborAngles, target)
            diff_klDiv_NDist = dict_NDist_NAngles["klDiv_NDist"] - klDiv_NDist[currStepNum]
            diffToTarget_klDiv_NDist = dict_NDist_NAngles["klDiv_NDist"] - target_klDiv_NDist_avg
            diff_klDiv_NAngles = dict_NDist_NAngles["klDiv_NAngles"] - klDiv_NDist[currStepNum]
            diffToTarget_klDiv_NAngles = dict_NDist_NAngles["klDiv_NAngles"] - target_klDiv_NAngles_avg

        # accept the move if the divergences decrease, otherwise accept/reject according to probability
        if runNDist == 0 and runNDist_NAngles == 0:
            if diff_klDiv_global <= 0:
                continue
            else:
                newLocations[raftIDs, :] = newLocations[raftIDs, :] - incrementInXY
                rejectionRates[currStepNum] += batchSize
        elif runNDist == 1:
            if diff_klDiv_global <= 0 and diff_klDiv_NDist <= 0:
                continue
            elif (diffToTarget_klDiv_global <= target_klDiv_global_std) and (diff_klDiv_NDist <= 0):
                continue
            elif (diffToTarget_klDiv_global <= target_klDiv_global_std) and \
                    (diffToTarget_klDiv_NDist <= target_klDiv_NDist_std):
                continue
            else:
                randomProb = np.random.uniform(low=0, high=1, size=1)
                diff_max = max(diffToTarget_klDiv_global, diffToTarget_klDiv_NDist)
                # higher diff or higher beta means less likely to jump
                if randomProb < np.exp(- diff_max * beta):
                    acceptedBasedOnProbs.append(currStepNum)
                    continue
                else:
                    newLocations[raftIDs, :] = newLocations[raftIDs, :] - incrementInXY
                    rejectionRates[currStepNum] += batchSize

        elif runNDist_NAngles == 1:
            if (diff_klDiv_global <= 0) and (diff_klDiv_NDist <= 0) and (diff_klDiv_NAngles <= 0):
                continue
            elif (diffToTarget_klDiv_global <= target_klDiv_global_std) and (diff_klDiv_NDist <= 0) and \
                    (diff_klDiv_NAngles <= 0):
                continue
            elif (diffToTarget_klDiv_global <= target_klDiv_global_std) and \
                    (diffToTarget_klDiv_NDist <= target_klDiv_NDist_std) and \
                    (diffToTarget_klDiv_NAngles <= target_klDiv_NAngles_std):
                continue
            else:
                randomProb = np.random.uniform(low=0, high=1, size=1)
                diff_max = max(diffToTarget_klDiv_global, diffToTarget_klDiv_NDist, diffToTarget_klDiv_NAngles)
                # higher diff or higher beta means less likely to jump
                jumpThresholdProb = np.exp(- diff_max * beta)
                if randomProb < np.exp(- diff_max * beta):
                    continue
                else:
                    newLocations[raftIDs, :] = newLocations[raftIDs, :] - incrementInXY
                    rejectionRates[currStepNum] += batchSize

    # if the KL divergences of the global distributions are good, then switch on runNDist or runNDist_NAngles
    if runNDist == 0 and runNDist_NAngles == 0 and currStepNum > 100:
        if XY_or_ODist == 0:
            global_klDiv_BelowThreshold = np.all(klDiv_X[currStepNum - 5: currStepNum] < switchThreshold) and \
                np.all(klDiv_Y[currStepNum - 5: currStepNum] < switchThreshold)
        elif XY_or_ODist == 1:
            global_klDiv_BelowThreshold = np.all(klDiv_ODist[currStepNum - 5: currStepNum] < switchThreshold)

        if global_klDiv_BelowThreshold:
            if masterSwitch == 1:
                runNDist = 1
            elif masterSwitch == 2:
                runNDist_NAngles = 1
            #
            incrementSize = finalIncrementSize  # unit: radius
            incrementSwitchStep = currStepNum

    if (initialPositionMethod == 4 and currStepNum > 5) and (runNDist == 1 or runNDist_NAngles == 1):
        global_klDiv_target_fulfilled = 0
        local_klDiv_target_fulfilled = 0
        if XY_or_ODist == 0:
            global_klDiv_target_fulfilled = np.all(klDiv_X[currStepNum - 5: currStepNum] <
                                                   target_klDiv_global_avg + target_klDiv_global_std) and \
                                            np.all(klDiv_Y[currStepNum - 5: currStepNum] <
                                                   target_klDiv_global_avg + target_klDiv_global_std)
        elif XY_or_ODist == 1:
            global_klDiv_target_fulfilled = np.all(klDiv_ODist[currStepNum - 5: currStepNum] <
                                                   target_klDiv_global_avg + target_klDiv_global_std)

        if runNDist == 1:
            local_klDiv_target_fulfilled = np.all(klDiv_NDist[currStepNum - 5: currStepNum] <
                                                  target_klDiv_NDist_avg + target_klDiv_NDist_std)
        elif runNDist_NAngles == 1:
            local_klDiv_target_fulfilled = np.all(klDiv_NDist[currStepNum - 5: currStepNum] <
                                                  target_klDiv_NDist_avg + target_klDiv_NDist_std) and \
                                           np.all(klDiv_NAngles[currStepNum - 5: currStepNum] <
                                                  target_klDiv_NAngles_avg + target_klDiv_NAngles_std)

        if global_klDiv_target_fulfilled and local_klDiv_target_fulfilled:
            target_klDiv_NDist_avg = target_klDiv_NDist_avg/2
            target_klDiv_NDist_std = target_klDiv_NDist_std/2
            target_klDiv_NAngles_avg = target_klDiv_NAngles_avg/2
            target_klDiv_NAngles_std = target_klDiv_NAngles_std/2
            target_klDiv_global_avg = target_klDiv_global_avg/2
            target_klDiv_global_std = target_klDiv_global_std/2
            beta = 5 / target_klDiv_NDist_std

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

    # annealing like protocol
    if annealingSwitch == 1:
        if currStepNum % numOfTimeSteps in [5000]:
            beta = 0.01
            incrementSize = 0.1
        else:
            beta = 5 / target_klDiv_NDist_std
            incrementSize = finalIncrementSize


    raftLocations[:, currStepNum + 1, :] = newLocations

# %% plotting simulation results
# KL divergence of neighbor distances vs time steps
fig, ax = plt.subplots(ncols=1, nrows=1)
ax.plot(np.arange(numOfTimeSteps - 1), klDiv_NDist[:-1], label='kldiv_NDist vs steps')
# ax.plot(np.arange(900, numOfTimeSteps - 2), np.diff(klDiv_NDist[:-1])[900:], label='kldiv_NDist vs steps')
ax.set_xlabel('time steps', size=20)
ax.set_ylabel('KL divergence of NDist', size=20)
ax.set_title('KL divergence of NDist min = {0:3.3f}'.format(klDiv_NDist.min()))
ax.set_yscale("log")
ax.legend()
plt.show()
figName = 'KL divergence of NDist'
fig.savefig(figName)

# recount NDist based on 16 bins instead of 96 bins
count_NDist_16bins = np.ones((16, numOfTimeSteps))
count_NDist_16bins[0:15, :] = count_NDist[0:15, :]
count_NDist_16bins[15, :] = count_NDist[15:, :].sum(axis=0, keepdims=1)
entropy_NDist_16bins = np.zeros(numOfTimeSteps)
for ii in np.arange(numOfTimeSteps-1):
    if count_NDist_16bins[:, ii].sum() > 0:
        entropy_NDist_16bins[ii] = fsr.shannon_entropy(count_NDist_16bins[:, ii])
fig, ax = plt.subplots(ncols=1, nrows=1)
ax.plot(np.arange(numOfTimeSteps - 1), entropy_NDist[:-1], label='entropy_NDist min = {0:3.3f}'.format(
    entropy_NDist[np.nonzero(entropy_NDist)].min()))
ax.plot(np.arange(numOfTimeSteps - 1), entropy_NDist_16bins[:-1], label='entropy_NDist_16bins min = {0:3.3f}'.format(
    entropy_NDist_16bins[np.nonzero(entropy_NDist_16bins)].min()))
ax.set_xlabel('time steps', size=20)
ax.set_ylabel('entropy of NDist', size=20)
ax.set_title('entropy of NDist')
ax.legend()
plt.show()
figName = 'entropy of NDist'
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

fig, ax = plt.subplots(ncols=1, nrows=1)
ax.plot(np.arange(numOfTimeSteps - 1), klDiv_ODist[:-1], label='kldiv_ODist vs steps')
ax.set_xlabel('time steps', size=20)
ax.set_ylabel('KL divergence of ODist', size=20)
ax.set_title('KL divergence of ODist')
ax.set_yscale("log")
ax.legend()
plt.show()
figName = 'KL divergence of ODist'
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

# Histogram of orbiting distances
fig, ax = plt.subplots(ncols=1, nrows=1)
ax.plot(binEdgesOrbitingDistances[:-1],
        count_ODist[:, currStepNum] / count_ODist[:, currStepNum].sum(),
        label='ODist distribution of the last step')
ax.plot(binEdgesOrbitingDistances[:-1], target['count_ODist']/target['count_ODist'].sum(),
        label='target ODist distribution')
ax.set_xlabel('orbiting distances', size=20)
ax.set_ylabel('probability', size=20)
ax.set_title('histogram of orbiting distances')
ax.legend()
ax.set_yscale("log")
plt.show()
figName = 'Histogram of orbiting distances'
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
            yerr=hexaticOrderParameterModuliiStds[:-1], errorevery=int(numOfTimeSteps/10),
            label='psi6 norms and averages')
ax.set_xlabel('time step', size=20)
ax.set_ylabel('order parameter', size=20)
ax.set_title('hexatic order parameters over time')
ax.legend()
plt.show()
figName = 'hexatic order parameters over time'
fig.savefig(figName)

# Accepted based on probability
fig, ax = plt.subplots(ncols=1, nrows=1)
ax.hist(np.asarray(acceptedBasedOnProbs), bins=25, rwidth=0.8, label='accepted based on probability')
ax.set_xlabel('frame number', size=20)
ax.set_ylabel('counts', size=20)
ax.set_title('accepted based on probability')
ax.legend()
plt.show()
figName = 'Accepted based on probability'
fig.savefig(figName)

# save last frame
currentFrameBGR = fsr.draw_rafts_rh_coord(blankFrameBGR.copy(),
                                          np.int32(raftLocations[:, currStepNum, :] / scaleBar),
                                          np.int64(raftRadii / scaleBar), numOfRafts)
currentFrameBGR = fsr.draw_raft_num_rh_coord(currentFrameBGR,
                                             np.int64(raftLocations[:, currStepNum, :] / scaleBar),
                                             numOfRafts)
currentFrameBGR = fsr.draw_frame_info_many(currentFrameBGR, currStepNum, hexaticOrderParameterAvgNorms[currStepNum],
                                           hexaticOrderParameterModuliiAvgs[currStepNum], entropy_NDist[currStepNum])
outputImageName = 'MonteCarlo_{}Rafts_numOfSteps{}_currStepNum{}.jpg'.format(
    numOfRafts, numOfTimeSteps, currStepNum)
cv.imwrite(outputImageName, currentFrameBGR)

# save the frame with maximum hex order parameter:
stepNumOfMaxHex = hexaticOrderParameterModuliiAvgs.argmax()
currentFrameBGR = fsr.draw_rafts_rh_coord(blankFrameBGR.copy(),
                                          np.int32(raftLocations[:, stepNumOfMaxHex, :] / scaleBar),
                                          np.int64(raftRadii / scaleBar), numOfRafts)
currentFrameBGR = fsr.draw_raft_num_rh_coord(currentFrameBGR,
                                             np.int64(raftLocations[:, stepNumOfMaxHex, :] / scaleBar),
                                             numOfRafts)
currentFrameBGR = fsr.draw_frame_info_many(currentFrameBGR, stepNumOfMaxHex,
                                           hexaticOrderParameterAvgNorms[stepNumOfMaxHex],
                                           hexaticOrderParameterModuliiAvgs[stepNumOfMaxHex],
                                           entropy_NDist[stepNumOfMaxHex])
outputImageName = 'MonteCarlo_{}Rafts_numOfSteps{}_currStepNum{}.jpg'.format(
    numOfRafts, numOfTimeSteps, stepNumOfMaxHex)
cv.imwrite(outputImageName, currentFrameBGR)

# save files
shelfToSave = shelve.open('simulation_{}Rafts_{}rps'.format(numOfRafts, spinSpeed))
for key in dir():  # dir() gives all the names in the current scope
    if type(globals()[key]) == shelve.DbfilenameShelf:  # saving shelve object causes trouble
        print('Avoid shelving shelve object: {}'.format(key))
        continue
    elif key == 'exit' or key == 'get_ipython' or key == 'quit':
        print('IPython command not shelved: {}'.format(key))
        continue
    else:
        try:
            shelfToSave[key] = globals()[key]
        except TypeError:
            # __builtins__, my_shelf, and imported modules can not be shelved.
            print('TypeError shelving : {0}'.format(key))
shelfToSave.close()


#%% load existing simulation data shelve file
os.chdir(dataDir)
resultFolders = next(os.walk(dataDir))[1]
# resultFolders.sort()

resultFolderID = -1  # last folder
os.chdir(resultFolders[resultFolderID])

numOfRafts = 218
spinSpeed = 30
shelfToRead = shelve.open('simulation_{}Rafts_{}rps'.format(numOfRafts, spinSpeed), flag='r')
listOfVariablesInShelfToRead = list(shelfToRead.keys())
for key in shelfToRead:
    globals()[key] = shelfToRead[key]
shelfToRead.close()


