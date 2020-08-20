"""
This is for the Monte Carlo simulation of the configuration of many rafts
The maximum characters per line is set to be 120.

The algorithm is as follows:
randomly put ~200 rafts in a square
for each step do
    calculate the distribution of neighbor distances (count_NDist) and marginal distribution of x (count_X)
    calculate the KL divergence between the simulated distributions and perfect hexagonal one (klDiv_NDist and klDiv_X)
    for each raft do
        generate a random movement vector moveInXY, uniformly sampled from [-R, R]^2
        move the raft
        if the new position is outside the arena or the new position is within 2R of other rafts
            move back the raft
            continue the for loop of next raft
        calculate count_NDist and count_X of the new configuration
        calculate klDiv_NDist and klDiv_X
        if klDiv_NDist OR klDiv_X decreases (this is the key conditional statement)
            continue the for loop for the next raft
        else
            move back the raft
    end for
end for


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
numOfRafts = 50
numOfTimeSteps = 600
arenaSize = 1.5e4  # unit: micron
centerOfArena = np.array([arenaSize / 2, arenaSize / 2])
R = raftRadius = 1.5e2  # unit: micron
binSize = 0.5  # unit: radius
binStart = 2  # unit: radius
binEnd_NDist = 50  # unit: radius
binEnd_ODist = 80  # unit: radius
binEdgesNeighborDistances = list(np.arange(binStart, binEnd_NDist, binSize)) + [100]
binEdgesOrbitingDistances = list(np.arange(binStart, binEnd_ODist, binSize)) + [100]
binEdgesX = list(np.arange(0, arenaSize/R, binSize))
binEdgesY = list(np.arange(0, arenaSize/R, binSize))

# load target distributions
os.chdir(dataDir)
tempShelf = shelve.open('targetDistributions')
variableListOfTargetDistributions = list(tempShelf.keys())
target = {}
for key in tempShelf:
    try:
        target[key] = tempShelf[key]
    except TypeError:
        pass
tempShelf.close()

# make folder for the current dataset
now = datetime.datetime.now()
if parallel_mode == 1:
    outputFolderName = now.strftime("%Y-%m-%d") + '_' + str(numOfRafts) + 'Rafts_' + \
                   'totalSteps' + str(numOfTimeSteps)
else:
    outputFolderName = now.strftime("%Y-%m-%d_%H-%M-%S") + '_' + str(numOfRafts) + 'Rafts_' + \
                       'totalSteps' + str(numOfTimeSteps)

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
count_ODist = np.zeros((len(binEdgesOrbitingDistances)-1, numOfTimeSteps))
count_X = np.zeros((len(binEdgesX)-1, numOfTimeSteps))
count_Y = np.zeros((len(binEdgesY)-1, numOfTimeSteps))
klDiv_NDist = np.zeros(numOfTimeSteps)
klDiv_ODist = np.zeros(numOfTimeSteps)
klDiv_X = np.zeros(numOfTimeSteps)
klDiv_Y = np.zeros(numOfTimeSteps)
entropy_NDist = np.zeros(numOfTimeSteps)
entropy_ODist = np.zeros(numOfTimeSteps)
entropy_X = np.zeros(numOfTimeSteps)
entropy_Y = np.zeros(numOfTimeSteps)
rejectionRates = np.zeros(numOfTimeSteps)

# initialize rafts positions: 1 - random positions, 2 - fixed initial position,
# 3 - hexagonal fixed position
initialPositionMethod = 1
currStepNum = 0
if initialPositionMethod == 1:
    paddingAroundArena = 5  # unit: R
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
    raftLocations[:, currStepNum, :] = fsr.hexagonal_spiral(numOfRafts, raftRadius * 2 + 400, centerOfArena)

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

for currStepNum in progressbar.progressbar(np.arange(0, numOfTimeSteps - 1)):
    dict_NDist = fsr.count_kldiv_entropy_ndist(raftLocations[:, currStepNum, :], raftRadius,
                                               binEdgesNeighborDistances, target)
    dict_X = fsr.count_kldiv_entropy_x(raftLocations[:, currStepNum, :], raftRadius, binEdgesX, target)
    dict_Y = fsr.count_kldiv_entropy_y(raftLocations[:, currStepNum, :], raftRadius, binEdgesY, target)

    # assignments
    count_NDist[:, currStepNum], klDiv_NDist[currStepNum], entropy_NDist[currStepNum] = \
        dict_NDist['count_NDist'], dict_NDist['klDiv_NDist'], dict_NDist['entropy_NDist']
    count_X[:, currStepNum], klDiv_X[currStepNum], entropy_X[currStepNum] = \
        dict_X['count_X'], dict_X['klDiv_X'], dict_X['entropy_X']
    count_Y[:, currStepNum], klDiv_Y[currStepNum], entropy_Y[currStepNum] = \
        dict_Y['count_Y'], dict_Y['klDiv_Y'], dict_Y['entropy_Y']

    newLocations = raftLocations[:, currStepNum, :].copy()
    for raftID in np.arange(numOfRafts):
        # raftID = 0
        moveInXY = np.random.uniform(low=-1, high=1, size=2) * R
        # take care of the cases where moving the rafts outside the arena or overlapping with another raft.
        newXY = newLocations[raftID, :] + moveInXY
        while newXY.max() > arenaSize or newXY.min() < 0 or \
              scipy_distance.cdist(newLocations[np.arange(numOfRafts) != raftID, :],
                                   newXY.reshape(1, 2)).min() < 2 * R:
            moveInXY = np.random.uniform(low=-1, high=1, size=2) * R
            newXY = newLocations[raftID, :] + moveInXY
        newLocations[raftID, :] = newXY

        dict_counts = fsr.count_distribution(newLocations, raftRadius, binEdgesNeighborDistances,
                                             binEdgesX, binEdgesY)
        dict_klDiv = fsr.divergences_curr_target(dict_counts, target)

        # if the selected divergences decreases, then accept the move, otherwise reject the move
        if (dict_klDiv["klDiv_NDist"] <= klDiv_NDist[currStepNum]) and \
                (dict_klDiv["klDiv_X"] <= klDiv_X[currStepNum]):
            continue
        else:
            newLocations[raftID, :] = newLocations[raftID, :] - moveInXY
            rejectionRates[currStepNum] += 1

    raftLocations[:, currStepNum + 1, :] = newLocations


# %% saving dataset
listOfVariablesToSave = ['numOfRafts', 'numOfTimeSteps', 'arenaSize', 'binSize',
                         'raftLocations', 'neighborDistancesList', 'orbitingDistances',
                         'entropyByNeighborDistances', 'count_NDist',
                         'entropyByOrbitingDistances', 'count_ODist',
                         'entropyByX', 'count_X',
                         'entropyByY', 'count_Y']
tempShelf = shelve.open('variables')
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


# %% plotting
fig, ax = plt.subplots(ncols=1, nrows=1)
ax.plot(np.arange(numOfTimeSteps - 1), klDiv_NDist[:-1], label='kldiv_NDist vs steps')
ax.set_xlabel('time steps', size=20)
ax.set_ylabel('KL divergence of NDist', size=20)
ax.set_title('KL divergence of NDist')
ax.legend()
plt.show()
figName = 'KL divergence of NDist'
fig.savefig(figName)

fig, ax = plt.subplots(ncols=1, nrows=1)
ax.plot(np.arange(numOfTimeSteps - 1), klDiv_X[:-1], label='kldiv_X vs steps')
ax.set_xlabel('time steps', size=20)
ax.set_ylabel('KL divergence of X', size=20)
ax.set_title('KL divergence of X')
ax.legend()
plt.show()
figName = 'KL divergence of X'
fig.savefig(figName)

fig, ax = plt.subplots(ncols=1, nrows=1)
ax.plot(np.arange(binStart, binEnd_NDist, binSize), count_NDist / count_NDist.sum(), label='NDist distribution')
ax.set_xlabel('edge-edge distance', size=20)
ax.set_ylabel('probability', size=20)
ax.set_title('histogram of neighbor distances')
ax.legend()
plt.show()
figName = 'Histogram of neighbor distances'
fig.savefig(figName)

fig, ax = plt.subplots(ncols=1, nrows=1)
ax.plot(np.arange(binStart, binEnd_ODist, binSize), count_ODist / count_ODist.sum(), label='ODist distribution')
# ax.plot(np.arange(binStart, binEnd_ODist, binSize), count_ODist / count_ODist.sum() /
#         (0.5*binEdgesOrbitingDistances[0:-1] + 0.5*binEdgesOrbitingDistances[1:]), label='normal ODist distribution')
ax.set_xlabel('radial distance r', size=20)
ax.set_ylabel('probability', size=20)
ax.set_title('histogram of orbiting distances')
ax.legend()
plt.show()
figName = 'Histogram of orbiting distances'
fig.savefig(figName)

fig, ax = plt.subplots(ncols=1, nrows=1)
ax.plot(binEdgesX[:-1], count_X / count_X.sum(), label='marginal distribution of x')
ax.set_xlabel('x', size=20)
ax.set_ylabel('probability', size=20)
ax.set_title('histogram of marginal distribution of X')
ax.legend()
plt.show()
figName = 'Histogram of marginal distribution of X'
fig.savefig(figName)

fig, ax = plt.subplots(ncols=1, nrows=1)
ax.plot(binEdgesX[:-1], count_X / count_X.sum(), label='marginal distribution of y')
ax.set_xlabel('y', size=20)
ax.set_ylabel('probability', size=20)
ax.set_title('histogram of marginal distribution of y')
ax.legend()
plt.show()
figName = 'Histogram of marginal distribution of y'
fig.savefig(figName)

#%% old snippets, may or may not be useful
