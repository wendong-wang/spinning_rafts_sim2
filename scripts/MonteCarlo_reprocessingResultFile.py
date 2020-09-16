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

# os.chdir(dataDir)
# %% reprocessing the result file
# enter the experimental data folder
expDuration = 20  # unit: second
numOfRafts = 218
spinSpeeds = [15, 20, 30, 40, 50, 60, 70]  # spin speeds for 20s data
expDataDir = os.path.join(projectDir, '2020-09-14_exp patterns', '{}s'.format(expDuration))
os.chdir(expDataDir)

# reading all the experimental data
targetList = []
for spinSpeed in spinSpeeds:
    # load target distributions
    shelfOfTarget = shelve.open('target_{}Rafts_{}rps'.format(numOfRafts, spinSpeed))
    variableListOfTargetDistributions = list(shelfOfTarget.keys())
    target = {}
    for key in shelfOfTarget:
        try:
            target[key] = shelfOfTarget[key]
        except TypeError:
            pass
    shelfOfTarget.close()
    targetList.append(target)

listOfAllShelves = []
for ssId, spinSpeed in enumerate(spinSpeeds):
    # load key data and parameters
    # ssId, spinSpeed = 0, 15
    numOfFrames = targetList[ssId]['numOfFrames']
    R = targetList[ssId]['radius']  # R in pixel,  make the assignment explicit
    raftRadii = np.ones(numOfRafts) * R  # in Pixel
    raftLocations = targetList[ssId]['raftLocations']  # in pixel
    outputFileName = 'target_{}s_{}Rafts_{}rps'.format(expDuration, numOfRafts, spinSpeed)

    # binEdgesNeighborDistances = targetList[ssId]['binEdgesNeighborDistances']  # in unit of R
    # binEdgesOrbitingDistances = binEdgesNeighborDistances  # in unit of R
    # binEdgesX = targetList[ssId]['binEdgesX']  # in unit of R
    # binEdgesY = targetList[ssId]['binEdgesY']  # in unit of R
    arenaSizeInR = targetList[ssId]['sizeOfArenaInRadius_pixels']  # arena size in R
    arenaSizeInPixel = arenaSizeInR * R

    # redefine binEdges if necessary:
    binSize_NDist = 0.5  # unit: R
    binStart_NDist = 2  # unit: R
    binEnd_NDist = 50  # unit: R
    binEdgesNeighborDistances = list(np.arange(binStart_NDist, binEnd_NDist, binSize_NDist)) + [100]
    binSize_NAngles = 10  # unit: deg
    binStart_NAngles = 0  # unit: deg
    binEnd_NAngles = 360  # unit: deg
    binEdgesNeighborAngles = list(np.arange(binStart_NAngles, binEnd_NAngles, binSize_NAngles)) + [360]
    binSize_ODist = 0.5  # unit: R
    binStart_ODist = 2  # unit: R
    binEnd_ODist = 80  # unit: R
    binEdgesOrbitingDistances = list(np.arange(binStart_ODist, binEnd_ODist, binSize_ODist)) + [100]
    binSize_XY = 2  # unit: R
    binEdgesX = list(np.arange(0, int(arenaSizeInR + binSize_XY), binSize_XY))
    binEdgesY = list(np.arange(0, int(arenaSizeInR + binSize_XY), binSize_XY))

    # drawing related parameters and variables
    canvasSizeInPixel = int(1000)  # unit: pixel
    arenaScaleFactor = arenaSizeInPixel / canvasSizeInPixel  # canvas size is 1000, arena size is about ~1720
    blankFrameBGR = np.ones((canvasSizeInPixel, canvasSizeInPixel, 3), dtype='int32') * 255
    currentFrameBGR = fsr.draw_rafts_rh_coord(blankFrameBGR.copy(),
                                              np.int32(raftLocations[:, -1, :] / arenaScaleFactor),
                                              np.int32(raftRadii / arenaScaleFactor), numOfRafts)
    currentFrameBGR = fsr.draw_raft_num_rh_coord(currentFrameBGR,
                                                 np.int32(raftLocations[:, -1, :] / arenaScaleFactor),
                                                 numOfRafts)
    outputImageName = outputFileName + '.jpg'
    cv.imwrite(outputImageName, currentFrameBGR)

    # declare count variables
    count_NDist_allFrames = np.zeros((len(binEdgesNeighborDistances) - 1, numOfFrames))
    count_NAngles_allFrames = np.zeros((len(binEdgesNeighborAngles) - 1, numOfFrames))
    count_ODist_allFrames = np.zeros((len(binEdgesOrbitingDistances)-1, numOfFrames))
    count_X_allFrames = np.zeros((len(binEdgesX) - 1, numOfFrames))
    count_Y_allFrames = np.zeros((len(binEdgesY) - 1, numOfFrames))

    # declare order parameters
    hexaticOrderParameterAvg = np.zeros(numOfFrames, dtype=np.csingle)
    hexaticOrderParameterAvgNorm = np.zeros(numOfFrames)
    hexaticOrderParameterModulii = np.zeros((numOfRafts, numOfFrames))
    hexaticOrderParameterModuliiAvgs = np.zeros(numOfFrames)
    hexaticOrderParameterModuliiStds = np.zeros(numOfFrames)

    for currFrameNum in progressbar.progressbar(np.arange(numOfFrames)):
        # use the raft location in one frame (last) to calculate all the distributions
        raftLocationsOneFrame = raftLocations[:, currFrameNum, :]  # get one frame, unit in pixel
        # collect count_NDist, count_NAngles, count_ODist
        # distribution by neighbor distances and neighbor angles
        neighborDistances, neighborAngles, hexOrderParas = fsr.neighbor_distances_angles_array(raftLocationsOneFrame)
        count_NDist_allFrames[:, currFrameNum], _ = np.histogram(neighborDistances / R, binEdgesNeighborDistances)
        count_NAngles_allFrames[:, currFrameNum], _ = np.histogram(neighborAngles, binEdgesNeighborAngles)
        hexaticOrderParameterAvg[currFrameNum] = hexOrderParas.mean()
        hexaticOrderParameterAvgNorm[currFrameNum] = np.sqrt(hexaticOrderParameterAvg[currFrameNum].real ** 2 +
                                                             hexaticOrderParameterAvg[currFrameNum].imag ** 2)
        hexaticOrderParameterModulii[:, currFrameNum] = np.absolute(hexOrderParas)
        hexaticOrderParameterModuliiAvgs[currFrameNum] = hexaticOrderParameterModulii[:, currFrameNum].mean()
        hexaticOrderParameterModuliiStds[currFrameNum] = hexaticOrderParameterModulii[:, currFrameNum].std()

        # distribution by orbiting distances
        centerOfMass = raftLocationsOneFrame.mean(axis=0, keepdims=True)
        orbitingDistances = scipy_distance.cdist(raftLocationsOneFrame, centerOfMass, 'euclidean')
        count_ODist_allFrames[:, currFrameNum], _ = np.histogram(np.asarray(orbitingDistances) / R,
                                                                 binEdgesOrbitingDistances)

        # marginal distributoin of X and Y
        count_X_allFrames[:, currFrameNum], _ = np.histogram(raftLocations[:, currFrameNum, 0] / R, binEdgesX)
        count_Y_allFrames[:, currFrameNum], _ = np.histogram(raftLocations[:, currFrameNum, 1] / R, binEdgesY)

    # distribution by NDist and NAngles
    count_NDist = count_NDist_allFrames.sum(axis=1)
    count_NAngles = count_NAngles_allFrames.sum(axis=1)
    count_NDist_lastFrame = count_NDist_allFrames[:, -1]
    count_NAngles_lastFrame = count_NAngles_allFrames[:, -1]
    entropy_NDist = fsr.shannon_entropy(count_NDist)  # entropy calculated on all frames
    entropy_NAngles = fsr.shannon_entropy(count_NAngles)  # entropy calculated on all frames

    # distribution by orbiting distances only last frame
    count_ODist = count_ODist_allFrames.sum(axis=1)
    count_ODist_lastFrame = count_ODist_allFrames[:, -1]
    entropy_ODist = fsr.shannon_entropy(count_ODist)  # entropy calculated based on the last frame

    # distribution by X
    count_X, _ = np.histogram(raftLocations[:, :, 0] / R, binEdgesX)
    count_X_lastFrame, _ = np.histogram(raftLocations[:, -1, 0] / R, binEdgesX)
    entropy_X = fsr.shannon_entropy(count_X)  # entropy calculated on all frames

    # distribution by y
    count_Y, _ = np.histogram(raftLocations[:, :, 1] / R, binEdgesY)
    count_Y_lastFrame, _ = np.histogram(raftLocations[:, -1, 1] / R, binEdgesY)
    entropy_Y = fsr.shannon_entropy(count_Y)  # entropy calculated on all frames

    # saving reprocessed result file
    listOfVariablesToSave = ['numOfRafts', 'arenaSizeInR', 'spinSpeed', 'raftLocationsOneFrame', 'numOfFrames',
                             'binEdgesNeighborDistances', 'binEdgesOrbitingDistances', 'binEdgesNeighborAngles',
                             'binEdgesX', 'binEdgesY',
                             'count_NDist', 'count_NDist_allFrames', 'count_NDist_lastFrame', 'entropy_NDist',
                             'count_NAngles', 'count_NAngles_allFrames', 'count_NAngles_lastFrame', 'entropy_NAngles',
                             'count_ODist', 'count_ODist_allFrames', 'count_ODist_lastFrame', 'entropy_ODist',
                             'count_X', 'count_X_allFrames', 'count_X_lastFrame', 'entropy_X',
                             'count_Y', 'count_Y_allFrames', 'count_Y_lastFrame', 'entropy_Y',
                             'hexaticOrderParameterAvg', 'hexaticOrderParameterAvgNorm', 'hexaticOrderParameterModulii',
                             'hexaticOrderParameterModuliiAvgs', 'hexaticOrderParameterModuliiAvgs']

    shelfToSave = shelve.open(outputFileName + '_reprocessed')
    for key in listOfVariablesToSave:
        try:
            shelfToSave[key] = globals()[key]
        except TypeError:
            #
            # __builtins__, tempShelf, and imported modules can not be shelved.
            #
            # print('ERROR shelving: {0}'.format(key))
            pass
    listOfAllShelves.append(dict(shelfToSave))
    shelfToSave.close()

    # plotting
    # Histogram of target neighbor distances
    fig, ax = plt.subplots(ncols=1, nrows=1)
    ax.plot(binEdgesNeighborDistances[:-1], count_NDist / count_NDist.sum(), label='all frames')
    ax.plot(binEdgesNeighborDistances[:-1], count_NDist_lastFrame / count_NDist_lastFrame.sum(), label='last frames')
    ax.set_xlabel('edge-edge distance', size=20)
    ax.set_ylabel('probability', size=20)
    ax.set_title('histogram of neighbor distances')
    ax.legend()
    plt.show()
    figName = outputFileName + '_histogram of neighbor distances'
    fig.savefig(figName)

    # Histogram of neighbor angles
    fig, ax = plt.subplots(ncols=1, nrows=1)
    ax.plot(binEdgesNeighborAngles[:-1], count_NAngles / count_NAngles.sum(), label='all frames')
    ax.plot(binEdgesNeighborAngles[:-1], count_NAngles_lastFrame / count_NAngles_lastFrame.sum(), label='last frames')
    ax.set_xlabel('neighbor angles', size=20)
    ax.set_ylabel('probability', size=20)
    ax.set_title('histogram of neighbor angles')
    ax.legend()
    plt.show()
    figName = outputFileName + '_histogram of neighbor angles'
    fig.savefig(figName)

    # Histogram of orbiting distances
    fig, ax = plt.subplots(ncols=1, nrows=1)
    ax.plot(binEdgesOrbitingDistances[:-1], count_ODist / count_ODist.sum(), label='all frames')
    ax.plot(binEdgesOrbitingDistances[:-1], count_ODist_lastFrame / count_ODist_lastFrame.sum(), label='last frames')
    ax.set_xlabel('radial distance r', size=20)
    ax.set_ylabel('probability', size=20)
    ax.set_title('histogram of orbiting distances')
    ax.legend()
    plt.show()
    figName = outputFileName + '_histogram of orbiting distances'
    fig.savefig(figName)

    # histogram of the marginal distribution of x
    fig, ax = plt.subplots(ncols=1, nrows=1)
    ax.plot(binEdgesX[:-1], count_X / count_X.sum(), label='all frames')
    ax.plot(binEdgesX[:-1], count_X_lastFrame / count_X_lastFrame.sum(), label='last frames')
    ax.set_xlabel('x (R)', size=20)
    ax.set_ylabel('probability', size=20)
    ax.set_title('histogram of marginal distribution of X')
    ax.legend()
    plt.show()
    figName = outputFileName + '_histogram of marginal distribution of X'
    fig.savefig(figName)

    # histogram of the marginal distribution of y
    fig, ax = plt.subplots(ncols=1, nrows=1)
    ax.plot(binEdgesX[:-1], count_Y / count_Y.sum(), label='all frames')
    ax.plot(binEdgesX[:-1], count_Y_lastFrame / count_Y_lastFrame.sum(), label='last frames')
    ax.set_xlabel('y (R)', size=20)
    ax.set_ylabel('probability', size=20)
    ax.set_title('histogram of marginal distribution of X')
    ax.legend()
    plt.show()
    figName = outputFileName + '_histogram of marginal distribution of X'
    fig.savefig(figName)

    # hexatic order parameters over time
    fig, ax = plt.subplots(ncols=1, nrows=1)
    ax.plot(np.arange(numOfFrames), hexaticOrderParameterAvgNorm, color='r', label='|<psi6>|')
    ax.errorbar(np.arange(numOfFrames), hexaticOrderParameterModuliiAvgs, yerr=hexaticOrderParameterModuliiStds,
                errorevery=int(numOfFrames/200), marker='o', label='<|psi6|>')
    ax.set_xlabel('y', size=20)
    ax.set_ylabel('order parameter', size=20)
    ax.set_title('hexatic order parameters')
    ax.legend()
    plt.show()
    figName = outputFileName + '_hexatic order parameters over time'
    fig.savefig(figName)

    # hexatic order parameter box plot of the last frame
    fig, ax = plt.subplots(ncols=1, nrows=1)
    ax.boxplot(hexaticOrderParameterModulii[:, -1])
    ax.set_xlabel('last frame', size=20)
    ax.set_ylabel('modulus of the order parameters', size=20)
    ax.set_title('box plot of the hexatic order parameters of the last frame')
    # ax.legend()
    plt.show()
    figName = outputFileName + '_hexatic order parameters last frame box plot'
    fig.savefig(figName)

with shelve.open('shelveAllSpinSpeeds') as tempShelf:
    tempShelf['dataList'] = listOfAllShelves

#%% compare the divergence between 20 rps and 30 rps
with shelve.open('shelveAllSpinSpeeds', flag='r') as tempShelf:
    listOfAllShelves = tempShelf['dataList']

spinSpeed = listOfAllShelves[5]['spinSpeed']
count_NDist_20rps = listOfAllShelves[1]['count_NDist']
count_NDist_30rps = listOfAllShelves[2]['count_NDist']
count_NDist_60rps = listOfAllShelves[5]['count_NDist']
klDiv_NDist_20_30 = fsr.kl_divergence(count_NDist_20rps, count_NDist_30rps)
klDiv_NDist_30_20 = fsr.kl_divergence(count_NDist_30rps, count_NDist_20rps)
klDiv_NDist_60_30 = fsr.kl_divergence(count_NDist_60rps, count_NDist_30rps)
klDiv_NDist_30_60 = fsr.kl_divergence(count_NDist_30rps, count_NDist_60rps)

count_NDist_lastFrame_20rps = listOfAllShelves[1]['count_NDist_lastFrame']
count_NDist_lastFrame_30rps = listOfAllShelves[2]['count_NDist_lastFrame']
count_NDist_lastFrame_60rps = listOfAllShelves[5]['count_NDist_lastFrame']
klDiv_NDist_20_30_lastFrame = fsr.kl_divergence(count_NDist_lastFrame_20rps, count_NDist_lastFrame_30rps)
klDiv_NDist_30_20_lastFrame = fsr.kl_divergence(count_NDist_lastFrame_30rps, count_NDist_lastFrame_20rps)
klDiv_NDist_60_30_lastFrame = fsr.kl_divergence(count_NDist_lastFrame_60rps, count_NDist_lastFrame_30rps)
klDiv_NDist_30_60_lastFrame = fsr.kl_divergence(count_NDist_lastFrame_30rps, count_NDist_lastFrame_60rps)

count_NAngles_20rps = listOfAllShelves[1]['count_NAngles']
count_NAngles_30rps = listOfAllShelves[2]['count_NAngles']
count_NAngles_60rps = listOfAllShelves[5]['count_NAngles']
klDiv_NAngles_20_30 = fsr.kl_divergence(count_NAngles_20rps, count_NAngles_30rps)
klDiv_NAngles_30_20 = fsr.kl_divergence(count_NAngles_30rps, count_NAngles_20rps)
klDiv_NAngles_60_30 = fsr.kl_divergence(count_NAngles_60rps, count_NAngles_30rps)
klDiv_NAngles_30_60 = fsr.kl_divergence(count_NAngles_30rps, count_NAngles_60rps)

count_NAngles_lastFrame_20rps = listOfAllShelves[1]['count_NAngles_lastFrame']
count_NAngles_lastFrame_30rps = listOfAllShelves[2]['count_NAngles_lastFrame']
count_NAngles_lastFrame_60rps = listOfAllShelves[5]['count_NAngles_lastFrame']
klDiv_NAngles_20_30_lastFrame = fsr.kl_divergence(count_NAngles_lastFrame_20rps, count_NAngles_lastFrame_30rps)
klDiv_NAngles_30_20_lastFrame = fsr.kl_divergence(count_NAngles_lastFrame_30rps, count_NAngles_lastFrame_20rps)
klDiv_NAngles_60_30_lastFrame = fsr.kl_divergence(count_NAngles_lastFrame_60rps, count_NAngles_lastFrame_30rps)
klDiv_NAngles_30_60_lastFrame = fsr.kl_divergence(count_NAngles_lastFrame_30rps, count_NAngles_lastFrame_60rps)

count_ODist_20rps = listOfAllShelves[1]['count_ODist']
count_ODist_30rps = listOfAllShelves[2]['count_ODist']
count_ODist_60rps = listOfAllShelves[5]['count_ODist']
klDiv_ODist_20_30 = fsr.kl_divergence(count_ODist_20rps, count_ODist_30rps)
klDiv_ODist_30_20 = fsr.kl_divergence(count_ODist_30rps, count_ODist_20rps)
klDiv_ODist_60_30 = fsr.kl_divergence(count_ODist_60rps, count_ODist_30rps)
klDiv_ODist_30_60 = fsr.kl_divergence(count_ODist_30rps, count_ODist_60rps)

count_ODist_lastFrame_20rps = listOfAllShelves[1]['count_ODist_lastFrame']
count_ODist_lastFrame_30rps = listOfAllShelves[2]['count_ODist_lastFrame']
count_ODist_lastFrame_60rps = listOfAllShelves[5]['count_ODist_lastFrame']
klDiv_ODist_20_30_lastFrame = fsr.kl_divergence(count_ODist_lastFrame_20rps, count_ODist_lastFrame_30rps)
klDiv_ODist_30_20_lastFrame = fsr.kl_divergence(count_ODist_lastFrame_30rps, count_ODist_lastFrame_20rps)
klDiv_ODist_60_30_lastFrame = fsr.kl_divergence(count_ODist_lastFrame_60rps, count_ODist_lastFrame_30rps)
klDiv_ODist_30_60_lastFrame = fsr.kl_divergence(count_ODist_lastFrame_30rps, count_ODist_lastFrame_60rps)

count_X_20rps = listOfAllShelves[1]['count_X']
count_X_30rps = listOfAllShelves[2]['count_X']
count_X_60rps = listOfAllShelves[5]['count_X']
klDiv_X_20_30 = fsr.kl_divergence(count_X_20rps, count_X_30rps)
klDiv_X_30_20 = fsr.kl_divergence(count_X_30rps, count_X_20rps)
klDiv_X_60_30 = fsr.kl_divergence(count_X_60rps, count_X_30rps)
klDiv_X_30_60 = fsr.kl_divergence(count_X_30rps, count_X_60rps)

count_X_lastFrame_20rps = listOfAllShelves[1]['count_X_lastFrame']
count_X_lastFrame_30rps = listOfAllShelves[2]['count_X_lastFrame']
count_X_lastFrame_60rps = listOfAllShelves[5]['count_X_lastFrame']
klDiv_X_20_30_lastFrame = fsr.kl_divergence(count_X_lastFrame_20rps, count_X_lastFrame_30rps)
klDiv_X_30_20_lastFrame = fsr.kl_divergence(count_X_lastFrame_30rps, count_X_lastFrame_20rps)
klDiv_X_60_30_lastFrame = fsr.kl_divergence(count_X_lastFrame_60rps, count_X_lastFrame_30rps)
klDiv_X_30_60_lastFrame = fsr.kl_divergence(count_X_lastFrame_30rps, count_X_lastFrame_60rps)

count_Y_20rps = listOfAllShelves[1]['count_Y']
count_Y_30rps = listOfAllShelves[2]['count_Y']
count_Y_60rps = listOfAllShelves[5]['count_Y']
klDiv_Y_20_30 = fsr.kl_divergence(count_Y_20rps, count_Y_30rps)
klDiv_Y_30_20 = fsr.kl_divergence(count_Y_30rps, count_Y_20rps)
klDiv_Y_60_30 = fsr.kl_divergence(count_Y_60rps, count_Y_30rps)
klDiv_Y_30_60 = fsr.kl_divergence(count_Y_30rps, count_Y_60rps)

count_Y_lastFrame_20rps = listOfAllShelves[1]['count_Y_lastFrame']
count_Y_lastFrame_30rps = listOfAllShelves[2]['count_Y_lastFrame']
count_Y_lastFrame_60rps = listOfAllShelves[5]['count_Y_lastFrame']
klDiv_Y_20_30_lastFrame = fsr.kl_divergence(count_Y_lastFrame_20rps, count_Y_lastFrame_30rps)
klDiv_Y_30_20_lastFrame = fsr.kl_divergence(count_Y_lastFrame_30rps, count_Y_lastFrame_20rps)
klDiv_Y_60_30_lastFrame = fsr.kl_divergence(count_Y_lastFrame_60rps, count_Y_lastFrame_30rps)
klDiv_Y_30_60_lastFrame = fsr.kl_divergence(count_Y_lastFrame_30rps, count_Y_lastFrame_60rps)

#%% see how kl divergences with time average:
listID = 5
spinSpeed = listOfAllShelves[listID]['spinSpeed']   # spinSpeed
count_NDist_singleRPS = listOfAllShelves[listID]['count_NDist']
count_NDist_lastFrame_singleRPS = listOfAllShelves[listID]['count_NDist_lastFrame']
count_NDist_allFrames_singleRPS = listOfAllShelves[listID]['count_NDist_allFrames']

count_NAngles_singleRPS = listOfAllShelves[listID]['count_NAngles']
count_NAngles_lastFrame_singleRPS = listOfAllShelves[listID]['count_NAngles_lastFrame']
count_NAngles_allFrames_singleRPS = listOfAllShelves[listID]['count_NAngles_allFrames']

count_ODist_singleRPS = listOfAllShelves[listID]['count_ODist']
count_ODist_lastFrame_singleRPS = listOfAllShelves[listID]['count_ODist_lastFrame']
count_ODist_allFrames_singleRPS = listOfAllShelves[listID]['count_ODist_allFrames']

count_X_singleRPS = listOfAllShelves[listID]['count_X']
count_X_lastFrame_singleRPS = listOfAllShelves[listID]['count_X_lastFrame']
count_X_allFrames_singleRPS = listOfAllShelves[listID]['count_X_allFrames']

count_Y_singleRPS = listOfAllShelves[listID]['count_Y']
count_Y_lastFrame_singleRPS = listOfAllShelves[listID]['count_Y_lastFrame']
count_Y_allFrames_singleRPS = listOfAllShelves[listID]['count_Y_allFrames']

numOfFrames = listOfAllShelves[1]['numOfFrames']
klDiv_NDist_avg_eachFrame = np.zeros(numOfFrames)
klDiv_NDist_lastFrame_eachFrame = np.zeros(numOfFrames)
klDiv_NAngles_avg_eachFrame = np.zeros(numOfFrames)
klDiv_NAngles_lastFrame_eachFrame = np.zeros(numOfFrames)
klDiv_ODist_avg_eachFrame = np.zeros(numOfFrames)
klDiv_ODist_lastFrame_eachFrame = np.zeros(numOfFrames)
klDiv_X_avg_eachFrame = np.zeros(numOfFrames)
klDiv_X_lastFrame_eachFrame = np.zeros(numOfFrames)
klDiv_Y_avg_eachFrame = np.zeros(numOfFrames)
klDiv_Y_lastFrame_eachFrame = np.zeros(numOfFrames)

for currFrameNum in np.arange(numOfFrames):
    klDiv_NDist_avg_eachFrame[currFrameNum] = \
        fsr.kl_divergence(count_NDist_allFrames_singleRPS[:, currFrameNum], count_NDist_singleRPS)
    klDiv_NDist_lastFrame_eachFrame[currFrameNum] = \
        fsr.kl_divergence(count_NDist_allFrames_singleRPS[:, currFrameNum], count_NDist_lastFrame_singleRPS)

    klDiv_NAngles_avg_eachFrame[currFrameNum] = \
        fsr.kl_divergence(count_NAngles_allFrames_singleRPS[:, currFrameNum], count_NAngles_singleRPS)
    klDiv_NAngles_lastFrame_eachFrame[currFrameNum] = \
        fsr.kl_divergence(count_NAngles_allFrames_singleRPS[:, currFrameNum], count_NAngles_lastFrame_singleRPS)

    klDiv_ODist_avg_eachFrame[currFrameNum] = \
        fsr.kl_divergence(count_ODist_allFrames_singleRPS[:, currFrameNum], count_ODist_singleRPS)
    klDiv_ODist_lastFrame_eachFrame[currFrameNum] = \
        fsr.kl_divergence(count_ODist_allFrames_singleRPS[:, currFrameNum], count_ODist_lastFrame_singleRPS)

    klDiv_X_avg_eachFrame[currFrameNum] = \
        fsr.kl_divergence(count_X_allFrames_singleRPS[:, currFrameNum], count_X_singleRPS)
    klDiv_X_lastFrame_eachFrame[currFrameNum] = \
        fsr.kl_divergence(count_X_allFrames_singleRPS[:, currFrameNum], count_X_lastFrame_singleRPS)

    klDiv_Y_avg_eachFrame[currFrameNum] = \
        fsr.kl_divergence(count_Y_allFrames_singleRPS[:, currFrameNum], count_Y_singleRPS)
    klDiv_Y_lastFrame_eachFrame[currFrameNum] = \
        fsr.kl_divergence(count_Y_allFrames_singleRPS[:, currFrameNum], count_Y_lastFrame_singleRPS)

# plotting
fig, ax = plt.subplots(ncols=1, nrows=1)
ax.plot(np.arange(numOfFrames), klDiv_NDist_avg_eachFrame, color='r',
        label='eachFrame against allFrames: {0:.3f} +/- {1:.3f}'.format(klDiv_NDist_avg_eachFrame.mean(),
                                                                        klDiv_NDist_avg_eachFrame.std()))
ax.plot(np.arange(numOfFrames), klDiv_NDist_lastFrame_eachFrame, color='b',
        label='eachFrame against lastFrame: {0:.3f} +/- {1:.3f}'.format(klDiv_NDist_lastFrame_eachFrame.mean(),
                                                                        klDiv_NDist_lastFrame_eachFrame.std()))
ax.set_xlabel('frame number', size=20)
ax.set_ylabel('kl divergence', size=20)
ax.set_title('kl divergence of NDist at {} rps'.format(spinSpeed))
ax.legend()
plt.show()
figName = 'kl divergence_{}rps_NDist.jpg'.format(spinSpeed)
fig.savefig(figName)

fig, ax = plt.subplots(ncols=1, nrows=1)
ax.plot(np.arange(numOfFrames), klDiv_NAngles_avg_eachFrame, color='r',
        label='eachFrame against allFrames: {0:.3f} +/- {1:.3f}'.format(klDiv_NAngles_avg_eachFrame.mean(),
                                                                        klDiv_NAngles_avg_eachFrame.std()))
ax.plot(np.arange(numOfFrames), klDiv_NAngles_lastFrame_eachFrame, color='b',
        label='eachFrame against lastFrame: {0:.3f} +/- {1:.3f}'.format(klDiv_NAngles_lastFrame_eachFrame.mean(),
                                                                        klDiv_NAngles_lastFrame_eachFrame.std()))
ax.set_xlabel('frame number', size=20)
ax.set_ylabel('kl divergence', size=20)
ax.set_title('kl divergence of NAngles at {} rps'.format(spinSpeed))
ax.legend()
plt.show()
figName = 'kl divergence_{}rps_NAngles.jpg'.format(spinSpeed)
fig.savefig(figName)

fig, ax = plt.subplots(ncols=1, nrows=1)
ax.plot(np.arange(numOfFrames), klDiv_ODist_avg_eachFrame, color='r',
        label='eachFrame against allFrames: {0:.3f} +/- {1:.3f}'.format(klDiv_ODist_avg_eachFrame.mean(),
                                                                        klDiv_ODist_avg_eachFrame.std()))
ax.plot(np.arange(numOfFrames), klDiv_ODist_lastFrame_eachFrame, color='b',
        label='eachFrame against lastFrame: {0:.3f} +/- {1:.3f}'.format(klDiv_ODist_lastFrame_eachFrame.mean(),
                                                                        klDiv_ODist_lastFrame_eachFrame.std()))
ax.set_xlabel('frame number', size=20)
ax.set_ylabel('kl divergence', size=20)
ax.set_title('kl divergence of ODist at {} rps'.format(spinSpeed))
ax.legend()
plt.show()
figName = 'kl divergence_{}rps_ODist.jpg'.format(spinSpeed)
fig.savefig(figName)

fig, ax = plt.subplots(ncols=1, nrows=1)
ax.plot(np.arange(numOfFrames), klDiv_X_avg_eachFrame, color='r',
        label='eachFrame against allFrames: {0:.3f} +/- {1:.3f}'.format(klDiv_X_avg_eachFrame.mean(),
                                                                        klDiv_X_avg_eachFrame.std()))
ax.plot(np.arange(numOfFrames), klDiv_X_lastFrame_eachFrame, color='b',
        label='eachFrame against lastFrame: {0:.3f} +/- {1:.3f}'.format(klDiv_X_lastFrame_eachFrame.mean(),
                                                                        klDiv_X_lastFrame_eachFrame.std()))
ax.set_xlabel('frame number', size=20)
ax.set_ylabel('kl divergence', size=20)
ax.set_title('kl divergence of X at {} rps'.format(spinSpeed))
ax.legend()
plt.show()
figName = 'kl divergence_{}rps_X.jpg'.format(spinSpeed)
fig.savefig(figName)

fig, ax = plt.subplots(ncols=1, nrows=1)
ax.plot(np.arange(numOfFrames), klDiv_Y_avg_eachFrame, color='r',
        label='eachFrame against allFrames: {0:.3f} +/- {1:.3f}'.format(klDiv_Y_avg_eachFrame.mean(),
                                                                        klDiv_Y_avg_eachFrame.std()))
ax.plot(np.arange(numOfFrames), klDiv_Y_lastFrame_eachFrame, color='b',
        label='eachFrame against lastFrame: {0:.3f} +/- {1:.3f}'.format(klDiv_Y_lastFrame_eachFrame.mean(),
                                                                        klDiv_Y_lastFrame_eachFrame.std()))
ax.set_xlabel('frame number', size=20)
ax.set_ylabel('kl divergence', size=20)
ax.set_title('kl divergence of Y at {} rps'.format(spinSpeed))
ax.legend()
plt.show()
figName = 'kl divergence_{}rps_Y.jpg'.format(spinSpeed)
fig.savefig(figName)
