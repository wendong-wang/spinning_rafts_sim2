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


# %% reprocessing the result file
os.chdir(dataDir)
if parallel_mode == 1:
    numOfRafts = int(sys.argv[1])
    spinSpeed = int(sys.argv[2])
else:
    numOfRafts = 218
    spinSpeed = 25
numOfTimeSteps = 2000  # 80000
arenaSize = 1.5e4  # unit: micron
centerOfArena = np.array([arenaSize / 2, arenaSize / 2])
R = raftRadius = 1.5e2  # unit: micron
incrementSize = 50  # unit: radius
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
binEnd_ODist = 50  # unit: radius
binEdgesOrbitingDistances = list(np.arange(binStart_ODist, binEnd_ODist, binSize_ODist)) + [100]
binSize_XY = 2  # unit: radius
binEdgesX = list(np.arange(0, arenaSize/R + binSize_XY, binSize_XY))
binEdgesY = list(np.arange(0, arenaSize/R + binSize_XY, binSize_XY))

# load target distributions
tempShelf = shelve.open('target_' + str(numOfRafts) + "Rafts_" + str(spinSpeed) + 'rps')
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
outputFolderName = now.strftime("%Y-%m-%d_%H-%M-%S") + '_' + str(numOfRafts) + 'Rafts_' + str(spinSpeed) + \
                   'rps_exp-reprocessed'

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


# %% generating target dataset
# first run the previous section till the desired target pattern is generated
# currStepNum = 0
# raftLocationsOneFrame = raftLocations[:, currStepNum, :]  # directly simulated pattern, unit in micron

# count_NDist = target['count_NDist']
# count_X = target['count_X']
# count_Y = target['count_Y']
binEdgesNeighborDistances = target['binEdgesNeighborDistances']  # in unit of R
binEdgesOrbitingDistances = binEdgesNeighborDistances  # in R
binEdgesX = target['binEdgesX']  # in R
binEdgesY = target['binEdgesY']  # in R
raftLocations = target['raftLocations']  # in pixel
radiusInPixel = target['radius']  # R in pixel
raftRadius = radiusInPixel  # replace the original R, which is in micron
raftRadii = np.ones(numOfRafts) * raftRadius  # in Pixel
numOfFrames = target['numOfFrames']
arenaSizeInR = target['sizeOfArenaInRadius_pixels']  # arena size in R
arenaSizeInPixel = arenaSizeInR * radiusInPixel
arenaScaleFactor = arenaSizeInPixel / canvasSizeInPixel  # canvas size is 1000, arena size is about ~1720

# redeclare count variables
count_NDist_allFrames = np.zeros((len(binEdgesNeighborDistances)-1, numOfFrames))
count_NAngles_allFrames = np.zeros((len(binEdgesNeighborAngles)-1, numOfFrames))
# count_ODist_allFrames = np.zeros((len(binEdgesOrbitingDistances)-1, numOfFrames))
hexaticOrderParameterAvg = np.zeros(numOfFrames, dtype=np.csingle)
hexaticOrderParameterAvgNorm = np.zeros(numOfFrames)
hexaticOrderParameterModulii = np.zeros((numOfRafts, numOfFrames))
hexaticOrderParameterModuliiAvgs = np.zeros(numOfFrames)
hexaticOrderParameterModuliiStds = np.zeros(numOfFrames)

# draw the experimental image, make sure that you are in a newly created folder
currentFrameBGR = fsr.draw_rafts_rh_coord(blankFrameBGR.copy(),
                                          np.int32(raftLocations[:, -1, :] / arenaScaleFactor),
                                          np.int64(raftRadii / arenaScaleFactor), numOfRafts)
currentFrameBGR = fsr.draw_raft_num_rh_coord(currentFrameBGR,
                                             np.int64(raftLocations[:, -1, :] / arenaScaleFactor),
                                             numOfRafts)
outputFileName = 'Exp_' + str(numOfRafts) + 'Rafts'
outputImageName = outputFileName + '.jpg'
cv.imwrite(outputImageName, currentFrameBGR)

for currFrameNum in np.arange(numOfFrames):
    # use the raft location in one frame (last) to calculate all the distributions
    raftLocationsOneFrame = raftLocations[:, currFrameNum, :]  # get one frame, unit in pixel
    # collect count_NDist, count_NAngles, count_ODist
    # distribution by neighbor distances and neighbor angles
    neighborDistances, neighborAngles, hexOrderParas = fsr.neighbor_distances_angles_array(raftLocationsOneFrame)
    count_NDist_allFrames[:, currFrameNum], _ = np.histogram(neighborDistances / raftRadius, binEdgesNeighborDistances)
    count_NAngles_allFrames[:, currFrameNum], _ = np.histogram(neighborAngles, binEdgesNeighborAngles)
    hexaticOrderParameterAvg[currFrameNum] = hexOrderParas.mean()
    hexaticOrderParameterAvgNorm[currFrameNum] = np.sqrt(hexaticOrderParameterAvg[currFrameNum].real ** 2 +
                                                         hexaticOrderParameterAvg[currFrameNum].imag ** 2)
    hexaticOrderParameterModulii[:, currFrameNum] = np.absolute(hexOrderParas)
    hexaticOrderParameterModuliiAvgs[currFrameNum] = hexaticOrderParameterModulii[:, currFrameNum].mean()
    hexaticOrderParameterModuliiStds[currFrameNum] = hexaticOrderParameterModulii[:, currFrameNum].std()


# count_NAngles[0] -= numOfRafts
count_NDist = count_NDist_allFrames.sum(axis=1)
count_NAngles = count_NAngles_allFrames.sum(axis=1)
entropy_NDist = fsr.shannon_entropy(count_NDist)
entropy_NAngles = fsr.shannon_entropy(count_NAngles)


# distribution by orbiting distances
centerOfMass = raftLocationsOneFrame.mean(axis=0, keepdims=True)
orbitingDistances = scipy_distance.cdist(raftLocationsOneFrame, centerOfMass, 'euclidean')
count_ODist, _ = np.histogram(np.asarray(orbitingDistances) / raftRadius, binEdgesOrbitingDistances)
entropy_ODist = fsr.shannon_entropy(count_ODist)

# distribution by X
count_X, _ = np.histogram(raftLocations[:, :, 0] / raftRadius, binEdgesX)
entropy_X = fsr.shannon_entropy(count_X)

# distribution by y
count_Y, _ = np.histogram(raftLocations[:, :, 1] / raftRadius, binEdgesY)
entropy_Y = fsr.shannon_entropy(count_Y)

listOfVariablesToSave = ['numOfRafts', 'arenaSize', 'spinSpeed', 'arenaSizeInR',
                         'raftLocationsOneFrame', 'neighborDistances', 'neighborAngles', 'hexOrderParas',
                         'hexaticOrderParameterAvg', 'hexaticOrderParameterAvgNorm', 'hexaticOrderParameterModulii',
                         'hexaticOrderParameterModuliiAvgs', 'hexaticOrderParameterModuliiAvgs',
                         'orbitingDistances',
                         'binEdgesNeighborDistances', 'binEdgesOrbitingDistances', 'binEdgesNeighborAngles',
                         'binEdgesX', 'binEdgesY',
                         'entropy_NDist', 'count_NDist',
                         'entropy_NAngles', 'count_NAngles',
                         'entropy_ODist', 'count_ODist',
                         'entropy_X', 'count_X',
                         'entropy_Y', 'count_Y']
tempShelf = shelve.open('target_' + str(numOfRafts) + "Rafts_" + str(spinSpeed) + 'rps_exp-reprocessed')
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


# %% plotting for target distributions
# Histogram of target neighbor distances
fig, ax = plt.subplots(ncols=1, nrows=1)
ax.plot(binEdgesNeighborDistances[:-1], count_NDist / count_NDist.sum(),
        label='NDist distribution')
ax.set_xlabel('edge-edge distance', size=20)
ax.set_ylabel('probability', size=20)
ax.set_title('histogram of neighbor distances')
ax.legend()
plt.show()
figName = 'Histogram of neighbor distances'
fig.savefig(figName)

# Histogram of target neighbor angles
fig, ax = plt.subplots(ncols=1, nrows=1)
ax.plot(binEdgesNeighborAngles[:-1], count_NAngles / count_NAngles.sum(),
        label='NAngles distribution')
ax.set_xlabel('neighbor angles', size=20)
ax.set_ylabel('probability', size=20)
ax.set_title('histogram of neighbor angles')
ax.legend()
plt.show()
figName = 'Histogram of neighbor angles'
fig.savefig(figName)

# Histogram of target orbiting distances
fig, ax = plt.subplots(ncols=1, nrows=1)
ax.plot(binEdgesOrbitingDistances[:-1], count_ODist / count_ODist.sum(),
        label='ODist distribution')
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
ax.plot(binEdgesY[:-1], count_Y / count_Y.sum(), label='marginal distribution of y')
ax.set_xlabel('y', size=20)
ax.set_ylabel('probability', size=20)
ax.set_title('histogram of marginal distribution of y')
ax.legend()
plt.show()
figName = 'Histogram of marginal distribution of y'
fig.savefig(figName)

fig, ax = plt.subplots(ncols=1, nrows=1)
ax.scatter(np.arange(numOfFrames), hexaticOrderParameterAvgNorm,
           color='r', label='psi6 averages and norm')
ax.errorbar(np.arange(numOfFrames), hexaticOrderParameterModuliiAvgs,
            yerr=hexaticOrderParameterModuliiStds, errorevery=10, marker='o', label='psi6 norms and averages')
ax.set_xlabel('y', size=20)
ax.set_ylabel('order parameter', size=20)
ax.set_title('hexatic order parameters of the last frame')
ax.legend()
plt.show()
figName = 'hexatic order parameters of the last frame'
fig.savefig(figName)
