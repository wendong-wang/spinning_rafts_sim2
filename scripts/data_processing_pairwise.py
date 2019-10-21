"""
This is for the simulation of pairwise interactions.
The maximum characters per line is set to be 120.

"""

import glob
import os
import shelve

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
from scripts.functions_spinning_rafts import fft_general, adjust_phases


projectDir = "D:\\simulationFolder\\spinning_rafts_simulation_code"  # os.getcwd()
os.chdir("..")
outputDir = os.getcwd()

# %% load simulated data in one main folder
os.chdir(outputDir)
rootFolderTreeGen = os.walk(outputDir)
_, mainFolders, _ = next(rootFolderTreeGen)

mainFolderID = 40
os.chdir(mainFolders[mainFolderID])

dataFileList = glob.glob('*.dat')
dataFileList.sort()

mainDataList = []
variableListsForAllMainData = []

for dataID in range(len(dataFileList)):
    dataFileToLoad = dataFileList[dataID].partition('.dat')[0]

    tempShelf = shelve.open(dataFileToLoad)
    variableListOfOneMainDataFile = list(tempShelf.keys())

    expDict = {}
    for key in tempShelf:
        try:
            expDict[key] = tempShelf[key]
        except TypeError:
            pass

    tempShelf.close()
    mainDataList.append(expDict)
    variableListsForAllMainData.append(variableListOfOneMainDataFile)

# %% pairwise data treatment and output to csv

timeStepSize = mainDataList[0]['timeStepSize']  # unit: s, assuming the same for all data in the list
samplingRate = 1 / timeStepSize  # unit fps
diameterOfRaftInMicron = 300  # micron
startOfSamplingStep = 9000  # check the variable mainDataList[0]['numOfTimeSteps']
# diameterOfRaftInPixel = 146 # pixel 124 for 2x mag, 146 for 2.5x object,
# scaleBar = diameterOfRaftInMicron/diameterOfRaftInPixel # micron per pixel.
# 300 micron = 124 pixel -> 2x objective, 300 micron = 146 pixel -> 2.5x objective

# initialize data frames
varsForMainData = ['mainFolderName', 'experimentName', 'batchNum', 'magneticFieldRotationRPS',
                   'distancesMean', 'distancesSTD', 'orbitingSpeedsMean', 'orbitingSpeedsSTD',
                   'raft1SpinSpeedsMean', 'raft1SpinSpeedsSTD', 'raft2SpinSpeedsMean', 'raft2SpinSpeedsSTD']
dfMainData = pd.DataFrame(columns=varsForMainData, index=range(len(mainDataList)))

dfFFTDist = pd.DataFrame(columns=['fDistances'])

dfFFTOrbitingSpeeds = pd.DataFrame(columns=['fOrbitingSpeeds'])

dfFFTRaft1Spin = pd.DataFrame(columns=['fRaft1SpinSpeeds'])

dfFFTRaft2Spin = pd.DataFrame(columns=['fRaft2SpinSpeeds'])

for dataID in range(len(mainDataList)):
    raft1Locations = mainDataList[dataID]['raftLocations'][0, startOfSamplingStep:, :]
    raft2Locations = mainDataList[dataID]['raftLocations'][1, startOfSamplingStep:, :]

    vector1To2 = raft2Locations - raft1Locations

    distances = np.sqrt(vector1To2[:, 0] ** 2 + vector1To2[:, 1] ** 2)
    distancesMean = distances.mean()  #
    distancesSTD = np.std(distances)

    fDistances, pDistances = fft_general(samplingRate, distances)

    phase1To2 = np.arctan2(vector1To2[:, 1], vector1To2[:, 0]) * 180 / np.pi
    # note that the sign of y is flipped, so as to keep the coordination in the right-handed coordinate
    phasesAjusted = adjust_phases(phase1To2)
    orbitingSpeeds = np.gradient(phasesAjusted) * samplingRate / 180 * np.pi
    orbitingSpeedsMean = orbitingSpeeds.mean()
    orbitingSpeedsSTD = orbitingSpeeds.std()

    fOrbitingSpeeds, pOrbitingSpeeds = fft_general(samplingRate, orbitingSpeeds)

    raft1Orientations = mainDataList[dataID]['raftOrientations'][0, startOfSamplingStep:]
    raft2Orientations = mainDataList[dataID]['raftOrientations'][1, startOfSamplingStep:]
    raft1OrientationsAdjusted = adjust_phases(raft1Orientations)
    raft2OrientationsAdjusted = adjust_phases(raft2Orientations)
    raft1SpinSpeeds = np.gradient(raft1OrientationsAdjusted) * samplingRate / 360
    raft2SpinSpeeds = np.gradient(raft2OrientationsAdjusted) * samplingRate / 360
    raft1SpinSpeedsMean = raft1SpinSpeeds.mean()
    raft2SpinSpeedsMean = raft2SpinSpeeds.mean()
    raft1SpinSpeedsSTD = raft1SpinSpeeds.std()
    raft2SpinSpeedsSTD = raft2SpinSpeeds.std()

    fRaft1SpinSpeeds, pRaft1SpinSpeeds = fft_general(samplingRate, raft1SpinSpeeds)
    fRaft2SpinSpeeds, pRaft2SpinSpeeds = fft_general(samplingRate, raft2SpinSpeeds)

    # store in dataframes
    dfMainData.loc[dataID, 'mainFolderName'] = mainFolders[mainFolderID]
    # if mainDataList[dataID]['isVideo'] == 0:
    #     dfMainData.loc[dataID,'experimentName'] = mainDataList[dataID]['subfolders'][mainDataList[dataID]['expID']]
    # elif mainDataList[dataID]['isVideo'] == 1:
    #     dfMainData.loc[dataID,'experimentName'] = mainDataList[dataID]['videoFileList'][mainDataList[dataID]['expID']]
    # dfMainData.loc[dataID,'batchNum'] = mainDataList[dataID]['batchNum']
    dfMainData.loc[dataID, 'magneticFieldRotationRPS'] = - mainDataList[dataID]['magneticFieldRotationRPS']
    dfMainData.loc[dataID, 'distancesMean'] = distancesMean - diameterOfRaftInMicron
    dfMainData.loc[dataID, 'distancesSTD'] = distancesSTD
    dfMainData.loc[dataID, 'orbitingSpeedsMean'] = -orbitingSpeedsMean
    dfMainData.loc[dataID, 'orbitingSpeedsSTD'] = orbitingSpeedsSTD
    dfMainData.loc[dataID, 'raft1SpinSpeedsMean'] = -raft1SpinSpeedsMean
    dfMainData.loc[dataID, 'raft1SpinSpeedsSTD'] = raft1SpinSpeedsSTD
    dfMainData.loc[dataID, 'raft2SpinSpeedsMean'] = -raft2SpinSpeedsMean
    dfMainData.loc[dataID, 'raft2SpinSpeedsSTD'] = raft2SpinSpeedsSTD

    if len(dfFFTDist) == 0:
        dfFFTDist['fDistances'] = fDistances
    # colName = str(mainDataList[dataID]['batchNum']) + '_' \
    #           + str(mainDataList[dataID]['magneticFieldRotationRPS']).zfill(4)
    colName = str(-mainDataList[dataID]['magneticFieldRotationRPS']).zfill(4)
    dfFFTDist[colName] = pDistances

    if len(dfFFTOrbitingSpeeds) == 0:
        dfFFTOrbitingSpeeds['fOrbitingSpeeds'] = fOrbitingSpeeds
    dfFFTOrbitingSpeeds[colName] = pOrbitingSpeeds

    if len(dfFFTRaft1Spin) == 0:
        dfFFTRaft1Spin['fRaft1SpinSpeeds'] = fRaft1SpinSpeeds
    dfFFTRaft1Spin[colName] = pRaft1SpinSpeeds

    if len(dfFFTRaft2Spin) == 0:
        dfFFTRaft2Spin['fRaft2SpinSpeeds'] = fRaft2SpinSpeeds
    dfFFTRaft2Spin[colName] = pRaft2SpinSpeeds

dfMainData = dfMainData.infer_objects()
# dfMainData.sort_values(by = ['batchNum','magneticFieldRotationRPS'], ascending = [True, False], inplace = True)
dfMainData.sort_values(by=['magneticFieldRotationRPS'], ascending=[False], inplace=True)

dfFFTDist = dfFFTDist.infer_objects()
dfFFTOrbitingSpeeds = dfFFTOrbitingSpeeds.infer_objects()
dfFFTRaft1Spin = dfFFTRaft1Spin.infer_objects()
dfFFTRaft2Spin = dfFFTRaft2Spin.infer_objects()

dfFFTDist = dfFFTDist.reindex(sorted(dfFFTDist.columns, reverse=True), axis='columns')
dfFFTOrbitingSpeeds = dfFFTOrbitingSpeeds.reindex(sorted(dfFFTOrbitingSpeeds.columns, reverse=True), axis='columns')
dfFFTRaft1Spin = dfFFTRaft1Spin.reindex(sorted(dfFFTRaft1Spin.columns, reverse=True), axis='columns')
dfFFTRaft2Spin = dfFFTRaft2Spin.reindex(sorted(dfFFTRaft2Spin.columns, reverse=True), axis='columns')

dfMainData.plot.scatter(x='magneticFieldRotationRPS', y='distancesMean')

# output to csv files
mainDataFileName = mainFolders[mainFolderID]
colNames = ['batchNum', 'magneticFieldRotationRPS',
            'distancesMean', 'distancesSTD', 'orbitingSpeedsMean', 'orbitingSpeedsSTD',
            'raft1SpinSpeedsMean', 'raft1SpinSpeedsSTD', 'raft2SpinSpeedsMean', 'raft2SpinSpeedsSTD']
dfMainData.to_csv(mainDataFileName + '.csv', index=False, columns=colNames)

# BFieldStrength = '10mT'
BFieldStrength = str(mainDataList[0]['magneticFieldStrength'] * 1000).zfill(4) + 'mT'
dfFFTDist.to_csv('fft_' + BFieldStrength + '_distance.csv', index=False)
dfFFTOrbitingSpeeds.to_csv('fft_' + BFieldStrength + '_orbitingSpeeds.csv', index=False)
dfFFTRaft1Spin.to_csv('fft_' + BFieldStrength + '_raft1SpinSpeeds.csv', index=False)
dfFFTRaft2Spin.to_csv('fft_' + BFieldStrength + '_raft2SpinSpeeds.csv', index=False)

# testing the random distribution sampling:
# mu, sigma = 0, 0.01 # mean and standard deviation
# s = np.random.normal(mu, sigma, 10000)
# count, bins, ignored = plt.hist(s, 30, density=True)
# plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (bins - mu)**2 / (2 * sigma**2) ), linewidth=2, color='r')
# plt.show()

#%% load one specific simulated data and look at the results
dataID = 20

variableListFromSimulatedFile = list(mainDataList[dataID].keys())

for key, value in mainDataList[dataID].items():  # loop through key-value pairs of python dictionary
    globals()[key] = value

# data treatment
startOfSamplingStep = 9000  # 0, 10000
samplingRate = 1 / timeStepSize  #
raft1Locations = raftLocations[0, startOfSamplingStep:, :]  # unit: micron
raft2Locations = raftLocations[1, startOfSamplingStep:, :]  # unit: micron

vector1To2 = raft2Locations - raft1Locations  # unit: micron
distances = np.sqrt(vector1To2[:, 0] ** 2 + vector1To2[:, 1] ** 2)  # unit micron, pairwise ccDistances
distancesMean = distances.mean()
distancesSTD = distances.std()

distancesDownSampled = distances[::100]

fDistances, pDistances = fft_general(samplingRate, distances)

phase1To2 = np.arctan2(vector1To2[:, 1], vector1To2[:, 0]) * 180 / np.pi
phasesAjusted = adjust_phases(phase1To2)
orbitingSpeeds = np.gradient(phasesAjusted) * samplingRate / 180 * np.pi
orbitingSpeedsMean = orbitingSpeeds.mean()
orbitingSpeedsSTD = orbitingSpeeds.std()

fOrbitingSpeeds, pOrbitingSpeeds = fft_general(samplingRate, orbitingSpeeds)

raft1Orientations = raftOrientations[0, startOfSamplingStep:]
raft2Orientations = raftOrientations[1, startOfSamplingStep:]
raft1OrientationsAdjusted = adjust_phases(raft1Orientations)
raft2OrientationsAdjusted = adjust_phases(raft2Orientations)
raft1SpinSpeeds = np.gradient(raft1OrientationsAdjusted) * samplingRate / 360  # unit: rps
raft2SpinSpeeds = np.gradient(raft2OrientationsAdjusted) * samplingRate / 360  # unit: rps
raft1SpinSpeedsMean = raft1SpinSpeeds.mean()
raft2SpinSpeedsMean = raft2SpinSpeeds.mean()
raft1SpinSpeedsSTD = raft1SpinSpeeds.std()
raft2SpinSpeedsSTD = raft2SpinSpeeds.std()

fRaft1SpinSpeeds, pRaft1SpinSpeeds = fft_general(samplingRate, raft1SpinSpeeds)
fRaft2SpinSpeeds, pRaft2SpinSpeeds = fft_general(samplingRate, raft2SpinSpeeds)

# plotting analyzed results
# comparison of force terms
# fig, ax = plt.subplots(ncols=1, nrows=1)
# vector1To2X_Unitized = vector1To2[:, 0] / np.sqrt(vector1To2[:, 0] ** 2 + vector1To2[:, 1] ** 2)
# ax.plot(magDipoleForceOnAxisTerm[1, startOfSamplingStep:, 0] / vector1To2X_Unitized * timeStepSize, '-',
#         label='magnetic-dipole-force velocity term on raft2 / vector1To2')
# ax.plot(capillaryForceTerm[1, startOfSamplingStep:, 0] / vector1To2X_Unitized * timeStepSize, '-',
#         label='capillary-force velocity term on raft2 / vector1To2')
# ax.plot(hydrodynamicForceTerm[1, startOfSamplingStep:, 0] / vector1To2X_Unitized * timeStepSize, '-',
#         label='hydrodynamic-force velocity term on raft2 / vector1To2')
# ax.plot(wallRepulsionTerm[1, startOfSamplingStep:, 0] / vector1To2X_Unitized * timeStepSize, '-',
#         label='wall-repulsion velocity term on raft2 / vector1To2')
# ax.plot(forceCurvatureTerm[1, startOfSamplingStep:, 0] / vector1To2X_Unitized * timeStepSize, '-',
#         label='force-curvature velocity term on raft2 / vector1To2')
# ax.plot(magDipoleForceOnAxisTerm[0,startOfSamplingStep:,0]/vector1To2[:,0], '-',
#         label = 'magnetic-dipole-force velocity term on raft1 / vector1To2')
# ax.plot(capillaryForceTerm[0,startOfSamplingStep:,0]/vector1To2[:,0], '-',
#         label = 'capillary-force velocity term on raft1 / vector1To2')
# ax.plot(hydrodynamicForceTerm[0,startOfSamplingStep:,0]/vector1To2[:,0], '-',
#         label = 'hydrodynamic-force velocity term on raft1 / vector1To2')
# ax.plot(wallRepulsionTerm[0,startOfSamplingStep:,0]/vector1To2[:,0], '-',
#         label = 'wall-repulsion velocity term on raft1 / vector1To2')
# ax.plot(forceCurvatureTerm[0,startOfSamplingStep:,0]/vector1To2[:,0], '-',
#         label = 'force-curvature velocity term on raft1 / vector1To2')
# ax.set_xlabel('time step number', size=20)
# ax.set_ylabel('displacement along vector1To2 (um)', size=20)
# ax.set_title('Simulation at {}rps'.format(magneticFieldRotationRPS))
# ax.legend()
# plt.show()

# plotting distances between rafts vs frame# (time)
fig, ax = plt.subplots(ncols=1, nrows=1)
ax.plot(np.arange(len(distances)) * timeStepSize, distances, '-o', label='c')
ax.set_xlabel('Time (s)', size=20)
ax.set_ylabel('ccdistances between rafts (micron)', size=20)
ax.set_title('Simulation at {}rps'.format(magneticFieldRotationRPS))
ax.legend()
plt.show()

# plotting the fft of distances
fig, ax = plt.subplots(ncols=1, nrows=1)
ax.plot(fDistances[1:], pDistances[1:], '-o', label='c')
ax.set_xlabel('fDistances (Hz)', size=20)
ax.set_ylabel('Power P1 (a.u.)', size=20)
ax.set_title('Simulation at {}rps'.format(magneticFieldRotationRPS))
ax.legend()
plt.show()

# plotting orbiting speeds vs time
fig, ax = plt.subplots(ncols=1, nrows=1)
ax.plot(orbitingSpeeds, '-o', label='orbiting speeds calculated from orientation')
ax.set_xlabel('Frames(Time)', size=20)
ax.set_ylabel('orbiting speeds in rad/s', size=20)
ax.set_title('Simulation at {}rps'.format(magneticFieldRotationRPS))
ax.legend()
plt.show()

# plotting the fft of orbiting speed
fig, ax = plt.subplots(ncols=1, nrows=1)
ax.plot(fOrbitingSpeeds[1:], pOrbitingSpeeds[1:], '-o', label='c')
ax.set_xlabel('fOrbitingSpeeds (Hz)', size=20)
ax.set_ylabel('Power P1 (a.u.)', size=20)
ax.set_title('Simulation at {}rps'.format(magneticFieldRotationRPS))
ax.legend()
plt.show()

# comparison of torque terms
# fig, ax = plt.subplots(ncols=1, nrows=1)
# ax.plot(magneticFieldTorqueTerm[1, startOfSamplingStep:] / np.pi * 180 * timeStepSize, '-',
#         label='magnetic field torque term of raft 2')
# ax.plot(magneticDipoleTorqueTerm[1, startOfSamplingStep:] / np.pi * 180 * timeStepSize, '-',
#         label='magnetic dipole torque term of raft 2')
# ax.plot(capillaryTorqueTerm[1, startOfSamplingStep:] / np.pi * 180 * timeStepSize, '-',
#         label='capillary torque term of raft 2')
# # ax.plot(magneticFieldTorqueTerm[0,startOfSamplingStep:], '-', label = 'magnetic field torque term of raft 1')
# # ax.plot(magneticDipoleTorqueTerm[0,startOfSamplingStep:], '-', label = 'magnetic dipole torque term of raft 1')
# # ax.plot(capillaryTorqueTerm[0,startOfSamplingStep:], '-', label = 'capillary torque term of raft 1')
# ax.set_xlabel('time step number', size=20)
# ax.set_ylabel('rotation d_alpha (deg)', size=20)
# ax.set_title('Simulation at {}rps'.format(magneticFieldRotationRPS))
# ax.legend()
# plt.show()

# plotting raft relative orientation phi_ji
# fig, ax = plt.subplots(ncols=1, nrows=1)
# ax.plot(raftRelativeOrientationInDeg[0, 1, :], '-o', label='relative orientation of raft 2 and neighbor 1')
# # ax.plot(raftRelativeOrientationInDeg[1, 0, :],'-o', label = 'relative orientation of raft 1 and neighbor 2')
# ax.set_xlabel('Steps(Time)', size=20)
# ax.set_ylabel('orientation angles', size=20)
# ax.set_title('Simulation at {}rps'.format(magneticFieldRotationRPS))
# ax.legend()
# plt.show()

# the angles between the magnetic dipole moment and the magnetic field
# ref: magneticFieldTorqueTerm[raftID, currentStepNum] = tb * magneticFieldStrength * magneticMomentOfOneRaft \
#                                                        * np.sin(np.deg2rad(magneticFieldDirection -
#                                                                            raftOrientations[raft_id,currentStepNum])) \
#                                                        * 1e6 /(8*np.pi*miu*R**3)

# miu = 1e-15  # dynamic viscosity of water; unit conversion: 1e-3 Pa.s = 1e-3 N.s/m^2 = 1e-15 N.s/um^2
# R = 1.5e2  # unit: micron
#
# fig, ax = plt.subplots(ncols=1, nrows=1)
# ax.plot(np.arcsin((magneticFieldTorqueTerm[0, startOfSamplingStep:] * (8 * np.pi * miu * R ** 3)) / (
#         magneticFieldStrength * magneticMomentOfOneRaft * 1e6)) / np.pi * 180, '-',
#         label='the angle between B and m')
# ax.set_xlabel('time step number', size=20)
# ax.set_ylabel('angle (deg)', size=20)
# ax.set_title('Simulation at {}rps'.format(magneticFieldRotationRPS))
# ax.legend()
# plt.show()

# plotting raft orientations vs frame# (time)
fig, ax = plt.subplots(ncols=1, nrows=1)
ax.plot(raft1Orientations, '-o', label='raft 1 orientation before adjustment')
ax.plot(raft2Orientations, '-o', label='raft 2 orientation before adjustment')
ax.set_xlabel('Steps(Time)', size=20)
ax.set_ylabel('orientation angles', size=20)
ax.set_title('Simulation at {}rps'.format(magneticFieldRotationRPS))
ax.legend()
plt.show()

# plotting raft orientations adjusted vs frame# (time)
fig, ax = plt.subplots(ncols=1, nrows=1)
ax.plot(raft1OrientationsAdjusted, '-o', label='raft 1 orientation adjusted')
ax.plot(raft2OrientationsAdjusted, '-o', label='raft 2 orientation adjusted')
ax.set_xlabel('Steps(Time)', size=20)
ax.set_ylabel('orientation angles adjusted', size=20)
ax.set_title('Simulation at {}rps'.format(magneticFieldRotationRPS))
ax.legend()
plt.show()

# plotting raft spin speeds vs frame# (time)
fig, ax = plt.subplots(ncols=1, nrows=1)
ax.plot(raft1SpinSpeeds, '-', label='raft 1 spin speeds')
ax.plot(raft2SpinSpeeds, '-', label='raft 2 spin speeds')
ax.plot(np.deg2rad(adjust_phases(raftOrientations[0,startOfSamplingStep+1:]) -
                   adjust_phases(raftOrientations[0,startOfSamplingStep:-1]))/timeStepSize/(2*np.pi), '-')
ax.set_xlabel('Steps(Time)', size=20)
ax.set_ylabel('spin speeds (rps)', size=20)
ax.set_title('Simulation at {}rps'.format(magneticFieldRotationRPS))
ax.legend()
plt.show()

# plotting the fft of spin speeds of raft 1
fig, ax = plt.subplots(ncols=1, nrows=1)
ax.plot(fRaft1SpinSpeeds[1:], pRaft1SpinSpeeds[1:], '-o', label='c')
ax.set_xlabel('fRaft1SpinSpeeds (Hz)', size=20)
ax.set_ylabel('Power P1 (a.u.)', size=20)
ax.set_title('Simulation at {}rps'.format(magneticFieldRotationRPS))
ax.legend()
plt.show()

# plotting the fft of spin speeds of raft 2
fig, ax = plt.subplots(ncols=1, nrows=1)
ax.plot(fRaft2SpinSpeeds[1:], pRaft2SpinSpeeds[1:], '-o', label='c')
ax.set_xlabel('fRaft2SpinSpeeds (Hz)', size=20)
ax.set_ylabel('Power P1 (a.u.)', size=20)
ax.set_title('Simulation at {}rps'.format(magneticFieldRotationRPS))
ax.legend()
plt.show()

