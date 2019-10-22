"""
This module processes the data of simulated many-raft interactions.
The maximum characters per line is set to be 120.
"""
import glob
import os
import shelve
import platform

# import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# import progressbar
# from scipy.integrate import RK45
# from scipy.integrate import solve_ivp
# from scipy.spatial import Voronoi as scipyVoronoi
# import scipy.io
# from scipy.spatial import distance as scipy_distance


if platform.node() == 'NOTESIT43' and platform.system() == 'Windows':
    projectDir = "D:\\simulationFolder\\spinning_rafts_sim2"
elif platform.node() == 'NOTESIT71' and platform.system() == 'Linux':
    projectDir = r'/media/wwang/shared/spinning_rafts_simulation/spinning_rafts_sim2'
else:
    projectDir = os.getcwd()

if projectDir != os.getcwd():
    os.chdir(projectDir)

# import scripts.functions_spinning_rafts as fsr

dataDir = os.path.join(projectDir, 'data')

# %% load simulated data in one main folder
os.chdir(dataDir)

rootFolderTreeGen = os.walk(dataDir)
_, mainFolders, _ = next(rootFolderTreeGen)

mainFolderID = 0
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

# %% many-raft data treatment and output to csv

dfOrderParameters = pd.DataFrame(columns=['time(s)'])
dfEntropies = pd.DataFrame(columns=['time(s)'])
selectEveryNPoint = 10

for dataID in range(len(mainDataList)):
    numOfTimeSteps = mainDataList[dataID]['numOfTimeSteps']
    timeStepSize = mainDataList[dataID]['timeStepSize']

    dfOrderParameters['time(s)'] = np.arange(0, numOfTimeSteps, selectEveryNPoint) * timeStepSize
    dfEntropies['time(s)'] = np.arange(0, numOfTimeSteps, selectEveryNPoint) * timeStepSize

    magneticFieldRotationRPS = mainDataList[dataID]['magneticFieldRotationRPS']
    hexaticOrderParameterAvgNorms = mainDataList[dataID]['hexaticOrderParameterAvgNorms']
    hexaticOrderParameterModuliiAvgs = mainDataList[dataID]['hexaticOrderParameterModuliiAvgs']
    hexaticOrderParameterModuliiStds = mainDataList[dataID]['hexaticOrderParameterModuliiStds']
    entropyByNeighborDistances = mainDataList[dataID]['entropyByNeighborDistances']

    colName = str(-mainDataList[dataID]['magneticFieldRotationRPS']).zfill(4)

    dfOrderParameters[colName + '_avgNorm'] = hexaticOrderParameterAvgNorms[0::selectEveryNPoint]
    dfOrderParameters[colName + '_ModuliiAvg'] = hexaticOrderParameterModuliiAvgs[0::selectEveryNPoint]
    dfOrderParameters[colName + '_ModuliiStds'] = hexaticOrderParameterModuliiStds[0::selectEveryNPoint]
    dfEntropies[colName] = entropyByNeighborDistances[0::selectEveryNPoint]

dfOrderParameters.to_csv('orderParameters.csv', index=False)
dfEntropies.to_csv('entropies.csv', index=False)


# %% load one specific simulation data
dataID = 0

variableListFromSimulatedFile = list(mainDataList[dataID].keys())

numOfTimeSteps = mainDataList[dataID]['numOfTimeSteps']
timeStepSize = mainDataList[dataID]['timeStepSize']
magneticFieldRotationRPS = mainDataList[dataID]['magneticFieldRotationRPS']
hexaticOrderParameterModuliiAvgs = mainDataList[dataID]['hexaticOrderParameterModuliiAvgs']
hexaticOrderParameterModuliiStds = mainDataList[dataID]['hexaticOrderParameterModuliiStds']
hexaticOrderParameterAvgNorms = mainDataList[dataID]['hexaticOrderParameterAvgNorms']
entropyByNeighborDistances = mainDataList[dataID]['entropyByNeighborDistances']


# hexatic order parameter vs time, error bars
selectEveryNPoint = 100
_, ax = plt.subplots(ncols=1, nrows=1)
ax.errorbar(np.arange(0, numOfTimeSteps, selectEveryNPoint) * timeStepSize,
            hexaticOrderParameterModuliiAvgs[0::selectEveryNPoint],
            yerr=hexaticOrderParameterModuliiStds[0::selectEveryNPoint], label='<|phi6|>')
ax.set_xlabel('Time (s)', size=20)
ax.set_ylabel('order parameter', size=20)
ax.set_title('Simulation at {}rps'.format(magneticFieldRotationRPS))
ax.legend()
plt.show()

# hexatic order parameter vs time
selectEveryNPoint = 100
_, ax = plt.subplots(ncols=1, nrows=1)
ax.plot(np.arange(0, numOfTimeSteps, selectEveryNPoint) * timeStepSize,
        hexaticOrderParameterModuliiAvgs[0::selectEveryNPoint], label='<|phi6|>')
ax.plot(np.arange(0, numOfTimeSteps, selectEveryNPoint) * timeStepSize,
        hexaticOrderParameterAvgNorms[0::selectEveryNPoint], label='|<phi6>|')
ax.set_xlabel('Time (s)', size=20)
ax.set_ylabel('order parameter', size=20)
ax.set_title('Simulation at {}rps'.format(magneticFieldRotationRPS))
ax.legend()
plt.show()

# entropy vs time
selectEveryNPoint = 100
_, ax = plt.subplots(ncols=1, nrows=1)
ax.plot(np.arange(0, numOfTimeSteps, selectEveryNPoint) * timeStepSize,
        entropyByNeighborDistances[0::selectEveryNPoint], label='entropy by distances')
ax.set_xlabel('Time (s)', size=20)
ax.set_ylabel('entropy by neighbor distances', size=20)
ax.set_title('Simulation at {}rps'.format(magneticFieldRotationRPS))
ax.legend()
plt.show()

