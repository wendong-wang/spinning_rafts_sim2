#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 14:52:17 2020

@author: vimal
"""

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
    
#%%
os.chdir(dataDir)
resultFoldersFull = next(os.walk(dataDir))[1]
resultFoldersFull.sort()
resultFolders = resultFoldersFull[266:-2].copy()

#%%
arenaSize = 1.5e4  # unit: micron
R = raftRadius = 1.5e2  # unit: micron
numFrames_last = 10000
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

summaryDataFrameColNames = ['date', 'time', 'spinspeed', 'Numofsteps', 'lastframe', 'Max|psi6|FrameNum', 'Max_|psi6|', 'Last_|psi6|', 'Last_psi6_avg', 
                            'Min_KLD_X', 'Min_KLD_Y', 'Min_KLD_NDist', 'Min_KLD_ODist', 'NumAcceptedProb', 'XYorODist', '|psi6|_avg', '|psi6|_std', ' psi6_avg', 'psi6_std']

dfSummary = pd.DataFrame(columns = summaryDataFrameColNames)

Mod_psi6 = np.zeros((len(resultFolders), numFrames_last)) # len(resultFolders) = number of simulations for each spinspeed or target distribution # num of frames to analyze = 10000. change this if needded.
psi6_avg = np.zeros((len(resultFolders), numFrames_last)) # this is the average norm and the one above i the norm average.

for index, resultFolderID in progressbar.progressbar(enumerate(range(len(resultFolders)))):
    
    parts = resultFolders[resultFolderID].split('_')
    os.chdir(resultFolders[resultFolderID])
    
    numOfRafts = int(parts[2][:-5])
    spinSpeed = int(parts[4][:2])
    numOfSteps = int(parts[3][10:])
    lastFrame = int(parts[7][9:])
    date = parts[0]
    time = parts[1]
    XYorODist = parts[8]
    
    shelfToRead = shelve.open('simulation_{}Rafts_{}rps'.format(numOfRafts, spinSpeed), flag='r')
    listOfVariablesInShelfToRead = list(shelfToRead.keys())
    for key in shelfToRead:
        globals()[key] = shelfToRead[key]
    shelfToRead.close()
        
    #store distributions in csv file
    count_NDist_all = np.zeros((len(binEdgesNeighborDistances[:-1]), 4)) # bin edges, frame with highest psi6, last frame, experimental distribution
    count_X_all = np.zeros((len(binEdgesX[:-1]), 4)) # bin edges, frame with highest psi6, last frame, experimental distribution
    count_Y_all = np.zeros((len(binEdgesY[:-1]), 4)) # bin edges, frame with highest psi6, last frame, experimental distribution
    count_ODist_all = np.zeros((len(binEdgesOrbitingDistances[:-1]), 4)) # bin edges, frame with highest psi6, last frame, experimental distribution
    
    Frame_max = hexaticOrderParameterModuliiAvgs.argmax()
    Mod_psi6_max = hexaticOrderParameterModuliiAvgs.max()
    Mod_psi6_last = hexaticOrderParameterModuliiAvgs[-2]
    psi6_avg_last = hexaticOrderParameterAvgNorms[-2]
    
    Mod_psi6[index,:] = hexaticOrderParameterModuliiAvgs[-numFrames_last:]
    psi6_avg[index,:] = hexaticOrderParameterAvgNorms[-numFrames_last:]
    
    count_NDist_all[:,0] = binEdgesNeighborDistances[:-1]
    count_NDist_all[:,1] = count_NDist[:,Frame_max]
    count_NDist_all[:,2] = count_NDist[:,-2]
    count_NDist_all[:,3] = target['count_NDist']
    
    count_X_all[:,0] = binEdgesX[:-1]
    count_X_all[:,1] = count_X[:,Frame_max]
    count_X_all[:,2] = count_X[:,-2]
    count_X_all[:,3] = target['count_X']
    
    count_Y_all[:,0] = binEdgesY[:-1]
    count_Y_all[:,1] = count_Y[:,Frame_max]
    count_Y_all[:,2] = count_Y[:,-2]
    count_Y_all[:,3] = target['count_Y']
    
    count_ODist_all[:,0] = binEdgesOrbitingDistances[:-1]
    count_ODist_all[:,1] = count_ODist[:,Frame_max]
    count_ODist_all[:,2] = count_ODist[:,-2]
    count_ODist_all[:,3] = target['count_ODist']
    
    count_accepted, _ = np.histogram(np.asarray(acceptedBasedOnProbs), bins=25)
    
    dfSummary.loc[resultFolderID, 'date'] = date
    dfSummary.loc[resultFolderID, 'time'] = time
    dfSummary.loc[resultFolderID, 'spinspeed'] = spinSpeed
    dfSummary.loc[resultFolderID, 'Numofsteps'] = numOfSteps
    dfSummary.loc[resultFolderID, 'lastframe'] = lastFrame
    dfSummary.loc[resultFolderID, 'Max|psi6|FrameNum'] = Frame_max
    dfSummary.loc[resultFolderID, 'Max_|psi6|'] = Mod_psi6_max
    dfSummary.loc[resultFolderID, 'Last_|psi6|'] = Mod_psi6_last
    dfSummary.loc[resultFolderID, 'Last_psi6_avg'] = psi6_avg_last
    dfSummary.loc[resultFolderID, 'Min_KLD_X'] = klDiv_X.min()
    dfSummary.loc[resultFolderID, 'Min_KLD_Y'] = klDiv_Y.min()
    dfSummary.loc[resultFolderID, 'Min_KLD_NDist'] = klDiv_NDist.min()
    dfSummary.loc[resultFolderID, 'Min_KLD_ODist'] = klDiv_ODist.min()
    dfSummary.loc[resultFolderID, 'NumAcceptedProb'] = count_accepted.sum()
    dfSummary.loc[resultFolderID, 'XYorODist'] = XYorODist

    np.savetxt("NDist_hist_" + str(numOfRafts) + "Rafts_" + str(spinSpeed) + "rps_" + "LastFrame" + str(ifLastFrameCount) + "_totalAcceptedProbability" + str(count_accepted.sum()), count_NDist_all, delimiter=",")
    np.savetxt("X_hist_" + str(numOfRafts) + "Rafts_" + str(spinSpeed) + "rps_" + "LastFrame" + str(ifLastFrameCount) + "_totalAcceptedProbability" + str(count_accepted.sum()), count_X_all, delimiter=",")
    np.savetxt("Y_hist_" + str(numOfRafts) + "Rafts_" + str(spinSpeed) + "rps_" + "LastFrame" + str(ifLastFrameCount) + "_totalAcceptedProbability" + str(count_accepted.sum()), count_Y_all, delimiter=",")
    np.savetxt("ODist_hist_" + str(numOfRafts) + "Rafts_" + str(spinSpeed) + "rps_" + "LastFrame" + str(ifLastFrameCount) + "_totalAcceptedProbability" + str(count_accepted.sum()), count_ODist_all, delimiter=",")
    np.savetxt("Mod_psi6" + str(numOfRafts) + "Rafts_" + str(spinSpeed) + "rps_" + "LastFrame" + str(ifLastFrameCount) + 
               "_totalAcceptedProbability" + str(count_accepted.sum()), Mod_psi6, delimiter=",")
    np.savetxt("psi6_avg" + str(numOfRafts) + "Rafts_" + str(spinSpeed) + "rps_" + "LastFrame" + str(ifLastFrameCount) + 
               "_totalAcceptedProbability" + str(count_accepted.sum()), psi6_avg, delimiter=",")
 
    
    os.chdir(dataDir)


Mod_psi6_avg = Mod_psi6.mean()
Mod_psi6_std = Mod_psi6.std()
psi6_avg_avg = psi6_avg.mean()
psi6_avg_std =psi6_avg.std()


dfSummary.loc[resultFolderID, '|psi6|_avg'] = Mod_psi6_avg
dfSummary.loc[resultFolderID, '|psi6|_std'] = Mod_psi6_std
dfSummary.loc[resultFolderID, ' psi6_avg'] = psi6_avg_avg
dfSummary.loc[resultFolderID, 'psi6_std'] = psi6_avg_std


dfSummaryConverted = dfSummary.infer_objects()
dfSummarySorted = dfSummaryConverted.sort_values(by = ['spinspeed', 'date'], ascending = [True, False])
dfSummarySorted.to_csv('MonteCarlo' + '_summary.csv', index = False, columns = summaryDataFrameColNames )

    
    
    
    
    
    
    