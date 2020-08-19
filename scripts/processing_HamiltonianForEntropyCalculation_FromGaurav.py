#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 19:01:27 2020

@author: gardi
"""

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from scipy.io import loadmat
#import scipy.stats
from scipy.stats import entropy
#from scipy.signal import find_peaks
"""
This module processes the data of simulated pairwise interactions.
The maximum characters per line is set to be 120.
"""

import glob
import os
import shelve
import platform
import progressbar

# import cv2 as cv
#import matplotlib.pyplot as plt
#import numpy as np
import pandas as pd
# import progressbar
# from scipy.integrate import RK45
# from scipy.integrate import solve_ivp
# from scipy.spatial import Voronoi as scipyVoronoi
# import scipy.io
# from scipy.spatial import distance as scipy_distance

def ShannonEntropy(c):
    """calculate the Shannon entropy of 1 d data. The unit is bits """
    
    c_normalized = c / float(np.sum(c))
    c_normalized_nonzero = c_normalized[np.nonzero(c_normalized)] # gives 1D array
    H = -sum(c_normalized_nonzero* np.log2(c_normalized_nonzero))  # unit in bits
    return H

def HistogramP(L, magneticFieldRotationRPS, Beta, E_total, HydrodynamicsLift_Energy) :
    ############## Potential ebergy due to boundary #########
    
#    print(L, magneticFieldRotationRPS, Beta)
#    print(L/1.5e-4, Beta/1e12)
    cuttOff_distance = 1000
#    miu = 1e-3 # dynamic viscosity of water units Pa.s = 1e-3 N.s/m^2
    densityOfWater = 1e3  # units : 1000 kg/m^3 = 1e-15 kg/um^3
    Hydrodynamics_EEDistances = np.arange(0, 10001) / 1e6  # unit: m
    radiusOfRaft = 1.5e-4  # unit: m
    
#    L =5*radiusOfRaft # length of the effective boundary
    #L = L_optm # length of the effective boundary
    #L = 1.5e-2 # unit micron length of the actual boundary
    #R_COM = np.arange(0,L/2 - 2*radiusOfRaft) / 1e6 # unit m radial distance of centre of mass of the pair of rafts
    
    Hydrodynamics_CCDistances = Hydrodynamics_EEDistances + radiusOfRaft * 2  # unit: m
#    HydrodynamicsLift_Energy = np.zeros((len(Hydrodynamics_EEDistances), numOfBatches))  # unit: J
    Boundary_Energy = np.zeros((len(Hydrodynamics_EEDistances)))  # unit: J
    #Boundary_Energy = np.zeros((len(Hydrodynamics_EEDistances), numOfBatches, len(R_COM)))  # unit: J
    
    for index, d in enumerate(Hydrodynamics_CCDistances):
#        HydrodynamicsLift_Energy[index, dataID] = densityOfWater*(radiusOfRaft**7)*((magneticFieldRotationRPS*2*np.pi)**2)/(2*(d)**2)
        Boundary_Energy[index] = densityOfWater*(radiusOfRaft**7)*((magneticFieldRotationRPS*2*np.pi)**2)*(1/((L + d)**2) + (d - 2*radiusOfRaft)/(L**3))
    #        Boundary_Energy[index, dataID] = 1e-8*d**2
    
    #######calculate probability and make histogram of the probability distribution
#    Beta = 1e14
    #Beta = Beta_optm
    binEdgesEEDistances = np.arange(2,8.5,0.5).tolist() + [100]
    
#    H = np.zeros((cuttOff_distance))
#    P = np.zeros((cuttOff_distance))

    H = E_total + HydrodynamicsLift_Energy + Boundary_Energy[:cuttOff_distance]
    P = np.exp(-Beta*H)
    
    Hist_P = np.zeros((len(binEdgesEEDistances[:-1])))
    
    for i in range(Hist_P.shape[0]) :
        #Hist_P[i,dataID] = P[i*75:(i+1)*75,dataID].mean()
        Hist_P[i] = P[i*75:(i+1)*75].sum()/P[:].sum()
        #Hist_P[i,dataID] = P[i*75:(i+1)*75,dataID].sum()
        #Hist_P[:,dataID] = Hist_P[:,dataID]/Hist_P[:,dataID].sum()
#    print(np.argmax(Hist_P))
    return Hist_P, binEdgesEEDistances

def optimiseL(magneticFieldRotationRPS, Beta, Hist_exp, E_total, HydrodynamicsLift_Energy) :
    L_opt = 0
    for L_1 in np.arange(0.5,10,0.2)*1.5e-4 :
        Hist_P, _ = HistogramP(L_1, magneticFieldRotationRPS, Beta, E_total, HydrodynamicsLift_Energy)
        if np.argmax(Hist_P) == np.argmax(Hist_exp) :   
           L_opt = L_1
           break
#        print("simulated peak")
#        print(np.argmax(Hist_P))
#        print("experimental peak")
#        print(np.argmax(Hist_exp)) 
#    print(L_opt/1.5e-4)
    return L_opt

def optimiseBeta(magneticFieldRotationRPS, L_in, Hist_exp, E_total, HydrodynamicsLift_Energy) :
    KLD_min = 100
    Beta_opt = 1e18
    Hist_P_opt = HistogramP(L_in, magneticFieldRotationRPS, 1e12, E_total, HydrodynamicsLift_Energy)
    for index, Beta in enumerate(np.arange(1,1000,5)*1e12) :
#        print(Beta/1e12)
        Hist_P, binEdgesEEDistances_opt = HistogramP(L_in, magneticFieldRotationRPS, Beta, E_total, HydrodynamicsLift_Energy)
#        KLD = entropy(Hist_exp, qk = Hist_P)
        KLD = entropy(Hist_P, qk = Hist_exp + 1e-10)
        print(KLD)
        if KLD < KLD_min :
            KLD_min = KLD
            Beta_opt = Beta
            Hist_P_opt = Hist_P
            
    print(KLD)
    return Beta_opt, KLD_min, Hist_P_opt, binEdgesEEDistances_opt


def optimiseLBeta(magneticFieldRotationRPS, Hist_exp, E_total, HydrodynamicsLift_Energy) :
    KLD_min = 100
    Beta_opt = 1e18
    L_opt = 0.5*1.5e-4
    Hist_P_opt = HistogramP(L_opt, magneticFieldRotationRPS, 1e12, E_total, HydrodynamicsLift_Energy)
    for index, Beta in enumerate(np.arange(1,1000,5)*1e12) :
        for L_1 in np.arange(0.5,10,0.2)*1.5e-4 :
            Hist_P, binEdgesEEDistances_opt = HistogramP(L_1, magneticFieldRotationRPS, Beta, E_total, HydrodynamicsLift_Energy)
#            KLD = entropy(Hist_exp, qk = Hist_P)
            KLD = entropy(Hist_P, qk = Hist_exp + 1e-10)
            print(KLD)
            if KLD < KLD_min :
                KLD_min = KLD
                Beta_opt = Beta
                Hist_P_opt = Hist_P
                L_opt = L_1
    
    return Beta_opt, L_opt, KLD_min, Hist_P_opt, binEdgesEEDistances_opt


def optimiseBetaEntropy(magneticFieldRotationRPS, L_in, Hist_exp, E_total, HydrodynamicsLift_Energy) :
    DeltaEntropy_min = 4
    Beta_opt = 1e18
    Hist_P_opt = HistogramP(L_in, magneticFieldRotationRPS, 1e12, E_total, HydrodynamicsLift_Energy)
    for index, Beta in enumerate(np.arange(1,1000,5)*1e12) :
        print(Beta/1e12)
        Hist_P, binEdgesEEDistances_opt = HistogramP(L_in, magneticFieldRotationRPS, Beta, E_total, HydrodynamicsLift_Energy)
        Entropy_beta = ShannonEntropy(Hist_P)
        Entropy_Exp = ShannonEntropy(Hist_exp)
        DeltaEntropy = abs(Entropy_beta - Entropy_Exp)
        print(DeltaEntropy)
        if DeltaEntropy < DeltaEntropy_min :
            DeltaEntropy_min = DeltaEntropy
            Beta_opt = Beta
            Hist_P_opt = Hist_P
    print(DeltaEntropy_min)
    return Beta_opt, DeltaEntropy_min, Hist_P_opt, binEdgesEEDistances_opt


def optimiseLBetaEntropy(magneticFieldRotationRPS, Hist_exp, E_total, HydrodynamicsLift_Energy) :
    DeltaEntropy_min = 4
    Beta_opt = 1e18
    L_opt = 0.5*1.5e-4
    Hist_P_opt = HistogramP(L_opt, magneticFieldRotationRPS, 1e12, E_total, HydrodynamicsLift_Energy)
    for index, Beta in enumerate(np.arange(1,1000,5)*1e12) :
        for L_1 in np.arange(0.5,10,0.2)*1.5e-4 :
            Hist_P, binEdgesEEDistances_opt = HistogramP(L_1, magneticFieldRotationRPS, Beta, E_total, HydrodynamicsLift_Energy)
            if np.argmax(Hist_P) == np.argmax(Hist_exp) :   
                L_opt = L_1
                Entropy_beta = ShannonEntropy(Hist_P)
                Entropy_Exp = ShannonEntropy(Hist_exp)
                DeltaEntropy = abs(Entropy_beta - Entropy_Exp)
                
                if DeltaEntropy < DeltaEntropy_min :
                    DeltaEntropy_min = DeltaEntropy
                    Beta_opt = Beta
                    Hist_P_opt = Hist_P
                    L_opt = L_1
    print(DeltaEntropy)
    return Beta_opt, L_opt, DeltaEntropy_min, Hist_P_opt, binEdgesEEDistances_opt



#%% changing of directory related to simulations
# simulationFolderName = '/home/gardi/spinning_rafts_sim2'
# os.chdir(simulationFolderName)

if platform.node() == 'NOTESIT43' and platform.system() == 'Windows':
    projectDir = "D:\\simulationFolder\\spinning_rafts_sim2"
elif platform.node() == 'NOTESIT71' and platform.system() == 'Linux':
    projectDir = r'/media/wwang/shared/spinning_rafts_simulation/spinning_rafts_sim2'
else:
    projectDir = os.getcwd()

if projectDir != os.getcwd():
    os.chdir(projectDir)

import scripts.functions_spinning_rafts as fsr

scriptDir = os.path.join(projectDir, "scripts")
dataDir = os.path.join(projectDir, 'data')
capSym6Dir = os.path.join(projectDir, '2019-05-13_capillaryForceCalculations-sym6')

#%%  change to directory contatining experiments data

#rootFolderNameFromWindows = '/media/gardi/DataMPI_10/Data_PhantomMiroLab140'
#rootFolderNameFromWindows = '/media/gardi/VideoFiles_Raw_PP/Data_Camera_Basler-acA2500-60uc/'
# rootFolderNameFromWindows = '/home/gardi/Rafts/Experiments Data/Data_Camera_Basler-acA2500-60uc/'
# os.chdir(rootFolderNameFromWindows)
# rootFolderTreeGen = os.walk(rootFolderNameFromWindows)
# _, mainFolders_experiments, _ = next(rootFolderTreeGen)

os.chdir(dataDir)
rootFolderTreeGen = os.walk(dataDir)
_, mainFolders_experiments, _ = next(rootFolderTreeGen)

#%% loading all the data in a specific main folder into mainDataList
# at the moment, it handles one main folder at a time. 

#for mainFolderID in np.arange(0,1):
#    os.chdir(mainFolders[mainFolderID])

mainFolderID_experiments = 0
os.chdir(mainFolders_experiments[mainFolderID_experiments])
expDataDir = os.getcwd()
dataFileList_experiments = glob.glob('*.dat')
dataFileList_experiments.sort()
dataFileListExcludingPostProcessed_experiments = dataFileList_experiments.copy()
numberOfPostprocessedFiles_experiments = 0

mainDataList_experiments = []
variableListsForAllMainData_experiments = []

for dataID in range(len(dataFileList_experiments)):
#for dataID in range(1, 61):
    dataFileToLoad_experiments = dataFileList_experiments[dataID].partition('.dat')[0]
    
    if 'postprocessed' in dataFileToLoad_experiments:
        # the list length changes as items are deleted
        del dataFileListExcludingPostProcessed_experiments[dataID - numberOfPostprocessedFiles_experiments] 
        numberOfPostprocessedFiles_experiments = numberOfPostprocessedFiles_experiments + 1
        continue
    
    tempShelf = shelve.open(dataFileToLoad_experiments)
    variableListOfOneMainDataFile_experiments = list(tempShelf.keys())
    
    expDict = {}
    for key in tempShelf:
        try:
            expDict[key] = tempShelf[key]
        except TypeError:
            pass
    
    tempShelf.close()
    mainDataList_experiments.append(expDict)
    variableListsForAllMainData_experiments.append(variableListsForAllMainData_experiments)
    
    
#    # go one level up to the root folder
#    os.chdir('..')

#%% capillary energy landscape extraction from .mat file

os.chdir(capSym6Dir)

#x = loadmat('ResultsCombined_L4_amp2_arcAngle30_ccDist301-350um-count50_rotAngle61_bathRad500.mat')
#capillary_data_341_to_1300 = loadmat('Results_sym6_arcAngle30_ccDistance341to1300step1um_angleCount361_errorPower-10_treated.mat')
capillary_data_341_to_1600 = loadmat('Results_sym6_arcAngle30_ccDistance341to1600step1um_angleCount361_errorPower-10_treated.mat')
capillary_data_301_to_350 = loadmat('ResultsCombined_L4_amp2_arcAngle30_ccDist301-350um-count50_rotAngle61_bathRad500.mat')

#fig = plt.figure()
#ax = fig.gca(projection='3d')

# Make data.
EEDistances_capillary = np.arange(1, 1001, 1) # E-E distamce unit um
orientationAngles_capillary = np.arange(0, 360, 1) # relative orientationm angle in degrees
#E_capillary_341_to_1300 = capillary_data_341_to_1300['energyScaledToRealSurfaceTensionRezeroed']/1e15 # unit: J
E_capillary_341_to_1600 = capillary_data_341_to_1600['energyScaledToRealSurfaceTensionRezeroed']/1e15 # unit: J
E_capillary_301_to_350 = (capillary_data_301_to_350['netEnergy_2Rafts'].transpose() - capillary_data_301_to_350['netEnergy_2Rafts'].transpose()[49,0] )/1e15  # unit: J
#E_capillary_combined = np.vstack((E_capillary_301_to_350[:40,:], E_capillary_341_to_1300[:,:61]))
E_capillary_combined = np.vstack((E_capillary_301_to_350[:40,:], E_capillary_341_to_1600[:,:61]))
E_capillary_combined_All360 = np.hstack((E_capillary_combined[:,:60],E_capillary_combined[:,:60],E_capillary_combined[:,:60],E_capillary_combined[:,:60],E_capillary_combined[:,:60],E_capillary_combined[:,:60]))


EEDistances_capillary, orientationAngles_capillary = np.meshgrid(EEDistances_capillary, orientationAngles_capillary)
EEDistances_capillary = np.transpose(EEDistances_capillary)
orientationAngles_capillary = np.transpose(orientationAngles_capillary)

cuttOff_distance_capillary = 100 # unit micron

## Plot the surface.
#surf = ax.plot_surface(EEDistances_capillary[:cuttOff_distance_capillary,:], orientationAngles_capillary[:cuttOff_distance_capillary,:], E_capillary_combined_All360[:cuttOff_distance_capillary,:], 
#                       cmap=cm.coolwarm, linewidth=0, antialiased=False)
#
## Customize the z axis.
#
#ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
#
## Add a color bar which maps values to colors.
#fig.colorbar(surf, shrink=0.5, aspect=5)
#
#plt.show()


shelveName = 'capillaryForceAndTorque_sym6'
shelveDataFileName = shelveName + '.dat'
listOfVariablesToLoad = ['eeDistanceCombined', 'forceCombinedDistancesAsRowsAll360',
                         'torqueCombinedDistancesAsRowsAll360']

if not os.path.isfile(shelveDataFileName):
    print('the capillary data file is missing')

tempShelf = shelve.open(shelveName)
capillaryEEDistances = tempShelf['eeDistanceCombined']  # unit: m
#capillaryForcesDistancesAsRowsLoaded = tempShelf['forceCombinedDistancesAsRowsAll360']  # unit: N
capillaryTorquesDistancesAsRowsLoaded = tempShelf['torqueCombinedDistancesAsRowsAll360']  # unit: N.m

# further data treatment on capillary force profile
# insert the force and torque at eeDistance = 1um as the value for eedistance = 0um.
capillaryEEDistances = np.insert(capillaryEEDistances, 0, 0)
#capillaryForcesDistancesAsRows = np.concatenate(
#    (capillaryForcesDistancesAsRowsLoaded[:1, :], capillaryForcesDistancesAsRowsLoaded), axis=0)
capillaryTorquesDistancesAsRows = np.concatenate(
    (capillaryTorquesDistancesAsRowsLoaded[:1, :], capillaryTorquesDistancesAsRowsLoaded), axis=0)

# add angle=360, the same as angle = 0
#capillaryForcesDistancesAsRows = np.concatenate(
#    (capillaryForcesDistancesAsRows, capillaryForcesDistancesAsRows[:, 0].reshape(1001, 1)), axis=1)
capillaryTorquesDistancesAsRows = np.concatenate(
    (capillaryTorquesDistancesAsRows, capillaryTorquesDistancesAsRows[:, 0].reshape(1001, 1)), axis=1)

# correct for the negative sign of the torque
capillaryTorquesDistancesAsRows = - capillaryTorquesDistancesAsRows

## some extra treatment for the force matrix
## note the sharp transition at the peak-peak position (45 deg): only 1 deg difference,
## the force changes from attraction to repulsion. consider replacing values at eeDistance = 0, 1, 2,
## with values at eeDistance = 5um.
#nearEdgeSmoothingThres = 1  # unit: micron; if 1, then it is equivalent to no smoothing.
#for distanceToEdge in np.arange(nearEdgeSmoothingThres):
#    capillaryForcesDistancesAsRows[distanceToEdge, :] = capillaryForcesDistancesAsRows[nearEdgeSmoothingThres, :]
#    capillaryTorquesDistancesAsRows[distanceToEdge, :] = capillaryTorquesDistancesAsRows[nearEdgeSmoothingThres, :]
#
## select a cut-off distance below which all the attractive force (negative-valued) becomes zero,
## due to raft wall-wall repulsion
#capAttractionZeroCutoff = 0
#mask = np.concatenate((capillaryForcesDistancesAsRows[:capAttractionZeroCutoff, :] < 0,
#                       np.zeros((capillaryForcesDistancesAsRows.shape[0] - capAttractionZeroCutoff,
#                                 capillaryForcesDistancesAsRows.shape[1]), dtype=int)),
#                      axis=0)
#capillaryForcesDistancesAsRows[mask.nonzero()] = 0
#
## set capillary force = 0 at 0 distance
## capillaryForcesDistancesAsRows[0,:] = 0
#
## realign the first peak-peak direction with an angle = capillaryPeakOffset from the x-axis.
#capillaryPeakOffset = 0
#capillaryForcesDistancesAsRows = np.roll(capillaryForcesDistancesAsRows, capillaryPeakOffset,
#                                         axis=1)  # 45 is due to original data
#capillaryTorquesDistancesAsRows = np.roll(capillaryTorquesDistancesAsRows, capillaryPeakOffset, axis=1)
#
#capillaryForceAngleAveraged = capillaryForcesDistancesAsRows[1:, :-1].mean(axis=1)  # starting from 1 um to 1000 um
#capillaryForceMaxRepulsion = capillaryForcesDistancesAsRows[1:, :-1].max(axis=1)
#capillaryForceMaxRepulsionIndex = capillaryForcesDistancesAsRows[1:, :-1].argmax(axis=1)
#capillaryForceMaxAttraction = capillaryForcesDistancesAsRows[1:, :-1].min(axis=1)
#capillaryForceMaxAttractionIndex = capillaryForcesDistancesAsRows[1:, :-1].argmin(axis=1)

# %% magnetic force and torque calculation:
miu0 = 4 * np.pi * 1e-7  # unit: N/A**2, or T.m/A, or H/m

# from the data 2018-09-28, 1st increase:
# (1.4e-8 A.m**2 for 14mT), (1.2e-8 A.m**2 for 10mT), (0.96e-8 A.m**2 for 5mT), (0.78e-8 A.m**2 for 1mT)
# from the data 2018-09-28, 2nd increase:
# (1.7e-8 A.m**2 for 14mT), (1.5e-8 A.m**2 for 10mT), (1.2e-8 A.m**2 for 5mT), (0.97e-8 A.m**2 for 1mT)
magneticMomentOfOneRaft = 1e-8  # unit: A.m**2

orientationAngles = np.arange(0, 361)  # unit: degree;
orientationAnglesInRad = np.radians(orientationAngles)

magneticDipoleEEDistances = np.arange(0, 10001) / 1e6  # unit: m

radiusOfRaft = 1.5e-4  # unit: m

magneticDipoleCCDistances = magneticDipoleEEDistances + radiusOfRaft * 2  # unit: m

magDpEnergy = np.zeros((len(magneticDipoleEEDistances), len(orientationAngles)))  # unit: J
magDpForceOnAxis = np.zeros((len(magneticDipoleEEDistances), len(orientationAngles)))  # unit: N
magDpForceOffAxis = np.zeros((len(magneticDipoleEEDistances), len(orientationAngles)))  # unit: N
magDpTorque = np.zeros((len(magneticDipoleEEDistances), len(orientationAngles)))  # unit: N.m

for index, d in enumerate(magneticDipoleCCDistances):
    magDpEnergy[index, :] = \
        miu0 * magneticMomentOfOneRaft ** 2 * (1 - 3 * (np.cos(orientationAnglesInRad) ** 2)) / (4 * np.pi * d ** 3)
#    magDpEnergy[index, :] = \
#        miu0 * magneticMomentOfOneRaft ** 2 * (3 * (np.cos(orientationAnglesInRad) ** 2) - 1) / (4 * np.pi * d ** 3)
#    
    magDpForceOnAxis[index, :] = \
        3 * miu0 * magneticMomentOfOneRaft ** 2 * (1 - 3 * (np.cos(orientationAnglesInRad) ** 2)) / (4 * np.pi * d ** 4)
    magDpForceOffAxis[index, :] = \
        3 * miu0 * magneticMomentOfOneRaft ** 2 * (2 * np.cos(orientationAnglesInRad) *
                                                   np.sin(orientationAnglesInRad)) / (4 * np.pi * d ** 4)
    magDpTorque[index, :] = \
        miu0 * magneticMomentOfOneRaft ** 2 * (3 * np.cos(orientationAnglesInRad) *
                                               np.sin(orientationAnglesInRad)) / (4 * np.pi * d ** 3)


#fig = plt.figure()
#ax = fig.gca(projection='3d')

# Make data.
EEDistances_magDipole = np.arange(0, 10001, 1) # E-E distamce unit um
orientationAngles_magDipole = np.arange(0, 361, 1) # relative orientationm angle in degrees
E_magDipole = magDpEnergy

EEDistances_magDipole, orientationAngles_magDipole = np.meshgrid(EEDistances_magDipole, orientationAngles_magDipole)
EEDistances_magDipole = np.transpose(EEDistances_magDipole)
orientationAngles_magDipole = np.transpose(orientationAngles_magDipole)

cuttOff_distance_magDipole = 100 # unit um

## Plot the surface.
#surf = ax.plot_surface(EEDistances_magDipole[:cuttOff_distance_magDipole,:], orientationAngles_magDipole[:cuttOff_distance_magDipole,:], E_magDipole[:cuttOff_distance_magDipole,:], 
#                       cmap=cm.coolwarm, linewidth=0, antialiased=False)
#
## Customize the z axis.
#
#ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
#
## Add a color bar which maps values to colors.
#fig.colorbar(surf, shrink=0.5, aspect=5)
#
#plt.show()

# magnetic force at 1um(attractionZeroCutoff) should have no attraction, due to wall-wall repulsion.
# Treat it similarly as capillary cutoff
# attractionZeroCutoff = 0  # unit: micron
# mask = np.concatenate((magDpForceOnAxis[:attractionZeroCutoff, :] < 0,
#                        np.zeros((magDpForceOnAxis.shape[0] - attractionZeroCutoff, magDpForceOnAxis.shape[1]),
#                                 dtype=int)), axis=0)
# magDpForceOnAxis[mask.nonzero()] = 0
#
# magDpMaxRepulsion = magDpForceOnAxis.max(axis=1)
# magDpForceAngleAverage = magDpForceOnAxis[:, :-1].mean(axis=1)

# set on-axis magnetic force = 0 at 0 distance
# magDpForceOnAxis[0,:] = 0


#%% HYdrodynamic life force energy and potential ebergy due to boundary
miu = 1e-3 # dynamic viscosity of water units Pa.s = 1e-3 N.s/m^2
densityOfWater = 1e3  # units : 1000 kg/m^3 = 1e-15 kg/um^3
Hydrodynamics_EEDistances = np.arange(0, 10001) / 1e6  # unit: m
radiusOfRaft = 1.5e-4  # unit: m

#L = 5*radiusOfRaft # length of the effective boundary
#L = 1.5e-2 # unit micron length of the actual boundary
#R_COM = np.arange(0,L/2 - 2*radiusOfRaft) / 1e6 # unit m radial distance of centre of mass of the pair of rafts
numOfBatches = 59

Hydrodynamics_CCDistances = Hydrodynamics_EEDistances + radiusOfRaft * 2  # unit: m
HydrodynamicsLift_Energy = np.zeros((len(Hydrodynamics_EEDistances), numOfBatches))  # unit: J
Boundary_Energy = np.zeros((len(Hydrodynamics_EEDistances), numOfBatches))  # unit: J
#Boundary_Energy = np.zeros((len(Hydrodynamics_EEDistances), numOfBatches, len(R_COM)))  # unit: J

for dataID in range(0, numOfBatches) :
#    magneticFieldRotationRPS = magneticFieldRotationRPS_all[dataID]
    magneticFieldRotationRPS = dataID + 11
    for index, d in enumerate(Hydrodynamics_CCDistances):
        HydrodynamicsLift_Energy[index, dataID] = densityOfWater*(radiusOfRaft**7)*((magneticFieldRotationRPS*2*np.pi)**2)/(2*(d)**2)
#        Boundary_Energy[index, dataID] = densityOfWater*(radiusOfRaft**7)*((magneticFieldRotationRPS*2*np.pi)**2)*(1/((L + d)**2) + (d - 2*radiusOfRaft)/(L**3))
#        Boundary_Energy[index, dataID] = 1e-8*d**2

#%% angle averaged energies
        
cuttOff_distance = 1300 #um
cuttOff_time = 2500 # frame number

miu = 1e-3 # dynamic viscosity of water units Pa.s = 1e-3 N.s/m^2
densityOfWater = 1e3  # units : 1000 kg/m^3 = 1e-15 kg/um^3
Beta = 1e14

E_total = np.zeros((cuttOff_distance, 1))
H = np.zeros((cuttOff_distance , numOfBatches))
P = np.zeros((cuttOff_distance , numOfBatches))

E_cap_AA = E_capillary_combined_All360.mean(axis=1)
E_magdp_AA = E_magDipole[:,0:360].mean(axis=1)


E_total = E_cap_AA + E_magdp_AA[0:cuttOff_distance]
#E_total[:E_capillary_combined_All360.shape[0],-1] = E_magDipole[300:E_capillary_combined_All360.shape[0]+300,-1]
#E_total[E_capillary_combined_All360.shape[0]:,:] = E_magDipole[E_capillary_combined_All360.shape[0]+300:,:]

#E_total_AA = E_total.mean(axis=1)

#%% load data corresponding to a specific experiment (subfolder or video) into variables
for dataID_experiments in progressbar.progressrange(range(0,len(mainDataList_experiments))) :
    date_experiments = mainDataList_experiments[dataID_experiments]['date']
    batchNum_experiments = mainDataList_experiments[dataID_experiments]['batchNum']
    spinSpeed_experiments = mainDataList_experiments[dataID_experiments]['spinSpeed']
    numOfRafts_experiments = mainDataList_experiments[dataID_experiments]['numOfRafts']
    numOfFrames_experiments = mainDataList_experiments[dataID_experiments]['numOfFrames']
    raftRadii_experiments = mainDataList_experiments[dataID_experiments]['raftRadii']
    raftLocations_experiments = mainDataList_experiments[dataID_experiments]['raftLocations']
#    raftOrbitingCenters_experiments = mainDataList[dataID_experiments]['raftOrbitingCenters']
#    raftOrbitingDistances_experiments = mainDataList[dataID_experiments]['raftOrbitingDistances']
#    raftOrbitingAngles_experiments = mainDataList[dataID_experiments]['raftOrbitingAngles']
#    raftOrbitingLayerIndices_experiments = mainDataList[dataID_experiments]['raftOrbitingLayerIndices']
    magnification_experiments = mainDataList_experiments[dataID_experiments]['magnification']
    commentsSub_experiments = mainDataList_experiments[dataID_experiments]['commentsSub']
    #dataID_experiments = 3
    variableListFromProcessedFile_experiments = list(mainDataList_experiments[dataID_experiments].keys())

    for key, value in mainDataList_experiments[dataID_experiments].items(): # loop through key-value pairs of python dictionary
        globals()[key] = value

#    outputDataFileName_experiments = date_experiments + '_' + str(numOfRafts_experiments) + 'Rafts_' + str(batchNum_experiments) + '_' + str(spinSpeed_experiments) + 'rps_' + str(magnification_experiments) + 'x_' + commentsSub_experiments 

    ######### load all variables from postprocessed file corresponding to the specific experiment above

    analysisType_experiments = 5 # 1: cluster, 2: cluster+Voronoi, 3: MI, 4: cluster+Voronoi+MI, 5: velocity/MSD + cluster + Voronoi
    
    shelveDataFileName_experiments = expDataDir + '/' + date_experiments + '_' + str(numOfRafts_experiments) + 'Rafts_' + str(batchNum_experiments) + '_' + str(spinSpeed_experiments) + 'rps_' + str(magnification_experiments) + 'x_' + 'postprocessed' + str(analysisType_experiments)
    
    shelveDataFileExist = glob.glob(shelveDataFileName_experiments +'.dat')
    
    if shelveDataFileExist:
        print(shelveDataFileName_experiments + ' exists, load additional variables. ' )
        tempShelf = shelve.open(shelveDataFileName_experiments)
        variableListFromPostProcessedFile_experiments = list(tempShelf.keys())
        
        for key in tempShelf: # just loop through all the keys in the dictionary
            globals()[key] = tempShelf[key]
        
        tempShelf.close()
        print('loading complete.' )
        
    elif len(shelveDataFileExist) == 0:
        print(shelveDataFileName_experiments + ' does not exist')
    
    ######## experimental distribution ######################
    binEdgesNeighborDistances = np.arange(2,8.5,0.5).tolist() + [100]
    neighborDistancesList = np.concatenate(dfNeighborsAllFrames['neighborDistances'].iloc[-numOfRafts_experiments:].tolist())
    count_neighborDistances, _ = np.histogram(np.asarray(neighborDistancesList), binEdgesNeighborDistances)
#    entropyByNeighborDistances[currentFrameNum] = ShannonEntropy(count_neighborDistances)

    
        
    ################## optimise L ###################
    dataID = dataID_experiments
    Beta = 1e14
#    L_optm = optimiseL(spinSpeed_experiments, Beta, count_neighborDistances,  E_total, HydrodynamicsLift_Energy[:cuttOff_distance, dataID])
#    Beta_optm, DeltaEntropy_minm, Hist_P_optm, _ = optimiseBetaEntropy(spinSpeed_experiments, L_optm, count_neighborDistances,  E_total, HydrodynamicsLift_Energy[:cuttOff_distance, dataID])
#    Beta_optm, L_optm, DeltaEntropy_minm, HIst_P_optm, _ = optimiseLBetaEntropy(spinSpeed_experiments, count_neighborDistances, E_total, HydrodynamicsLift_Energy[:cuttOff_distance,dataID])
#    Beta_optm, KLD_minm, Hist_P_optm, _ = optimiseBeta(spinSpeed_experiments, L_optm, count_neighborDistances,  E_total, HydrodynamicsLift_Energy[:cuttOff_distance, dataID])
    Beta_optm, L_optm, KLD_minm, HIst_P_optm, _ = optimiseLBeta(spinSpeed_experiments, count_neighborDistances, E_total, HydrodynamicsLift_Energy[:cuttOff_distance,dataID])
#%% Debug

#%% Hydrodynamics lift force energy and potential energy due to boundary 
miu = 1e-3 # dynamic viscosity of water units Pa.s = 1e-3 N.s/m^2
densityOfWater = 1e3  # units : 1000 kg/m^3 = 1e-15 kg/um^3
Hydrodynamics_EEDistances = np.arange(0, 10001) / 1e6  # unit: m
radiusOfRaft = 1.5e-4  # unit: m

L = 3.6*radiusOfRaft # length of the effective boundary
#L = L_optm # length of the effective boundary
#L = 1.5e-2 # unit micron length of the actual boundary
#R_COM = np.arange(0,L/2 - 2*radiusOfRaft) / 1e6 # unit m radial distance of centre of mass of the pair of rafts
numOfBatches = 59

Hydrodynamics_CCDistances = Hydrodynamics_EEDistances + radiusOfRaft * 2  # unit: m
HydrodynamicsLift_Energy = np.zeros((len(Hydrodynamics_EEDistances), numOfBatches))  # unit: J
Boundary_Energy = np.zeros((len(Hydrodynamics_EEDistances), numOfBatches))  # unit: J
#Boundary_Energy = np.zeros((len(Hydrodynamics_EEDistances), numOfBatches, len(R_COM)))  # unit: J

for dataID in range(0, numOfBatches) :
#    magneticFieldRotationRPS = magneticFieldRotationRPS_all[dataID]
    magneticFieldRotationRPS = dataID + 11
    for index, d in enumerate(Hydrodynamics_CCDistances):
        HydrodynamicsLift_Energy[index, dataID] = densityOfWater*(radiusOfRaft**7)*((magneticFieldRotationRPS*2*np.pi)**2)/(2*(d)**2)
#        Boundary_Energy[index, dataID] = densityOfWater*(radiusOfRaft**7)*((magneticFieldRotationRPS*2*np.pi)**2)*(1/((L + d)**2))
        Boundary_Energy[index, dataID] = densityOfWater*(radiusOfRaft**7)*((magneticFieldRotationRPS*2*np.pi)**2)*(1/((L + d)**2) + (d - 2*radiusOfRaft)/(L**3))
#        Boundary_Energy[index, dataID] = 1e-8*d**2

#%% calculate probability and make histogram of the probability distribution
Beta = 1e12
#Beta = Beta_optm
binEdgesEEDistances = np.arange(2,8.5,0.5).tolist() + [100]

H = np.zeros((cuttOff_distance , numOfBatches))
P = np.zeros((cuttOff_distance , numOfBatches))

for dataID in range(0, numOfBatches) :
    H[:,dataID] = E_total + 2*HydrodynamicsLift_Energy[:cuttOff_distance,dataID] + Boundary_Energy[:cuttOff_distance,dataID]
    P[:,dataID] = np.exp(-Beta*H[:,dataID])


#for dataID in range(0, numOfBatches) :
#    count, edges = np.histogram(P[:,dataID], binEdgesEEDistances)

Hist_P = np.zeros((len(binEdgesEEDistances[:-1]), numOfBatches))

for dataID in range(0, numOfBatches) :
    for i in range(Hist_P.shape[0]) :
#        Hist_P[i,dataID] = P[i*75:(i+1)*75,dataID].mean()
         Hist_P[i,dataID] = P[i*75:(i+1)*75,dataID].sum()/P[:,dataID].sum()
#        Hist_P[i,dataID] = P[i*75:(i+1)*75,dataID].sum()
#    Hist_P[:,dataID] = Hist_P[:,dataID]/Hist_P[:,dataID].sum()

#KLD = entropy(Hist_P[:,14], qk=Hist_P[:,4])

fig, ax = plt.subplots(1,1, figsize = (10,15))
ax.bar(binEdgesEEDistances[:-1], Hist_P[:,12]/Hist_P[:,12].sum(), align = 'edge', width = 0.25)
ax.set_xlabel('EEDistances',  {'size': 15})
ax.set_ylabel('count', {'size': 15})
ax.set_title('histogram of EEDistances')
fig.show()

#peaks, _ = find_peaks(Hist_P[:,14], height=0)
#hist = [P[:,4], np.arange(0,cuttOff_distance + 1,1)]
#hist_dist = scipy.stats.rv_histogram(hist)
#plt.plot(hist[1],hist_dist.pdf(hist[1]))
#count3, _ = np.histogram(np.asarray(neighborDistancesList), binEdgesNeighborDistances)
#entropyByNeighborDistances[currentFrameNum] = ShannonEntropy(count3)