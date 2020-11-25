# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 14:03:30 2019

@author: wwang
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io

import os
import glob
import shelve

rootFolderNameFromWindows = os.getcwd()

os.chdir(rootFolderNameFromWindows)

sym = 6
if sym == 4:
    os.chdir('2019-03-29_capillaryForceCalculations')
elif sym == 6:
    os.chdir('2019-05-13_capillaryForceCalculations-sym6')

# these results are for D=300 micron rafts with amplitude 2um and arc angle 30 degree
if sym == 4:
    resultSurfaceEvolver = scipy.io.loadmat(
        'ResultsCombined_L4_amp2_arcAngle30_ccDist301-350um-count50_rotAngle91_bathRad500.mat')
    resultTheory = scipy.io.loadmat('Results_ccDistance341to1300step1um_angleCount91_errorPower-10_treated.mat')
elif sym == 6:
    resultSurfaceEvolver = scipy.io.loadmat(
        'ResultsCombined_L4_amp2_arcAngle30_ccDist301-350um-count50_rotAngle61_bathRad500.mat')
    resultTheory = scipy.io.loadmat(
        'Results_sym6_arcAngle30_ccDistance341to8000step1um_angleCount61_errorPower-10_treated')
    # Results_sym6_arcAngle30_ccDistance341to1600step1um_angleCount361_errorPower-10_treated.mat

# load data of surface evolver
raftAmpSE = resultSurfaceEvolver['raftAmp'][0][0]  # unit: um
raftRadSE = resultSurfaceEvolver['raftRad'][0][0]  # unit: um
eeDistanceSE = resultSurfaceEvolver['edgeToEdgeDistance'].transpose()  # unit: um
ccDistanceSE = resultSurfaceEvolver['centerToCenterDistance'].transpose()  # unit: um
rotationAnglesSE = resultSurfaceEvolver['rotationAngle'].transpose()  # unit: deg
energySE = resultSurfaceEvolver['netEnergy_2Rafts_reZero'].transpose()  # unit: fJ
forceSE = resultSurfaceEvolver['force'].transpose()  # unit: nN
torqueSE = resultSurfaceEvolver['torque'].transpose()  # unit: nN.um = fN.m
angleAveragedForceSE = resultSurfaceEvolver['angleAveragedForce'].transpose()  # unit: nN
angleAveragedForceNormalizedSE = resultSurfaceEvolver[
    'angleAveragedForceNormalized'].transpose()  # normalized by 74 mN/m * 2 um = 148 nN
angleAveragedNetEnergySE = resultSurfaceEvolver['angleAveragedNetEnergy'].transpose()  # unit: fJ
angleAveragedNetEnergyNormalizedSE = resultSurfaceEvolver[
    'angleAveragedNetEnergyNormalized'].transpose()  # unit: normalized by 74 mN/m * 2 um * 2 um = 296 fJ

# load data from theorectical calculation
raftAmpTheory = resultTheory['raftAmp'][0][0]  # unit: um
raftRadTheory = resultTheory['raftRad'][0][0]  # unit: um
eeDistanceTheory = resultTheory['edgeToEdgeDistance'].transpose()  # unit: um, 41 - 1000 um
ccDistanceTheory = eeDistanceTheory + raftRadTheory * 2  # unit: um
rotationAngleTheory = np.arange(resultTheory['thetaStart'][0][0], resultTheory['thetaEnd'][0][0] + 1,
                                resultTheory['thetaStepSize'][0][0])
energyTheory = resultTheory[
    'energyScaledToRealSurfaceTensionRezeroed']  # unit: fJ, [41-1000um, 0-90deg], rezeroed by energy at 45 deg & 350 um.
forceTheory = resultTheory['force']  # unit: nN,
torqueTheory = resultTheory['torque']  # unit: nN.um = fN.m
angleAveragedForceTheory = resultTheory['angleAveragedForce']  # unit: nN
angleAveragedForceNormalizedTheory = resultTheory[
    'angleAveragedForceNormalized']  # normalized by 74 mN/m * 2 um = 148 nN
angleAveragedNetEnergyTheory = resultTheory['angleAveragedNetEnergy']  # unit: fJ
angleAveragedNetEnergyNormalizedTheory = resultTheory[
    'angleAveragedNetEnergyNormalized']  # unit: normalized by 74 mN/m * 2 um * 2 um = 296 fJ

# combined two datasets
if sym == 4:
    eeDistanceCombined = np.vstack((eeDistanceSE[:40], eeDistanceTheory)) / 1e6  # unit m
    angleAveragedForceCombined = np.vstack((angleAveragedForceSE[:40], angleAveragedForceTheory)) / 1e9  # unit: N
    forceCombinedDistancesAsRows = np.vstack((forceSE[:40, :], forceTheory)) / 1e9  # unit: N
    torqueCombinedDistancesAsRows = np.vstack((torqueSE[:40, :], torqueTheory)) / 1e15  # unit: N.m
    anglesForMaxForce = np.argmax(forceCombinedDistancesAsRows[:, :45],
                                  axis=1)  # max force is positive repulstion, unit: degree
    anglesForMinForce = np.argmin(forceCombinedDistancesAsRows[:, :50],
                                  axis=1)  # min force is negative, attraction unit: degree
    forceMaxRepulsion = forceCombinedDistancesAsRows.max(axis=1)  # unit: N
    forceMaxAttraction = forceCombinedDistancesAsRows.min(axis=1)  # unit: N
elif sym == 6:
    eeDistanceCombined = np.vstack((eeDistanceSE[:40], eeDistanceTheory)) / 1e6  # unit m
    angleAveragedForceCombined = np.vstack((angleAveragedForceSE[:40], angleAveragedForceTheory)) / 1e9  # unit: N
    forceCombinedDistancesAsRows = np.vstack((forceSE[:40, :], forceTheory[:, :61])) / 1e9  # unit: N
    torqueCombinedDistancesAsRows = np.vstack((torqueSE[:40, :], torqueTheory[:, :61])) / 1e15  # unit: N.m
    anglesForMaxForce = np.argmax(forceCombinedDistancesAsRows[:, :45],
                                  axis=1)  # max force is positive repulstion, unit: degree
    anglesForMinForce = np.argmin(forceCombinedDistancesAsRows[:, :50],
                                  axis=1)  # min force is negative, attraction unit: degree
    forceMaxRepulsion = forceCombinedDistancesAsRows.max(axis=1)  # unit: N
    forceMaxAttraction = forceCombinedDistancesAsRows.min(axis=1)  # unit: N

forceCombinedAnglesAsRows = forceCombinedDistancesAsRows.transpose()  # unit N
torqueCombinedAnglesAsRows = torqueCombinedDistancesAsRows.transpose()  # unit N.m

#  extends to all 360 deg, the torque at the begining and the end needs to readjusted to 0
#  because those values are due to one-sided gradient.
if sym == 4:
    forceCombinedDistancesAsRowsAll360 = np.hstack((forceCombinedDistancesAsRows[:, :90],
                                                    forceCombinedDistancesAsRows[:, :90],
                                                    forceCombinedDistancesAsRows[:, :90],
                                                    forceCombinedDistancesAsRows[:, :90]))
    torqueCombinedDistancesAsRows[:, 0] = 0
    torqueCombinedDistancesAsRows[:, -1] = 0
    torqueCombinedDistancesAsRowsAll360 = np.hstack((torqueCombinedDistancesAsRows[:, :90],
                                                     torqueCombinedDistancesAsRows[:, :90],
                                                     torqueCombinedDistancesAsRows[:, :90],
                                                     torqueCombinedDistancesAsRows[:, :90]))
elif sym == 6:
    forceCombinedDistancesAsRowsAll360 = np.hstack((forceCombinedDistancesAsRows[:, :60],
                                                    forceCombinedDistancesAsRows[:, :60],
                                                    forceCombinedDistancesAsRows[:, :60],
                                                    forceCombinedDistancesAsRows[:, :60],
                                                    forceCombinedDistancesAsRows[:, :60],
                                                    forceCombinedDistancesAsRows[:, :60]))
    torqueCombinedDistancesAsRows[:, 0] = 0
    torqueCombinedDistancesAsRows[:, -1] = 0
    torqueCombinedDistancesAsRowsAll360 = np.hstack((torqueCombinedDistancesAsRows[:, :60],
                                                     torqueCombinedDistancesAsRows[:, :60],
                                                     torqueCombinedDistancesAsRows[:, :60],
                                                     torqueCombinedDistancesAsRows[:, :60],
                                                     torqueCombinedDistancesAsRows[:, :60],
                                                     torqueCombinedDistancesAsRows[:, :60]))

outputDataFileName = 'capillaryForceAndTorque_sym' + str(sym)
listOfVariablesToSave = ['eeDistanceCombined', 'forceCombinedDistancesAsRowsAll360',
                         'torqueCombinedDistancesAsRowsAll360']

tempShelf = shelve.open(outputDataFileName, 'n')  # 'n' for new
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
