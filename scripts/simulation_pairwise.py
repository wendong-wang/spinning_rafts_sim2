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

from scripts.functions_spinning_rafts import draw_rafts_rh_coord, draw_b_field_in_rh_coord, draw_cap_peaks_rh_coord, \
    draw_raft_orientations_rh_coord, draw_raft_num_rh_coord, draw_frame_info


projectDir = "D:\\simulationFolder\\spinning_rafts_simulation_code"  # os.getcwd()
capSym6Dir = projectDir + "\\2019-05-13_capillaryForceCalculations-sym6"
capSym4Dir = projectDir + "\\2019-03-29_capillaryForceCalculations"
os.chdir("..")
outputDir = os.getcwd()


# %% load capillary force and torque

os.chdir(capSym6Dir)

shelveName = 'capillaryForceAndTorque_sym6'
shelveDataFileName = shelveName + '.dat'
listOfVariablesToLoad = ['eeDistanceCombined', 'forceCombinedDistancesAsRowsAll360',
                         'torqueCombinedDistancesAsRowsAll360']

if not os.path.isfile(shelveDataFileName):
    print('the capillary data file is missing')

tempShelf = shelve.open(shelveName)
capillaryEEDistances = tempShelf['eeDistanceCombined']  # unit: m
capillaryForcesDistancesAsRowsLoaded = tempShelf['forceCombinedDistancesAsRowsAll360']  # unit: N
capillaryTorquesDistancesAsRowsLoaded = tempShelf['torqueCombinedDistancesAsRowsAll360']  # unit: N.m

# further data treatment on capillary force profile
# insert the force and torque at eeDistance = 1um as the value for eedistance = 0um.
capillaryEEDistances = np.insert(capillaryEEDistances, 0, 0)
capillaryForcesDistancesAsRows = np.concatenate(
    (capillaryForcesDistancesAsRowsLoaded[:1, :], capillaryForcesDistancesAsRowsLoaded), axis=0)
capillaryTorquesDistancesAsRows = np.concatenate(
    (capillaryTorquesDistancesAsRowsLoaded[:1, :], capillaryTorquesDistancesAsRowsLoaded), axis=0)

# add angle=360, the same as angle = 0
capillaryForcesDistancesAsRows = np.concatenate(
    (capillaryForcesDistancesAsRows, capillaryForcesDistancesAsRows[:, 0].reshape(1001, 1)), axis=1)
capillaryTorquesDistancesAsRows = np.concatenate(
    (capillaryTorquesDistancesAsRows, capillaryTorquesDistancesAsRows[:, 0].reshape(1001, 1)), axis=1)

# correct for the negative sign of the torque
capillaryTorquesDistancesAsRows = - capillaryTorquesDistancesAsRows

# some extra treatment for the force matrix
# note the sharp transition at the peak-peak position (45 deg): only 1 deg difference,
# the force changes from attraction to repulsion. consider replacing values at eeDistance = 0, 1, 2,
# with values at eeDistance = 5um.
nearEdgeSmoothingThres = 1  # unit: micron; if 1, then it is equivalent to no smoothing.
for distanceToEdge in np.arange(nearEdgeSmoothingThres):
    capillaryForcesDistancesAsRows[distanceToEdge, :] = capillaryForcesDistancesAsRows[nearEdgeSmoothingThres, :]
    capillaryTorquesDistancesAsRows[distanceToEdge, :] = capillaryTorquesDistancesAsRows[nearEdgeSmoothingThres, :]

# select a cut-off distance below which all the attractive force (negative-valued) becomes zero,
# due to raft wall-wall repulsion
capAttractionZeroCutoff = 0
mask = np.concatenate((capillaryForcesDistancesAsRows[:capAttractionZeroCutoff, :] < 0,
                       np.zeros((capillaryForcesDistancesAsRows.shape[0] - capAttractionZeroCutoff,
                                 capillaryForcesDistancesAsRows.shape[1]), dtype=int)),
                      axis=0)
capillaryForcesDistancesAsRows[mask.nonzero()] = 0

# set capillary force = 0 at 0 distance
# capillaryForcesDistancesAsRows[0,:] = 0

# realign the first peak-peak direction with an angle = capillaryPeakOffset from the x-axis.
capillaryPeakOffset = 0
capillaryForcesDistancesAsRows = np.roll(capillaryForcesDistancesAsRows, capillaryPeakOffset,
                                         axis=1)  # 45 is due to original data
capillaryTorquesDistancesAsRows = np.roll(capillaryTorquesDistancesAsRows, capillaryPeakOffset, axis=1)

capillaryForceAngleAveraged = capillaryForcesDistancesAsRows[1:, :-1].mean(axis=1)  # starting from 1 um to 1000 um
capillaryForceMaxRepulsion = capillaryForcesDistancesAsRows[1:, :-1].max(axis=1)
capillaryForceMaxRepulsionIndex = capillaryForcesDistancesAsRows[1:, :-1].argmax(axis=1)
capillaryForceMaxAttraction = capillaryForcesDistancesAsRows[1:, :-1].min(axis=1)
capillaryForceMaxAttractionIndex = capillaryForcesDistancesAsRows[1:, :-1].argmin(axis=1)

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

# magDpEnergy = np.zeros((len(magneticDipoleEEDistances), len(orientationAngles)))  # unit: J
magDpForceOnAxis = np.zeros((len(magneticDipoleEEDistances), len(orientationAngles)))  # unit: N
magDpForceOffAxis = np.zeros((len(magneticDipoleEEDistances), len(orientationAngles)))  # unit: N
magDpTorque = np.zeros((len(magneticDipoleEEDistances), len(orientationAngles)))  # unit: N.m

for index, d in enumerate(magneticDipoleCCDistances):
    # magDpEnergy[index, :] = \
    #     miu0 * magneticMomentOfOneRaft ** 2 * (1 - 3 * (np.cos(orientationAnglesInRad) ** 2)) / (4 * np.pi * d ** 3)
    magDpForceOnAxis[index, :] = \
        3 * miu0 * magneticMomentOfOneRaft ** 2 * (1 - 3 * (np.cos(orientationAnglesInRad) ** 2)) / (4 * np.pi * d ** 4)
    magDpForceOffAxis[index, :] = \
        3 * miu0 * magneticMomentOfOneRaft ** 2 * (2 * np.cos(orientationAnglesInRad) *
                                                   np.sin(orientationAnglesInRad)) / (4 * np.pi * d ** 4)
    magDpTorque[index, :] = \
        miu0 * magneticMomentOfOneRaft ** 2 * (3 * np.cos(orientationAnglesInRad) *
                                               np.sin(orientationAnglesInRad)) / (4 * np.pi * d ** 3)

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

# %% lubrication equation coefficients:
RforCoeff = 150.0  # unit: micron
stepSizeForDist = 0.1
lubCoeffScaleFactor = 1 / stepSizeForDist
eeDistancesForCoeff = np.arange(0, 15 + stepSizeForDist, stepSizeForDist, dtype='double')  # unit: micron

eeDistancesForCoeff[0] = 1e-10  # unit: micron

x = eeDistancesForCoeff / RforCoeff  # unit: 1

lubA = x * (-0.285524 * x + 0.095493 * x * np.log(x) + 0.106103) / RforCoeff  # unit: 1/um

lubB = ((0.0212764 * (- np.log(x)) + 0.157378) * (- np.log(x)) + 0.269886) / (
        RforCoeff * (- np.log(x)) * ((- np.log(x)) + 6.0425) + 6.32549)  # unit: 1/um

# lubC = ((-0.0212758 * (- np.log(x)) - 0.089656) * (- np.log(x)) + 0.0480911) / \
#        (RforCoeff ** 2 * ((- np.log(x)) * ((- np.log(x)) + 6.0425) + 6.32549))  # unit: 1/um^2

# lubD = (0.0579125 * (- np.log(x)) + 0.0780201) / \
#        (RforCoeff ** 2 * ((- np.log(x)) * ((- np.log(x)) + 6.0425) + 6.32549))  # unit: 1/um^2

lubG = ((0.0212758 * (- np.log(x)) + 0.181089) * (- np.log(x)) + 0.381213) / (
        RforCoeff ** 3 * ((- np.log(x)) * ((- np.log(x)) + 6.0425) + 6.32549))  # unit: 1/um^3

lubC = - RforCoeff * lubG

# lubH = (0.265258 * (- np.log(x)) + 0.357355) / \
#        (RforCoeff ** 3 * ((- np.log(x)) * ((- np.log(x)) + 6.0425) + 6.32549))  # unit: 1/um^3

# lubCoeffCombined = np.column_stack((lubA,lubB,lubC,lubD,lubG,lubH))

# %% check capillary and magnetic forces
# all calculations are done in SI numbers, and only in drawing are the variables converted to pixel unit

# check the dipole orientation and capillary orientation
# eeDistanceForPlotting = 70
# fig, ax = plt.subplots(ncols=2, nrows=1)
# ax[0].plot(capillaryForcesDistancesAsRows[eeDistanceForPlotting, :], 'o-',
#            label='capillary force')  # 0 deg is the peak-peak alignment - attraction.
# ax[0].plot(magDpForceOnAxis[eeDistanceForPlotting, :], 'o-',
#            label='magnetic force')  # 0 deg is the dipole-dipole attraction
# ax[0].set_xlabel('angle')
# ax[0].set_ylabel('force (N)')
# ax[0].legend()
# ax[0].set_title('force at eeDistance = {}um, capillary peak offset angle = {}deg'.format(eeDistanceForPlotting,
#                                                                                          capillaryPeakOffset))
# ax[1].plot(capillaryTorquesDistancesAsRows[eeDistanceForPlotting, :], 'o-',
#            label='capillary torque')  # 0 deg is the peak-peak alignment - attraction.
# ax[1].plot(magDpTorque[eeDistanceForPlotting, :], 'o-',
#            label='magnetic torque')  # 0 deg is the dipole-dipole attraction
# ax[1].set_xlabel('angle')
# ax[1].set_ylabel('torque (N.m)')
# ax[1].legend()
# ax[1].set_title('torque at eeDistance = {}um, capillary peak offset angle = {}deg'.format(eeDistanceForPlotting,
#                                                                                           capillaryPeakOffset))

# plot the various forces and look for the transition rps
# densityOfWater = 1e-15 # unit conversion: 1000 kg/m^3 = 1e-15 kg/um^3
# raftRadius = 1.5e2 # unit: micron
# magneticFieldRotationRPS = 22
# omegaBField = magneticFieldRotationRPS * 2 * np.pi
# hydrodynamicRepulsion = densityOfWater * omegaBField ** 2 * raftRadius ** 7 * 1e-6 / \
#                         np.arange(raftRadius * 2 + 1, raftRadius * 2 + 1002) ** 3  # unit: N
# sumOfAllForces = capillaryForcesDistancesAsRows.mean(axis=1) + magDpForceOnAxis.mean(axis=1)[
#                                                                :1001] + hydrodynamicRepulsion
# fig, ax = plt.subplots(ncols = 1, nrows = 1)
# ax.plot(capillaryForcesDistancesAsRows.mean(axis = 1), label = 'angle-averaged capillary force')
# ax.plot(magDpForceOnAxis.mean(axis = 1)[:1000], label = 'angle-averaged magnetic force')
# ax.plot(hydrodynamicRepulsion, label = 'hydrodynamic repulsion')
# ax.plot(capillaryForcesDistancesAsRows.mean(axis = 1) + magDpForceOnAxis.mean(axis = 1)[:1001],
#         label = 'angle-avaraged sum of magnetic and capillary force')
# ax.set_xlabel('edge-edge distance (um)')
# ax.set_ylabel('Force (N)')
# ax.set_title('spin speed {} rps'.format(magneticFieldRotationRPS))
# ax.plot(sumOfAllForces, label = 'sum of angle-averaged magnetic and capillary forces and hydrodynamic force ')
# ax.legend()

#%% simulation of the pairwise
os.chdir(outputDir)

listOfVariablesToSave = ['numOfRafts', 'magneticFieldStrength', 'magneticFieldRotationRPS', 'omegaBField',
                         'timeStepSize', 'numOfTimeSteps',
                         'timeTotal', 'outputImageSeq', 'outputVideo', 'outputFrameRate', 'intervalBetweenFrames',
                         'raftLocations', 'raftOrientations', 'raftRadii', 'raftRotationSpeedsInRad',
                         'raftRelativeOrientationInDeg',
                         # 'velocityTorqueCouplingTerm', 'magDipoleForceOffAxisTerm', 'magDipoleForceOnAxisTerm',
                         # 'capillaryForceTerm', 'hydrodynamicForceTerm', 'stochasticTerm', 'forceCurvatureTerm',
                         # 'wallRepulsionTerm', 'magneticFieldTorqueTerm', 'magneticDipoleTorqueTerm',
                         # 'capillaryTorqueTerm',
                         'currentStepNum', 'currentFrameBGR']
# small number threshold
# eps = 1e-13

# constants of proportionality
cm = 1  # coefficient for the magnetic force term
cc = 1  # coefficient for the capillary force term
ch = 1  # coefficient for the hydrodynamic force term
tb = 1  # coefficient for the magnetic field torque term
tm = 1  # coefficient for the magnetic dipole-dipole torque term
tc = 1  # coefficient for the capillary torque term
forceDueToCurvature = 0  # unit: N
wallRepulsionForce = 1e-7  # unit: N
# elasticWallThickness = 5 # unit: micron

arenaSize = 2e3  # unit: micron 2e3, 5e3,
R = raftRadius = 1.5e2  # unit: micron
centerOfArena = np.array([arenaSize / 2, arenaSize / 2])

# all calculations are done in SI numbers, and only in drawing are the variables converted to pixel unit
canvasSizeInPixel = int(1000)  # unit: pixel
scaleBar = arenaSize / canvasSizeInPixel  # unit: micron/pixel

densityOfWater = 1e-15  # unit conversion: 1000 kg/m^3 = 1e-15 kg/um^3
miu = 1e-15  # dynamic viscosity of water; unit conversion: 1e-3 Pa.s = 1e-3 N.s/m^2 = 1e-15 N.s/um^2
piMiuR = np.pi * miu * raftRadius  # unit: N.s/um

numOfRafts = 2
magneticFieldStrength = 10e-3  # 14e-3 #10e-3 # unit: T
initialPositionMethod = 2  # 1 -random positions, 2 - fixed initial position,
# 3 - starting positions are the last positions of the previous spin speeds
ccSeparationStarting = 400  # unit: micron
initialOrientation = 0  # unit: deg
lastPositionOfPreviousSpinSpeeds = np.zeros((numOfRafts, 2))
lastOmegaOfPreviousSpinSpeeds = np.zeros(numOfRafts)
firstSpinSpeedFlag = 1

timeStepSize = 1e-3  # unit: s
numOfTimeSteps = 10000
timeTotal = timeStepSize * numOfTimeSteps

lubEqThreshold = 15  # unit micron, if the eeDistance is below this value, use lubrication equations
stdOfFluctuationTerm = 0.00
stdOfTorqueNoise = 0  # 1e-12 # unit: N.m

outputImageSeq = 0
outputVideo = 1
outputFrameRate = 10.0
intervalBetweenFrames = int(10)  # unit: steps
blankFrameBGR = np.ones((canvasSizeInPixel, canvasSizeInPixel, 3), dtype='int') * 255

solverMethod = 'RK45'  # RK45, RK23, Radau, BDF, LSODA


def funct_drdt_dalphadt(t, raft_loc_orient):
    """
    Two sets of ordinary differential equations that define dr/dt and dalpha/dt above and below the threshold value
    for the application of lubrication equations
    """
    #    raft_loc_orient = raftLocationsOrientations
    raft_loc = raft_loc_orient[0: numOfRafts * 2].reshape(numOfRafts, 2)  # in um
    raft_orient = raft_loc_orient[numOfRafts * 2: numOfRafts * 3]  # in deg

    drdt = np.zeros((numOfRafts, 2))  # unit: um
    raft_spin_speeds_in_rads = np.zeros(numOfRafts)  # in rad
    dalphadt = np.zeros(numOfRafts)  # unit: deg

    mag_dipole_force_on_axis_term = np.zeros((numOfRafts, 2))
    capillary_force_term = np.zeros((numOfRafts, 2))
    hydrodynamic_force_term = np.zeros((numOfRafts, 2))
    mag_dipole_force_off_axis_term = np.zeros((numOfRafts, 2))
    velocity_torque_coupling_term = np.zeros((numOfRafts, 2))
    velocity_mag_fd_torque_term = np.zeros((numOfRafts, 2))
    wall_repulsion_term = np.zeros((numOfRafts, 2))
    stochastic_force_term = np.zeros((numOfRafts, 2))
    force_curvature_term = np.zeros((numOfRafts, 2))

    magnetic_field_torque_term = np.zeros(numOfRafts)
    magnetic_dipole_torque_term = np.zeros(numOfRafts)
    capillary_torque_term = np.zeros(numOfRafts)
    stochastic_torque_term = np.zeros(numOfRafts)

    # stochastic torque term
    # stochastic_torque = omegaBField * np.random.normal(0, stdOfTorqueNoise, 1)
    # unit: N.m, assuming omegaBField is unitless
    # stochastic_torque_term = np.ones(numOfRafts) * stochastic_torque * 1e6 / (8 * piMiuR * R ** 2)
    # unit: 1/s assuming omegaBField is unitless.

    # loop for torques and calculate raft_spin_speeds_in_rads
    for raft_id in np.arange(numOfRafts):
        # raft_id = 0
        ri = raft_loc[raft_id, :]  # unit: micron

        # magnetic field torque:
        magnetic_field_torque = magneticFieldStrength * magneticMomentOfOneRaft * np.sin(
            np.deg2rad(magneticFieldDirection - raft_orient[raft_id]))  # unit: N.m
        magnetic_field_torque_term[raft_id] = tb * magnetic_field_torque * 1e6 / (8 * piMiuR * R ** 2)  # unit: 1/s

        rji_ee_dist_smallest = R  # initialize

        for neighbor_id in np.arange(numOfRafts):
            if neighbor_id == raft_id:
                continue
            rj = raft_loc[neighbor_id, :]  # unit: micron
            rji = ri - rj  # unit: micron
            rji_norm = np.sqrt(rji[0] ** 2 + rji[1] ** 2)  # unit: micron
            rji_ee_dist = rji_norm - 2 * R  # unit: micron
            rji_unitized = rji / rji_norm  # unit: micron
            rji_unitized_cross_z = np.asarray((rji_unitized[1], -rji_unitized[0]))
            phi_ji = (np.arctan2(rji[1], rji[0]) * 180 / np.pi - raft_orient[
                raft_id]) % 360  # unit: deg; assuming both rafts's orientations are the same

            #  print('{}, {}'.format(int(phi_ji), (np.arctan2(rji[1], rji[0]) * 180 / np.pi - raft_orient[raftID])))
            #  torque terms:
            if rji_ee_dist < lubEqThreshold and rji_ee_dist < rji_ee_dist_smallest:
                rji_ee_dist_smallest = rji_ee_dist
                if rji_ee_dist_smallest >= 0:
                    magnetic_field_torque_term[raft_id] = lubG[int(
                        rji_ee_dist_smallest * lubCoeffScaleFactor)] * magnetic_field_torque * 1e6 / miu  # unit: 1/s
                elif rji_ee_dist_smallest < 0:
                    magnetic_field_torque_term[raft_id] = lubG[0] * magnetic_field_torque * 1e6 / miu  # unit: 1/s

            if 10000 > rji_ee_dist >= 0:
                magnetic_dipole_torque_term[raft_id] = magnetic_dipole_torque_term[raft_id] + tm * magDpTorque[
                    int(rji_ee_dist + 0.5), int(phi_ji + 0.5)] * 1e6 / (8 * piMiuR * R ** 2)
            elif lubEqThreshold > rji_ee_dist >= 0:
                magnetic_dipole_torque_term[raft_id] = magnetic_dipole_torque_term[raft_id] \
                                                       + tm * lubG[int(rji_ee_dist * lubCoeffScaleFactor)] \
                                                       * magDpTorque[int(rji_ee_dist + 0.5), int(phi_ji + 0.5)] \
                                                       * 1e6 / miu  # unit: 1/s
            elif rji_ee_dist < 0:
                magnetic_dipole_torque_term[raft_id] = magnetic_dipole_torque_term[raft_id] + tm * lubG[0] * \
                                                       magDpTorque[0, int(phi_ji + 0.5)] * 1e6 / miu  # unit: 1/s

            if 1000 > rji_ee_dist >= lubEqThreshold:
                capillary_torque_term[raft_id] = capillary_torque_term[raft_id] + tc * \
                                                 capillaryTorquesDistancesAsRows[int(rji_ee_dist + 0.5),
                                                                                 int(phi_ji + 0.5)] \
                                                 * 1e6 / (8 * piMiuR * R ** 2)  # unit: 1/s
            elif lubEqThreshold > rji_ee_dist >= 0:
                capillary_torque_term[raft_id] = capillary_torque_term[raft_id] + tc * lubG[int(rji_ee_dist *
                                                                                                lubCoeffScaleFactor)] \
                                                 * capillaryTorquesDistancesAsRows[int(rji_ee_dist + 0.5),
                                                                                   int(phi_ji + 0.5)] \
                                                 * 1e6 / miu  # unit: 1/s
            elif rji_ee_dist < 0:
                capillary_torque_term[raft_id] = capillary_torque_term[raft_id] + tc * lubG[0] * \
                                                 capillaryTorquesDistancesAsRows[
                                                     0, int(phi_ji + 0.5)] * 1e6 / miu  # unit: 1/s

            # debug use:
        #       raftRelativeOrientationInDeg[neighborID, raftID, currentStepNum] = phi_ji

        # debug use
        #       capillaryTorqueTerm[raftID, currentStepNum] = capillary_torque_term[raftID]

        raft_spin_speeds_in_rads[raft_id] = stochastic_torque_term[raft_id] + magnetic_field_torque_term[raft_id] \
                                            + magnetic_dipole_torque_term[raft_id] + capillary_torque_term[raft_id]

    # loop for forces
    for raft_id in np.arange(numOfRafts):
        # raftID = 0
        ri = raft_loc[raft_id, :]  # unit: micron

        # force curvature term
        if forceDueToCurvature != 0:
            ri_center = centerOfArena - ri
            #            ri_center_Norm = np.sqrt(ri_center[0]**2 + ri_center[1]**2)
            #            ri_center_Unitized = ri_center / ri_center_Norm
            force_curvature_term[raft_id, :] = forceDueToCurvature / (6 * piMiuR) * ri_center / (arenaSize / 2)

        # magnetic field torque:
        magnetic_field_torque = magneticFieldStrength * magneticMomentOfOneRaft * np.sin(
            np.deg2rad(magneticFieldDirection - raft_orient[raft_id]))  # unit: N.m
        magnetic_field_torque_term[raft_id] = tb * magnetic_field_torque * 1e6 / (8 * piMiuR * R ** 2)  # unit: 1/s

        for neighbor_id in np.arange(numOfRafts):
            if neighbor_id == raft_id:
                continue
            rj = raft_loc[neighbor_id, :]  # unit: micron
            rji = ri - rj  # unit: micron
            rji_norm = np.sqrt(rji[0] ** 2 + rji[1] ** 2)  # unit: micron
            rji_ee_dist = rji_norm - 2 * R  # unit: micron
            rji_unitized = rji / rji_norm  # unit: micron
            rji_unitized_cross_z = np.asarray((rji_unitized[1], -rji_unitized[0]))
            phi_ji = (np.arctan2(rji[1], rji[0]) * 180 / np.pi - raft_orient[raft_id]) % 360
            # unit: deg; assuming both rafts's orientations are the same, modulo operation!
            # if phi_ji == 360:
            #     phi_ji = 0
            # raft_Relative_Orientation_InDeg[neighborID, raftID] = phi_ji

            # force terms:
            omegaj = raft_spin_speeds_in_rads[neighbor_id]
            # need to come back and see how to deal with this. maybe you need to define it as a global variable.

            if 10000 > rji_ee_dist >= lubEqThreshold:
                mag_dipole_force_on_axis_term[raft_id, :] = mag_dipole_force_on_axis_term[raft_id, :] + cm * \
                                                            magDpForceOnAxis[int(rji_ee_dist + 0.5), int(
                                                                phi_ji + 0.5)] * rji_unitized / (
                                                                    6 * piMiuR)  # unit: um/s
            elif lubEqThreshold > rji_ee_dist >= 0:
                mag_dipole_force_on_axis_term[raft_id, :] = mag_dipole_force_on_axis_term[raft_id, :] + cm * lubA[
                    int(rji_ee_dist * lubCoeffScaleFactor)] * magDpForceOnAxis[int(rji_ee_dist + 0.5), int(
                    phi_ji + 0.5)] * rji_unitized / miu  # unit: um/s
            elif rji_ee_dist < 0:
                mag_dipole_force_on_axis_term[raft_id, :] = mag_dipole_force_on_axis_term[raft_id, :] + cm * lubA[0] * \
                                                            magDpForceOnAxis[
                                                                0, int(phi_ji + 0.5)] * rji_unitized / miu  # unit: um/s

            if 1000 > rji_ee_dist >= lubEqThreshold:
                capillary_force_term[raft_id, :] = capillary_force_term[raft_id, :] + cc * \
                                                   capillaryForcesDistancesAsRows[int(rji_ee_dist + 0.5),
                                                                                  int(phi_ji + 0.5)] \
                                                   * rji_unitized / (6 * piMiuR)  # unit: um/s
            elif lubEqThreshold > rji_ee_dist >= 0:
                capillary_force_term[raft_id, :] = capillary_force_term[raft_id, :] + cc * \
                                                   lubA[int(rji_ee_dist * lubCoeffScaleFactor)] * \
                                                   capillaryForcesDistancesAsRows[int(rji_ee_dist + 0.5),
                                                                                  int(phi_ji + 0.5)] \
                                                   * rji_unitized / miu  # unit: um/s
            elif rji_ee_dist < 0:
                capillary_force_term[raft_id, :] = capillary_force_term[raft_id, :] + cc * lubA[0] * \
                                                   capillaryForcesDistancesAsRows[
                                                       0, int(phi_ji + 0.5)] * rji_unitized / miu  # unit: um/s

            if rji_ee_dist >= lubEqThreshold:
                hydrodynamic_force_term[raft_id, :] = hydrodynamic_force_term[raft_id, :] \
                                                      + ch * 1e-6 * densityOfWater * omegaj ** 2 * R ** 7 \
                                                      * rji / rji_norm ** 4 / (6 * piMiuR)
                # unit: um/s; 1e-6 is used to convert the implicit m to um in Newton in miu

            elif lubEqThreshold > rji_ee_dist > 0:
                hydrodynamic_force_term[raft_id, :] = hydrodynamic_force_term[raft_id, :] \
                                                      + ch * lubA[int(rji_ee_dist * lubCoeffScaleFactor)] \
                                                      * (1e-6 * densityOfWater * omegaj ** 2 * R ** 7
                                                         / rji_norm ** 3) * rji_unitized / miu  # unit: um/s

            if 10000 > rji_ee_dist >= lubEqThreshold:
                mag_dipole_force_off_axis_term[raft_id, :] = mag_dipole_force_off_axis_term[raft_id, :] \
                                                             + magDpForceOffAxis[int(rji_ee_dist + 0.5),
                                                                                 int(phi_ji + 0.5)] \
                                                             * rji_unitized_cross_z / (6 * piMiuR)
            elif lubEqThreshold > rji_ee_dist >= 0:
                mag_dipole_force_off_axis_term[raft_id, :] = mag_dipole_force_off_axis_term[raft_id, :] \
                                                             + lubB[int(rji_ee_dist * lubCoeffScaleFactor)] \
                                                             * magDpForceOffAxis[int(rji_ee_dist + 0.5),
                                                                                 int(phi_ji + 0.5)] \
                                                             * rji_unitized_cross_z / miu  # unit: um/s
            elif rji_ee_dist < 0:
                mag_dipole_force_off_axis_term[raft_id, :] = mag_dipole_force_off_axis_term[raft_id, :] \
                                                             + lubB[0] * magDpForceOffAxis[0, int(phi_ji + 0.5)] \
                                                             * rji_unitized_cross_z / miu  # unit: um/s

            if rji_ee_dist >= lubEqThreshold:
                velocity_torque_coupling_term[raft_id, :] = velocity_torque_coupling_term[raft_id, :] - R ** 3 * \
                                                            omegaj * rji_unitized_cross_z / (rji_norm ** 2)  # um/s
            elif lubEqThreshold > rji_ee_dist >= 0:
                velocity_mag_fd_torque_term[raft_id, :] = velocity_mag_fd_torque_term[raft_id, :] \
                                                          + lubC[int(rji_ee_dist * lubCoeffScaleFactor)] \
                                                          * magnetic_field_torque * 1e6 \
                                                          * rji_unitized_cross_z / miu  # unit: um/s
            elif rji_ee_dist < 0:
                velocity_mag_fd_torque_term[raft_id, :] = velocity_mag_fd_torque_term[raft_id, :] + lubC[0] * \
                                                          magnetic_field_torque * 1e6 \
                                                          * rji_unitized_cross_z / miu  # unit: um/s

            # if rji_ee_dist >= lubEqThreshold and currentStepNum > 1:
            #     prev_drdt = (raftLocations[raft_id, currentStepNum, :] -
            #                  raftLocations[raft_id, currentStepNum - 1, :]) / timeStepSize
            #     stochastic_force_term[raft_id, currentStepNum, :] = stochastic_force_term[raft_id,
            #                                                                               currentStepNum, :] \
            #                                                        + np.sqrt(prev_drdt[0] ** 2
            #                                                                  + prev_drdt[1] ** 2) * \
            #                                                        np.random.normal(0, stdOfFluctuationTerm, 1) \
            #                                                        * rji_unitized

            if rji_ee_dist < 0:
                wall_repulsion_term[raft_id, :] = wall_repulsion_term[raft_id, :] + wallRepulsionForce / (6 * piMiuR) \
                                                  * (-rji_ee_dist / R) * rji_unitized

        # update drdr and dalphadt
        drdt[raft_id, :] = mag_dipole_force_on_axis_term[raft_id, :] \
                           + capillary_force_term[raft_id, :] + hydrodynamic_force_term[raft_id, :] \
                           + mag_dipole_force_off_axis_term[raft_id, :] \
                           + velocity_torque_coupling_term[raft_id, :] + velocity_mag_fd_torque_term[raft_id, :] \
                           + stochastic_force_term[raft_id, :] + wall_repulsion_term[raft_id, :] \
                           + force_curvature_term[raft_id, :]

    dalphadt = raft_spin_speeds_in_rads / np.pi * 180  # in deg

    drdt_dalphadt = np.concatenate((drdt.flatten(), dalphadt))

    return drdt_dalphadt


# for stdOfFluctuationTerm in np.arange(0.01,0.11,0.04):
for magneticFieldRotationRPS in np.arange(-20, -22, -1):
    # negative magneticFieldRotationRPS means clockwise in rh coordinate,
    # positive magneticFieldRotationRPS means counter-clockwise
    # magneticFieldRotationRPS = -10 # unit: rps (rounds per seconds)
    omegaBField = magneticFieldRotationRPS * 2 * np.pi  # unit: rad/s

    # initialize key dataset
    raftLocations = np.zeros((numOfRafts, numOfTimeSteps, 2))  # in microns
    raftOrientations = np.zeros((numOfRafts, numOfTimeSteps))  # in deg
    raftRadii = np.ones(numOfRafts) * raftRadius  # in micron
    raftRotationSpeedsInRad = np.zeros((numOfRafts, numOfTimeSteps))  # in rad
    raftRelativeOrientationInDeg = np.zeros((numOfRafts, numOfRafts, numOfTimeSteps))
    #  in deg, (neighborID, raftID, frame#)

    #    magDipoleForceOnAxisTerm = np.zeros((numOfRafts, numOfTimeSteps, 2))
    #    capillaryForceTerm = np.zeros((numOfRafts, numOfTimeSteps, 2))
    #    hydrodynamicForceTerm = np.zeros((numOfRafts, numOfTimeSteps, 2))
    #    magDipoleForceOffAxisTerm = np.zeros((numOfRafts, numOfTimeSteps, 2))
    #    velocityTorqueCouplingTerm = np.zeros((numOfRafts, numOfTimeSteps, 2))
    #    velocityMagFdTorqueTerm = np.zeros((numOfRafts, numOfTimeSteps, 2))
    #    wallRepulsionTerm = np.zeros((numOfRafts, numOfTimeSteps, 2))
    #    stochasticForceTerm = np.zeros((numOfRafts, numOfTimeSteps, 2))
    #    forceCurvatureTerm = np.zeros((numOfRafts, numOfTimeSteps, 2))
    #
    #    magneticFieldTorqueTerm = np.zeros((numOfRafts, numOfTimeSteps))
    #    magneticDipoleTorqueTerm = np.zeros((numOfRafts, numOfTimeSteps))
    #    capillaryTorqueTerm = np.zeros((numOfRafts, numOfTimeSteps))
    #    stochasticTorqueTerm = np.zeros((numOfRafts, numOfTimeSteps))

    currentStepNum = 0
    if initialPositionMethod == 1:
        # initialize the raft positions in the first frame, check pairwise ccdistance all above 2R
        paddingAroundArena = 20  # unit: radius
        ccDistanceMin = 2.5  # unit: radius
        raftLocations[:, currentStepNum, :] = np.random.uniform(0 + raftRadius * paddingAroundArena,
                                                                arenaSize - raftRadius * paddingAroundArena,
                                                                (numOfRafts, 2))

        raftsToRelocate = np.arange(1, numOfRafts)
        while len(raftsToRelocate) > 0:
            raftLocations[raftsToRelocate, currentStepNum, :] = np.random.uniform(0 + raftRadius * paddingAroundArena,
                                                                                  arenaSize - raftRadius
                                                                                  * paddingAroundArena,
                                                                                  (len(raftsToRelocate), 2))
            pairwiseDistances = scipy_distance.cdist(raftLocations[:, currentStepNum, :],
                                                     raftLocations[:, currentStepNum, :], 'euclidean')
            np.fill_diagonal(pairwiseDistances, raftRadius * ccDistanceMin + 1)
            raftsToRelocate, _ = np.nonzero(pairwiseDistances < raftRadius * ccDistanceMin)
            raftsToRelocate = np.unique(raftsToRelocate)

    elif initialPositionMethod == 2 or (initialPositionMethod == 3 and firstSpinSpeedFlag == 1):
        raftLocations[0, currentStepNum, :] = np.array([arenaSize / 2 + ccSeparationStarting / 2, arenaSize / 2])
        raftLocations[1, currentStepNum, :] = np.array([arenaSize / 2 - ccSeparationStarting / 2, arenaSize / 2])
        firstSpinSpeedFlag = 0
    elif initialPositionMethod == 3 and firstSpinSpeedFlag == 0:
        raftLocations[0, currentStepNum, :] = lastPositionOfPreviousSpinSpeeds[0, :]
        raftLocations[1, currentStepNum, :] = lastPositionOfPreviousSpinSpeeds[1, :]

    raftOrientations[:, currentStepNum] = initialOrientation
    raftRotationSpeedsInRad[:, currentStepNum] = omegaBField

    outputFilename = 'Simulation_' + solverMethod + '_' + str(numOfRafts) + 'Rafts_' \
                     + str(magneticFieldRotationRPS).zfill(3) + 'rps_B' + str(magneticFieldStrength) \
                     + 'T_m' + str(magneticMomentOfOneRaft) + 'Am2_capPeak' + str(capillaryPeakOffset) \
                     + '_edgeSmooth' + str(nearEdgeSmoothingThres) + '_torqueNoise' + str(stdOfTorqueNoise) \
                     + '_lubEqThres' + str(lubEqThreshold) + '_timeStep' + str(timeStepSize) + '_' \
                     + str(timeTotal) + 's'

    if outputVideo == 1:
        outputVideoName = outputFilename + '.mp4'
        fourcc = cv.VideoWriter_fourcc(*'DIVX')
        frameW, frameH, _ = blankFrameBGR.shape
        videoOut = cv.VideoWriter(outputVideoName, fourcc, outputFrameRate, (frameH, frameW), 1)

    for currentStepNum in progressbar.progressbar(np.arange(0, numOfTimeSteps - 1)):
        # currentStepNum = 0
        # looping over raft i,

        magneticFieldDirection = (magneticFieldRotationRPS * 360 * currentStepNum * timeStepSize) % 360
        # modulo operation converts angles into [0, 360)

        raftLocationsOrientations = np.concatenate((raftLocations[:, currentStepNum, :].flatten(),
                                                    raftOrientations[:, currentStepNum]))

        sol = solve_ivp(funct_drdt_dalphadt, (0, timeStepSize), raftLocationsOrientations, method=solverMethod)

        # sol.y[np.logical_and((-sol.y < eps), (-sol.y > 0))] = 0

        raftLocations[:, currentStepNum + 1, :] = sol.y[0:numOfRafts * 2, -1].reshape(numOfRafts, 2)
        raftOrientations[:, currentStepNum + 1] = sol.y[numOfRafts * 2: numOfRafts * 3, -1]

        # draw for current frame
        if (outputImageSeq == 1 or outputVideo == 1) and (currentStepNum % intervalBetweenFrames == 0):
            currentFrameBGR = draw_rafts_rh_coord(blankFrameBGR.copy(),
                                                  np.int64(raftLocations[:, currentStepNum, :] / scaleBar),
                                                  np.int64(raftRadii / scaleBar), numOfRafts)
            currentFrameBGR = draw_b_field_in_rh_coord(currentFrameBGR, magneticFieldDirection)
            currentFrameBGR = draw_cap_peaks_rh_coord(currentFrameBGR,
                                                      np.int64(raftLocations[:, currentStepNum, :] / scaleBar),
                                                      raftOrientations[:, currentStepNum], 6, capillaryPeakOffset,
                                                      np.int64(raftRadii / scaleBar), numOfRafts)
            currentFrameBGR = draw_raft_orientations_rh_coord(currentFrameBGR,
                                                              np.int64(raftLocations[:, currentStepNum, :] / scaleBar),
                                                              raftOrientations[:, currentStepNum],
                                                              np.int64(raftRadii / scaleBar), numOfRafts)
            currentFrameBGR = draw_raft_num_rh_coord(currentFrameBGR,
                                                     np.int64(raftLocations[:, currentStepNum, :] / scaleBar),
                                                     numOfRafts)

            vector1To2SingleFrame = raftLocations[1, currentStepNum, :] - raftLocations[0, currentStepNum, :]
            distanceSingleFrame = np.sqrt(vector1To2SingleFrame[0] ** 2 + vector1To2SingleFrame[1] ** 2)
            phase1To2SingleFrame = np.arctan2(vector1To2SingleFrame[1], vector1To2SingleFrame[0]) * 180 / np.pi
            currentFrameBGR = draw_frame_info(currentFrameBGR, currentStepNum, distanceSingleFrame,
                                              raftOrientations[0, currentStepNum], magneticFieldDirection,
                                              raftRelativeOrientationInDeg[0, 1, currentStepNum])

            if outputImageSeq == 1:
                outputImageName = outputFilename + '_' + str(currentStepNum + 1).zfill(7) + '.jpg'
                cv.imwrite(outputImageName, currentFrameBGR)
            if outputVideo == 1:
                videoOut.write(np.uint8(currentFrameBGR))

    #        if distanceSingleFrame > 950:
    #            break

    if outputVideo == 1:
        videoOut.release()

    tempShelf = shelve.open(outputFilename)
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

#    lastPositionOfPreviousSpinSpeeds[:,:] = raftLocations[:,currentStepNum,:]
#    lastOmegaOfPreviousSpinSpeeds[:] = raftRotationSpeedsInRad[:, currentStepNum]


