"""
This file is to calculate/estimate the scaling relation of magnetic, capillary, and hydrodynamic lift force vs. radius
"""

import numpy as np
import matplotlib.pyplot as plt
import progressbar
import scipy.io
import os

#%% capillary forces
if os.getcwd() == 'D:\\repos\\spinning_rafts_sim2\\':
    os.chdir('2020-09-08_capillaryForceCalculations-sym6_radiusScaling')
result = scipy.io.loadmat('Results_sym6_arcAngle30_eeDistAt1R_angleCount60_radiusCount46_treated')
capForceAngleAveraged = result['angleAveragedForceSqueezed']  # unit: nN
radiiForCapForce = result['radii']  # unit: micron
eeDistInR = result['eeDistInR'].item()  # unit: radius
# convert to SI unit and reshape
capForceAngleAveragedAteeR = capForceAngleAveraged * 1e-9  # unit: N, only at eeDist = radius/2
radiiForCapForceAteeR = (radiiForCapForce * 1e-6).reshape(capForceAngleAveragedAteeR.shape)  # unit: m


#%% magnetic and hydrodynamic lift force calculated at specified eeDist in unit of R
# eeDistInR = 0.5  # unit: radius
miu0 = 4 * np.pi * 1e-7  # unit: N/A**2, or T.m/A, or H/m
magneticMomentPerUnitArea = 1e-8 / (np.pi * (150 * 1e-6) ** 2)  # unit: A (A.m**2/m**2)
R = radiiForCapForceAteeR  # unit: m. old: np.arange(1, 10001) / 1e6
densityOfWater = 1e3  # unit: 1000 kg/m^3
# magDpForceOnAxisAngleAveraged_halfR = np.zeros(len(R))  # unit: N
# hydroLiftForceAngleAveraged_1rps_halfR = np.zeros(len(R))  # unit: N
# hydroLiftForceAngleAveraged_10rps_halfR = np.zeros(len(R))  # unit: N
# hydroLiftForceAngleAveraged_20rps_halfR = np.zeros(len(R))  # unit: N

magDpForceOnAxisAngleAveraged_eeR = -(3 / 4) * miu0 * (np.pi * R ** 2 * magneticMomentPerUnitArea) ** 2 / \
                                      ((2 + eeDistInR) * R) ** 4
hydroLiftForceAngleAveraged_1rps_eeR = densityOfWater * (1 * 2 * np.pi) ** 2 * R ** 7 / ((2 + eeDistInR) * R) ** 3
hydroLiftForceAngleAveraged_10rps_eeR = densityOfWater * (10 * 2 * np.pi) ** 2 * R ** 7 / ((2 + eeDistInR) * R) ** 3
hydroLiftForceAngleAveraged_20rps_eeR = densityOfWater * (20 * 2 * np.pi) ** 2 * R ** 7 / ((2 + eeDistInR) * R) ** 3
hydroLiftForceAngleAveraged_70rps_eeR = densityOfWater * (70 * 2 * np.pi) ** 2 * R ** 7 / ((2 + eeDistInR) * R) ** 3

# plotting of forces vs radius range with eeDistance at 0.5R
fig, ax = plt.subplots(ncols=1, nrows=1)
ax.loglog(R * 1e3, -magDpForceOnAxisAngleAveraged_eeR, color='blue',
          label='angle-averaged d-d mag force')
# ax.loglog(R * 1e3, -magDpForceOnAxisAngleAveraged_halfR/10, color='blue',
#           label='angle-averaged d-d mag force')
ax.loglog(R * 1e3, hydroLiftForceAngleAveraged_1rps_eeR, color='orange',
          label='angle-averaged hydro lift force 1rps')
ax.loglog(R * 1e3, hydroLiftForceAngleAveraged_10rps_eeR, color='orange',
          label='angle-averaged hydro lift force 10rps')
# ax.loglog(R * 1e3, hydroLiftForceAngleAveraged_20rps_halfR, color='orange',
#           label='angle-averaged hydro lift force 20rps')
ax.loglog(R * 1e3, hydroLiftForceAngleAveraged_70rps_eeR, color='orange',
          label='angle-averaged hydro lift force 70rps')
ax.loglog(radiiForCapForceAteeR * 1e3, capForceAngleAveragedAteeR, color='green',
          label='angle-averaged capillary force')
ax.set_xlabel('radius of raft (mm)', size=20)
ax.set_ylabel('force (N)', size=20)
ax.set_title('force scaling at eeDist = {}R'.format(eeDistInR))
ax.legend()
plt.show()
figName = 'force scaling at eeDist = {}R.png'.format(eeDistInR)
fig.savefig(figName)

# save data in a csv file
radiiCapMagHydroForces = np.concatenate((R, capForceAngleAveragedAteeR, -magDpForceOnAxisAngleAveraged_eeR,
                                         hydroLiftForceAngleAveraged_1rps_eeR,
                                         hydroLiftForceAngleAveraged_10rps_eeR,
                                         hydroLiftForceAngleAveraged_70rps_eeR), axis=1)
filename = 'force scaling at eeDist = {}R.csv'.format(eeDistInR)
np.savetxt(filename, radiiCapMagHydroForces, delimiter=',',
           header='R(m), cap(N), magDp(N), hydro_1rps(N), hydro_10rps(N), hydro_70rps(N)')

#%% magnetic and hydrodynamic lift force calculation based on fixed range of ee-distances
miu0 = 4 * np.pi * 1e-7  # unit: N/A**2, or T.m/A, or H/m
magneticMomentPerUnitArea = 1e-8 / (np.pi * (150 * 1e-6) ** 2)  # unit: A (A.m**2/m**2)
radiiOfRaft = np.arange(1, 10001) / 1e6  # unit: m
eeDists = np.arange(0, 52) / 1e6  # unit: m
magDpForceOnAxisAngleAveraged = np.zeros((len(radiiOfRaft), len(eeDists)))  # unit: N
densityOfWater = 1e3  # unit: 1000 kg/m^3
hydroLiftForceAngleAveraged_1rps = np.zeros((len(radiiOfRaft), len(eeDists)))  # unit: N
hydroLiftForceAngleAveraged_10rps = np.zeros((len(radiiOfRaft), len(eeDists)))  # unit: N
hydroLiftForceAngleAveraged_20rps = np.zeros((len(radiiOfRaft), len(eeDists)))  # unit: N

for index, R in enumerate(radiiOfRaft):
    magDpForceOnAxisAngleAveraged[index, :] = -(3/4) * miu0 * (np.pi * R**2 * magneticMomentPerUnitArea)**2 / \
                                              (eeDists + 2*R)**4
    hydroLiftForceAngleAveraged_1rps[index, :] = densityOfWater * (1 * 2 * np.pi)**2 * R**7 / (eeDists + 2*R)**3
    hydroLiftForceAngleAveraged_10rps[index, :] = densityOfWater * (10 * 2 * np.pi)**2 * R**7 / (eeDists + 2*R)**3
    hydroLiftForceAngleAveraged_20rps[index, :] = densityOfWater * (20 * 2 * np.pi)**2 * R**7 / (eeDists + 2*R)**3


# plotting at a specific distance value
eeDistIndex = 51  # 50 micron eeDist
fig, ax = plt.subplots(ncols=1, nrows=1)
ax.loglog(radiiOfRaft * 1e3, -magDpForceOnAxisAngleAveraged[:, eeDistIndex], color='blue',
        label='angle-averaged d-d mag force')
ax.loglog(radiiOfRaft * 1e3, hydroLiftForceAngleAveraged_1rps[:, eeDistIndex], color='orange',
        label='angle-averaged hydro lift force 1rps')
ax.loglog(radiiOfRaft * 1e3, hydroLiftForceAngleAveraged_10rps[:, eeDistIndex], color='orange',
        label='angle-averaged hydro lift force 10rps')
# ax.semilogx(radiiOfRaft * 1e3, hydroLiftForceAngleAveraged_20rps[:, eeDistIndex], color='orange',
#         label='angle-averaged hydro lift force 20rps')
ax.set_xlabel('radius of raft (mm)', size=20)
ax.set_ylabel('force (N)', size=20)
ax.set_title('force scaling at eeDist = {} m'.format((eeDistIndex-1)/1e6))
# ax.set_ylim([-0.000001, 0.0001])
ax.legend()
plt.show()
# figName = 'force scaling relations'
# fig.savefig(figName)

