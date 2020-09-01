"""
This file is to calculate/estimate the scaling relation of magnetic, capillary, and hydrodynamic lift force vs. radius
"""

import numpy as np
import matplotlib.pyplot as plt
import progressbar

# magnetic force calculation
miu0 = 4 * np.pi * 1e-7  # unit: N/A**2, or T.m/A, or H/m
magneticMomentPerUnitArea = 1e-8 / (np.pi * (300 * 1e-6) ** 2)  # unit: A (A.m**2/m**2)

radiiOfRaft = np.arange(1, 10001) / 1e6  # unit: m

eeDists = np.arange(0, 10001) / 1e6  # unit: m

magDpForceOnAxisAngleAveraged = np.zeros((len(radiiOfRaft), len(eeDists)))  # unit: N

densityOfWater = 1e3  # unit: 1000 kg/m^3
hydroLiftForceAngleAveraged_1rps = np.zeros((len(radiiOfRaft), len(eeDists)))  # unit: N
hydroLiftForceAngleAveraged_10rps = np.zeros((len(radiiOfRaft), len(eeDists)))  # unit: N
hydroLiftForceAngleAveraged_20rps = np.zeros((len(radiiOfRaft), len(eeDists)))  # unit: N

surfaceTension = 72e-3  # unit: J/m^2
amplitude = 1e-6  # unit: m
capForceAngleAvg = np.zeros((len(radiiOfRaft), len(eeDists)))  # unit: N

for index, R in enumerate(radiiOfRaft):
    magDpForceOnAxisAngleAveraged[index, :] = -(3/4) * miu0 * (np.pi * R**2 * magneticMomentPerUnitArea)**2 / \
                                              (eeDists + 2*R)**4
    hydroLiftForceAngleAveraged_1rps[index, :] = densityOfWater * (1 * 2 * np.pi)**2 * R**7 / (eeDists + 2*R)**3
    hydroLiftForceAngleAveraged_10rps[index, :] = densityOfWater * (10 * 2 * np.pi)**2 * R**7 / (eeDists + 2*R)**3
    hydroLiftForceAngleAveraged_20rps[index, :] = densityOfWater * (20 * 2 * np.pi)**2 * R**7 / (eeDists + 2*R)**3

    capForceAngleAvg[index, :] = 2 * np.pi * surfaceTension * amplitude**2 * R**14 / (eeDists + 2*R)**15



# at a specific distance value
eeDistIndex = 11
fig, ax = plt.subplots(ncols=1, nrows=1)
ax.semilogx(radiiOfRaft * 1e3, magDpForceOnAxisAngleAveraged[:, eeDistIndex],
        label='angle-averaged d-d mag force at eeDist = {} m '.format((eeDistIndex-1)/1e6))
ax.semilogx(radiiOfRaft * 1e3, hydroLiftForceAngleAveraged_1rps[:, eeDistIndex],
        label='angle-averaged hydro lift force 1rps at eeDist = {} m '.format((eeDistIndex-1)/1e6))
# ax.semilogx(radiiOfRaft * 1e3, hydroLiftForceAngleAveraged_10rps[:, eeDistIndex],
#         label='angle-averaged hydro lift force 10rps at eeDist = {} m '.format((eeDistIndex-1)/1e6))
# ax.semilogx(radiiOfRaft * 1e3, hydroLiftForceAngleAveraged_20rps[:, eeDistIndex],
#         label='angle-averaged hydro lift force 20rps at eeDist = {} m '.format((eeDistIndex-1)/1e6))
ax.set_xlabel('radius of raft (mm)', size=20)
ax.set_ylabel('force (N)', size=20)
ax.set_title('force scaling')
ax.set_ylim([-0.000001, 0.0001])
ax.legend()
plt.show()
# figName = 'force scaling relations'
# fig.savefig(figName)

