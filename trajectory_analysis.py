import numpy as np
import lambert
import planet_data
import pandas as pd

def trajectory_calc(planet1: str, planet2: str):
    mu = 1.32712440020e11 # constant mu for the sun

    planet1_dt, planet1_jd, planet1_r, planet1_v = planet_data.read_planet_files(planet1)
    planet2_dt, planet2_jd, planet2_r, planet2_v = planet_data.read_planet_files(planet2)

    # matrix of cost function
    dv = [[0.0] * len(planet1_dt) for i in range(len(planet2_dt))]
    tof = [[0.0] * len(planet1_dt) for i in range(len(planet2_dt))]


    for i in range(len(planet1_jd)):
        for j in range(len(planet2_jd)):

            # time of flight to check validity
            tof[i][j] = planet2_jd[j] - planet1_jd[i]

            # perform calculation to check for flight angle limitations
            r1M = np.linalg.norm(planet1_r[i])
            r2M = np.linalg.norm(planet2_r[j])
            r1dr2 = np.dot(planet1_r[i], planet2_r[j])

            A = r1dr2 / (r1M * r2M)

            if A > 1.0:
                A = 1.0

            elif A < -1.0:
                A = -1.0

            # find the transfer angle (dth)
            dth = np.arccos(A)

            # angle of transfer restrictions and 60 days of tolerance
            if 3.0 < dth * (180.0 / np.pi) < 358.0 and tof[i][j] > 60.0:

                # time of flight convert to seconds
                v1, v2 = lambert.battin(planet1_r[i], planet1_v[i], planet2_r[j], tof[i][j] * 86400.0, mu)

                # cost function
                dv[i][j] = np.linalg.norm(abs(v1 - planet1_v[i]) + abs(v2 - planet2_v[j]))


    # convert to dataframe to store, including arrival date as header
    df_tof = pd.DataFrame(tof,  columns=planet2_dt)
    df_dv = pd.DataFrame(dv, columns=planet2_dt)

    # add departure date
    df_tof['departure'] = planet1_dt
    df_dv['departure'] = planet1_dt

    # save the csv
    PATH = './data'
    df_tof.to_csv(PATH + '/tof.csv', header=True, index=False)
    df_dv.to_csv(PATH + '/dv.csv', header=True, index=False)


if __name__ == '__main__':
    print(trajectory_calc('earth', 'mars'))