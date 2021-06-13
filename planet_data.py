'''
data is sandwiched between $$SOE and $$EOE
two sets of data: mars and earth

critical variables: julian date, X, Y, Z, VX, VY, VZ
---
2462502.500000000 = A.D. 2030-Jan-01 00:00:00.0000 TDB
 X = 1.279171664035613E+00 Y =-5.209546331561243E-01 Z =-4.223338454005444E-02
 VX= 5.811026046487512E-03 VY= 1.415900514341515E-02 VZ= 1.541843752026991E-04
 LT= 7.980791571489530E-03 RG= 1.381831225095228E+00 RR= 3.660996970503041E-05
'''
import numpy as np


def read_planet_files(planet: str):
    PATH = './data'

    with open(f'{PATH}/{planet}_data.txt', 'r') as planet_file:
        jd, r, v = [], [], [] # lists that will contain julian date, position vector, and velocity vector
        planet_list = planet_file.readlines() # set file to a list to be indexed


        for idx, line in enumerate(planet_list):

            # skip lines that do not have julian date
            try:
                if not line[0].isdigit():
                    raise Exception

                # add relevant variables to dictionary
                jd.append(float(planet_list[idx][0:18]))

                r.append((np.array([float(planet_list[idx + 1][4:26]),
                          float(planet_list[idx + 1][31:52]),
                          float(planet_list[idx + 1][56:78])])))

                v.append((np.array([float(planet_list[idx + 2][4:26]),
                          float(planet_list[idx + 2][31:52]),
                          float(planet_list[idx + 2][56:78])])))

            except Exception:
                pass

    return jd, r, v