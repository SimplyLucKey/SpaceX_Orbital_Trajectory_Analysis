'''
data is sandwiched between $$SOE and $$EOE
two sets of data: mars and earth

critical variables: julian date, X, Y, Z, VX, VY, VZ
---
2462502.500000000 = A.D. 2030-Jan-01 00:00:00.0000 TDB
 X =-2.592636728814095E+07 Y = 1.448368869091996E+08 Z =-3.787071235872805E+03
 VX=-2.982040565319705E+01 VY=-5.364759719314518E+00 VZ=-4.789742753545934E-04
 LT= 4.908030443266405E+02 RG= 1.471390510525665E+08 RR=-2.637168904908037E-02
'''
import numpy as np
from datetime import datetime
import pprint


def read_planet_files(planet: str):
    PATH = './data'

    with open(f'{PATH}/{planet}_data.txt', 'r') as planet_file:
        dt, jd, r, v = [], [], [], [] # lists that will contain julian date, position vector, and velocity vector
        planet_list = planet_file.readlines() # set file to a list to be indexed


        for idx, line in enumerate(planet_list):

            # skip lines that do not have julian date
            try:
                if not line[0].isdigit():
                    raise Exception

                # add relevant variables to dictionary
                # dt.append(datetime.strptime(planet_list[idx][25:36], '%Y-%b-%d'))
                dt.append(planet_list[idx][25:36])

                jd.append(float(planet_list[idx][0:18]))

                r.append((np.array([float(planet_list[idx + 1][4:26]),
                          float(planet_list[idx + 1][31:52]),
                          float(planet_list[idx + 1][56:78])])))

                v.append((np.array([float(planet_list[idx + 2][4:26]),
                          float(planet_list[idx + 2][31:52]),
                          float(planet_list[idx + 2][56:78])])))

            except Exception:
                pass

    return dt, jd, r, v

if __name__ == '__main__':
    planet1 = 'mars'
    pprint.pprint(read_planet_files('earth'))