import numpy as np
import time
from numba import njit

''''
this script implements the Lambert-Battin algorithm
input var: r1, r2, t, N, v1, mu
output var: v1, v2

variable naming:
r1M, r2M - square root of sum of all squares for r1 and r2 (unit matrix)
r1dr2 - dot product of r1 and r2
r1cvm1 - cross product of r1 and vm1
r1cr2 - cross product of r1 and r2
r1cvm1dr1cr2 - dot product of r1cvm1 and r1cr2
dth - transfer angle
'''

@njit
def battin(r1, vm1, r2, t, mu):
    # define the tolerances of the solution
    iter_tol1 = 1.0e-6
    iter_tol2 = 1.0e-12

    # normalize the constants for faster processing speed
    # du = np.linalg.norm(r1)
    # vu = np.sqrt(mu / du)
    # tu = du / vu
    #
    # mu = mu * tu**2 / du**3
    # r1 = np.divide(r1, du)
    # vm1 = np.divide(vm1, vu)
    # r2 = np.divide(r2, du)
    # t = t / tu

    # perform calculations on with input
    r1M = np.linalg.norm(r1)
    r2M = np.linalg.norm(r2)
    r1dr2 = np.dot(r1, r2)
    r1cvm1 = np.cross(r1, vm1)
    r1cr2 = np.cross(r1, r2)
    r1cvm1dr1cr2 = np.dot(r1cvm1, r1cr2)

    A = r1dr2 / (r1M * r2M)

    # exception handle
    if A > 1.0:
        A = 1.0

    elif A < -1.0:
        A = -1.0

    # find the transfer angle (dth)
    dth = np.arccos(A)
    
    # determine if this should be long or short way, long way fulfills the logic condition below
    if r1cvm1dr1cr2 < 0.0:
        dth = 2.0 * np.pi - dth

    # define more parameters for the algorithm
    c = np.linalg.norm(r2 - r1)
    s = 0.5 * (r1M + r2M + c)
    lamda = (1.0 / s) * np.sqrt(r1M * r2M) * np.cos(dth * 0.5)
    L = ((1.0 - lamda) / (1.0 + lamda))**2
    m = 8.0 * mu * t**2 / (s**3 * (1.0 + lamda)**6)
    
    # determine convergence for the low energy or 0 rev solution
    xdiff = 1.0 # initialize convergence value
    x = L # initial guess

    # loop for convergence
    while xdiff > iter_tol1:

        # perform hypergeometric function using top down method
        eta = x / (np.sqrt(1.0 + x) + 1.0)**2
        bn = 3.0
        dn = 1.0
        un = 8.0 * (np.sqrt(1.0 + x) + 1.0) / bn
        hgx = un
        i = 1
        counter = 0
        while np.abs(un) > iter_tol2:
            bp = bn
            dp = dn
            up = un
            i += 1
               
            if i == 2:
                an = -1.0
                bn = 5.0 + eta

            elif i == 3:
                an = -9.0 * eta / 7.0
                bn = 1.0

            else:
                i2 = float(i**2)
                an = -eta * i2 / (4.0 * i2 - 1.0)
                bn = 1.0

            dn = 1.0 / (1.0 - an * dp / bp / bn)
            un = up * (dn - 1.0)
            hgx += un
            counter += 1

        # find h1 and h2
        denom = 1.0 / ((1.0 + 2.0 * x + L) * (4.0 * x + hgx * (3.0 + x)))
        h1 = (L + x)**2 * (1.0 + 3.0 * x + hgx) * denom
        h2 = m * (x - L + hgx) * denom
        
        # root of the cubic function
        B = 27.0 * 0.25 * h2 / (1.0 + h1)**3
        u = 0.5 * B / (np.sqrt(1.0 + B) + 1.0)
        bn = 1.0
        dn = 1.0
        un = 1.0 / 3.0
        K = un
        nn = 0.0
        evenflag = True

        while np.abs(un) > iter_tol2:
            bp = bn
            dp = dn
            up = un
            
            if evenflag:
                an = -2.0 * u * (3.0 * nn +2.0) * (6.0 * nn + 1.0) / (9.0 * (4.0 * nn + 1.0) * (4.0 * nn + 3.0))
                evenflag = False
                nn += 1.0

            else:
                an = -2.0 * u * (3.0 * nn + 1.0) * (6.0 * nn - 1.0) / (9.0 * (4.0 * nn - 1.0) * (4.0 * nn + 1.0))
                evenflag = True
            
            dn = 1.0 / (1.0 - an * dp / bp / bn)
            un = up * (dn - 1.0)
            K += un
        
        y = (1.0 + h1) / 3.0 * (2.0 + np.sqrt(1.0 + B) / (1.0 + 2.0 * u * K**2))
        xnew = np.sqrt(((1.0 - L) * 0.5)**2 + m / (y**2)) - (1.0 + L) * 0.5
        xdiff = np.abs(xnew - x)
        x = xnew

    # solution parameter
    p = 2.0 * r1M * r2M * y**2 * (1.0 + x)**2 * np.sin(dth * 0.5)**2 / (m * s * (1.0 + lamda)**2)
    epilson = (r2M - r1M) / r1M
    e = np.sqrt((epilson**2 + 4.0 * r2M / r1M * np.sin(dth * 0.5)**2 * ((L - x) / (L + x))**2) \
                / (epilson**2 + 4.0 * r2M / r1M * np.sin(dth * 0.5)**2))
    
    # find the velocity using Hodograph technique
    L180 = 0.001 # [km] 1.0 meter in kilometers
    
    A = mu * (1.0 / r1M - 1.0 / p)
    B = (mu * e / p)**2 - A**2

    if B <= 0.0:
        x1 = 0.0

    else:
        x1 = -np.sqrt(B)

    if np.abs(np.sin(dth)) < L180 / r2M:
        vecC = np.cross(r1, vm1)
        vecCM = np.linalg.norm(vecC)
        
        nH = vecC / vecCM
        
        if e < 1.0:
            period = 2.0 * np.pi * np.sqrt(p**3 / (mu * (1.0 - e**2)**3))
            
            if np.mod(t, period) > np.pi:
                x1 *= -1
    else:
        vecC = np.cross(r1, r2)
        vecCM = np.linalg.norm(vecC)
        
        nH = vecC / vecCM
        
        if np.mod(dth, 2.0 * np.pi) > np.pi:
            nH *= -1
        
        y2a =mu / p - x1 * np.sin(dth) + A * np.cos(dth)
        y2b = mu / p + x1 * np.sin(dth) + A * np.cos(dth)
        
        if np.abs(mu / r2M - y2b) < np.abs(mu / r2M - y2a):
            x1 *= -1
    
    nHcr1 = np.cross(nH, r1)
    nHcr2 = np.cross(nH, r2)

    v1 = (np.sqrt(mu * p) / r1M) * (x1 / mu * r1 + nHcr1 / r1M)
    x2 = x1 * np.cos(dth) + A * np.sin(dth)
    v2 = (np.sqrt(mu * p) / r2M) * (x2 / mu * r2 + nHcr2 / r2M)

    # part of the normalization
    # v1 = np.multiply(v1, vu)
    # v2 = np.multiply(v2, vu)

    return v1, v2


if __name__ == '__main__':
    # test case
    r1 = np.array([3189068.15, 3826881.78, 4464695.41])
    vm1 = np.array([-2859.0, 6085.0, -4002.0])
    r2 = np.array([-11.0, 6378136.0, -16.0])
    mu = 3.986004415e14
    t = 780.0
    
    # Calls the function to test
    startTime = time.time()
    v1, v2 = battin(r1, vm1, r2, t, mu)
    endTime = time.time()
    
    # print output
    # m/s unit
    print(v1)
    print(v2)
    print(endTime - startTime)
