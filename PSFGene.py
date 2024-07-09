#! /home/zita1/miniforge3/bin/python3.10
import LightPipes as li
import matplotlib.pyplot as plt
import math
import numpy as np
from PIL import Image
import time
wavelength = 820*li.nm
size = 80*li.mm
N = 4000
print(size)
i = 0
x = np.linspace(-size/2, size/2, N) / li.mm
y = x
labels = []
coef = np.zeros(12)
for coef[5] in np.arange(0, 1, 0.1):
    for coef[6] in np.arange(0, 1, 0.1):
        for coef[7] in np.arange(0, 1, 0.1):
            for coef[8] in np.arange(0, 1, 0.1):
                for coef[9] in np.arange(0, 1, 0.1):
                    for coef[10] in np.arange(0, 1, 0.1):
                        for coef[11] in np.arange(0, 1, 0.1):
                            t =time.time() 
                            F = li.Begin(size, wavelength, N)
                            (nz, mz) = li.noll_to_zern(5)
                            Z = li.Zernike(F, nz, mz, size/4, coef[5], units='lam')
                            (nz, mz) = li.noll_to_zern(6)
                            Z = li.Zernike(Z, nz, mz, size/4, coef[6], units='lam')
                            (nz, mz) = li.noll_to_zern(7)
                            Z = li.Zernike(Z, nz, mz, size/4, coef[7], units='lam')
                            (nz, mz) = li.noll_to_zern(8)
                            Z = li.Zernike(Z, nz, mz, size/4, coef[8], units='lam')
                            (nz, mz) = li.noll_to_zern(9)
                            Z = li.Zernike(Z, nz, mz, size/4, coef[9], units='lam')
                            (nz, mz) = li.noll_to_zern(10)
                            Z = li.Zernike(Z, nz, mz, size/4, coef[10], units='lam')
                            (nz, mz) = li.noll_to_zern(11)
                            Z = li.Zernike(Z, nz, mz, size/4, coef[11], units='lam')
                            Z = li.CircAperture(size/4, 0, 0, Z)
                            L = li.Lens(Z, 1000*li.mm)
                            prog = li.Forvard(L, 1000*li.mm)  #Forvard
                            I2 = li.Intensity(prog)
                            I2 = I2[1985:2015, 1985:2015]
                            img_PIL = Image.fromarray(I2)
                            file = str(round(coef[5], 2))+'_'+str(round(coef[6]))+'_'+str(round(coef[7]))+'_'+str(round(coef[8]))
                            file = file + '_'+str(round(coef[9]))+'_'+str(round(coef[10]))+'_'+str(round(coef[11]))
                            img_PIL.save(file + '.TIFF', format='TIFF')
                            Labels = labels.append([coef])
                            i += 1
                            print(i,time.time()-t) 