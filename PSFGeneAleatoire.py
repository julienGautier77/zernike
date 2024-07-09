#! /home/zita1/miniforge3/bin/python3.10
import LightPipes as li
import matplotlib.pyplot as plt
import math
import numpy as np
from PIL import Image
import time
import random
wavelength = 820*li.nm
size = 80*li.mm
N = 4000
i = 0
x = np.linspace(-size/2, size/2, N) / li.mm
y = x
labels = np.zeros(12)

for i in range(0, 2000):
    coef = np.round(np.random.uniform(-0.2,0.2,12),2)
    #print(coef)
    t =time.time() 
    coef[0]=0
    coef[1]= 0
    coef[2]=0
    coef[3]= 0
    coef[4]= 0
    F = li.Begin(size, wavelength, N)
    Z = li.CircAperture(size/4, 0, 0, F)
    (nz, mz) = li.noll_to_zern(5)
    Z = li.Zernike(Z, nz, mz, size/4, coef[5], units='lam')
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
    
    L = li.Lens(Z, 1000*li.mm)
    prog = li.Forvard(L, 1000*li.mm)  #Forvard
    I2 = li.Intensity(prog)
    I2 = I2[1980:2020, 1980:2020]
    img_PIL = Image.fromarray(I2)
    # fig2=plt.figure()
    # ax3= fig2.add_subplot(121)

    # ax3.imshow(I2,cmap='jet')
    # plt.show()
    if i < 10:
        num = "0000"+ str(i)
    elif 9<i<100:
        num = "000" + str(i)
    elif 99<i<1000:
        num = "00" + str(i)
    elif 999<i<10000:
        num = "0" + str(i)
    else:
        num = "0" + str(i)
    file='dataTest/'+ 'img'+num+ '_'
    file =file + str(round(coef[5], 2))+'_'+str(round(coef[6],2))+'_'+str(round(coef[7],2))+'_'+str(round(coef[8],2))
    file = file + '_'+str(round(coef[9],2))+'_'+str(round(coef[10],2))+'_'+str(round(coef[11],2))

    img_PIL.save(file + '.TIFF', format='TIFF')
    
    labels=np.vstack((labels,coef))
    print('data',i,time.time()-t,'s',coef) 
labels=np.delete(labels,0,0)
np.savetxt('dataTest/lab.txt',labels,newline='\n')