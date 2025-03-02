# -*- coding: utf-8 -*-
"""
Encoding

Akhmad Sultoni
"""

import time
import cv2
import numpy as np

rsize=int(input('Masukan ukuran range: '))
start_time=time.time()
itr_1=0
itr_2=0
itr_3=0
image_url = "C:/Users/LENOVO/fic-uji-coba/forest_256x256_grayscale.png"
M=cv2.imread(image_url,cv2.IMREAD_GRAYSCALE)
sv,sh=M.shape

min0=100
#rsize=8
nd=int((sv/rsize)/2)
nr=int(sv/rsize)
M1 = np.zeros((rsize * nd, rsize * nd))
for i in range(1,int(nd)*rsize+1):
  for j in range(1,int(nd)*rsize+1):
    M1[i-1,j-1]=np.mean(M[(i-1)*2:i*2, (j-1)*2:j*2])
    itr_1+=1
s=[0.45,0.60,0.80,1]

# Misalnya, M1 adalah array NumPy yang sudah didefinisikan sebelumnya
# rsize dan nd sudah didefinisikan sebelumnya
# bigM adalah array kosong yang berukuran sesuai kebutuhan
bigM = np.zeros((rsize * nd, rsize * nd, 8))  # Ukuran sesuai dengan bigM yang digunakan

# Fungsi rotmat dan fliph, flipv bisa diimplementasikan sesuai kebutuhan
def rotmat(matrix):
    return np.rot90(matrix)  # Rotasi 90 derajat

def fliph(matrix):
    return np.fliplr(matrix)  # Flip horizontal

def flipv(matrix):
    return np.flipud(matrix)  # Flip vertikal

itr_2 = 0

for i in range(1, nd + 1):  # Python menggunakan indeks mulai dari 0, namun MATLAB mulai dari 1
    i1 = (i - 1) * rsize
    i2 = i * rsize
    for j in range(1, nd + 1):
        j1 = (j - 1) * rsize
        j2 = j * rsize
        
        # Mengambil blok D dari M1
        D = M1[i1:i2, j1:j2]
        
        # Menyimpan ke dalam bigM
        bigM[i1:i2, j1:j2, 0] = D
        tmp = rotmat(D)
        bigM[i1:i2, j1:j2, 1] = tmp
        tmp = rotmat(tmp)
        bigM[i1:i2, j1:j2, 2] = tmp
        tmp = rotmat(tmp)
        bigM[i1:i2, j1:j2, 3] = tmp
        bigM[i1:i2, j1:j2, 4] = fliph(D)
        bigM[i1:i2, j1:j2, 5] = flipv(D)
        bigM[i1:i2, j1:j2, 6] = D.T  # Transpose D
        bigM[i1:i2, j1:j2, 7] = rotmat(rotmat(D.T))  # Rotasi dua kali setelah transpose
        
        itr_2 += 1

# Misalnya M, bigM, dan s sudah didefinisikan sebelumnya, serta variabel lainnya (nr, rsize, nd)
T = np.zeros((nr, nr, 5))  # Array kosong untuk menyimpan hasil

for k in range(1, nr + 1):  # Indeks dimulai dari 1, karena di MATLAB mulai dari 1
    k1 = (k - 1) * rsize
    k2 = k * rsize
    for l in range(1, nr + 1):
        l1 = (l - 1) * rsize
        l2 = l * rsize
        R = M[k1:k2, l1:l2]
        o = np.mean(R)
        R = R.astype(float)
        dmin = 10**9
        i0, j0, m0, s0, g0 = 0, 0, 0, 0, 0
        
        for i in range(1, nd + 1):
            i1 = (i - 1) * rsize
            i2 = i * rsize
            for j in range(1, nd + 1):
                j1 = (j - 1) * rsize
                j2 = j * rsize
                for n in range(1, 5):  # n runs from 1 to 4
                    for m in range(1, 9):  # m runs from 1 to 8
                        D = s[n-1] * bigM[i1:i2, j1:j2, m-1]  # Adjust indices to Python's 0-based index
                        del_g = o - np.mean(D)
                        D = D + del_g
                        sum_dist = np.sum((R - D) ** 2)
                        dist = np.sqrt(sum_dist)
                        
                        if dist < dmin:
                            dmin = dist
                            i0, j0, m0, s0, g0 = i, j, m, s[n-1], del_g  # Store results
                        
                        itr_3 += 1  # Assuming itr_3 is defined earlier

        # Assign the final results for T
        T[k-1, l-1, :] = [i0, j0, m0, s0, g0]  # Adjust for Python's 0-based indexing


elapsed_time = time.time() - start_time
# Menghitung lp
lp = itr_1 + itr_2 + itr_3

# Menyimpan variabel ke file
np.savez('forest.npz', sv=sv, rsize=rsize, T=T, waktu=elapsed_time , lp=lp)
print(f'Waktu yang diperlukan: {elapsed_time:.5f} detik')