# -*- coding: utf-8 -*-
"""
Decoding

Akhmad Sultoni
"""

import numpy as np
import time
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Memuat file .npz
data = np.load('forest.npz')

# Mengakses variabel di dalam file .npz
sv=data['sv']
rsize = data['rsize']
T = data['T']
waktu = data['waktu']
lp = data['lp']

M = 100 * np.ones((sv, sv))  # Matrix M yang diinisialisasi dengan 100
itr_d_1 = 0
itr_d_2 = 0

# Mulai penghitungan waktu
start_time = time.time()

# Fungsi rotasi, flip horizontal, flip vertikal
def rotmat(matrix):
    return np.rot90(matrix)

def fliph(matrix):
    return np.fliplr(matrix)

def flipv(matrix):
    return np.flipud(matrix)

# Simulasi kode
for iter in range(15):
    rsize = 16
    nd = int(sv // rsize // 2)
    nr = int(sv // rsize)
    M1 = np.zeros((rsize * nd, rsize * nd))  # Inisialisasi M1
    for i in range(1,rsize * nd+1):
        for j in range(1,rsize * nd+1):
            i1 = (i - 1) * 2
            i2 = i * 2
            j1 = (j - 1) * 2
            j2 = j * 2
            M1[i-1, j-1] = np.mean(M[i1:i2, j1:j2])
            itr_d_1 += 1

    MM = np.zeros_like(M)  # Inisialisasi MM
    for k in range(1,nr+1):
        k1 = int((k - 1) * rsize)
        k2 = int(k * rsize)
        for l in range(1,nr+1):
            itr_d_2 += 1
            l1 = int((l - 1) * rsize)
            l2 = int(l * rsize)
            i0, j0, m0, s0, g0 = T[k-1, l-1, :]
            i1 = int((i0 - 1) * rsize)
            i2 = int(i0 * rsize)
            j1 = int((j0 - 1) * rsize)
            j2 = int(j0 * rsize)
            D = M1[i1:i2, j1:j2]
            if m0 == 2:
                D = rotmat(D)
            elif m0 == 3:
                D = rotmat(rotmat(D))
            elif m0 == 4:
                D = rotmat(rotmat(rotmat(D)))
            elif m0 == 5:
                D = fliph(D)
            elif m0 == 6:
                D = flipv(D)
            elif m0 == 7:
                D = D.T
            elif m0 == 8:
                D = rotmat(rotmat(D.T))

            R = s0 * D + g0 * np.ones_like(D)
            MM[k1:k2, l1:l2] = R

    M = MM

# Waktu selesai
waktu_running = time.time() - start_time

# Menghitung total iterasi
itr_d = itr_d_1 + itr_d_2

# Menampilkan gambar menggunakan Matplotlib
plt.imshow(M, cmap='gray')
plt.axis('off')  # Menyembunyikan axis
plt.show()

# Load gambar dari file .tif
M = cv2.imread('C:/Users/LENOVO/fic-uji-coba/forest_256x256_grayscale.png', cv2.IMREAD_GRAYSCALE)

# Mengubah ukuran gambar menjadi 256x256
#M = cv2.resize(M, (256, 256))

# Mengonversi gambar ke double

image = np.double(M)

# Melakukan operasi perhitungan dengan MM
K = image - MM
#print(K)
KK = 0
ZZ = 0

# Hitung MSE, RMSE, dan PSNR
for i in range(1,sv+1):
    for j in range(1,sv+1):
        Z = (K[i-1, j-1]) ** 2
        Z1 = abs(K[i-1, j-1])
        KK += Z
        ZZ += Z1

ape = ZZ / (sv * sv)
mse = KK / (sv * sv)
rmse = np.sqrt(mse)
PSNR = 20 * np.log10(255 / rmse)

# Menampilkan hasil PSNR
print("PSNR:", PSNR)
# Menampilkan gambar menggunakan Matplotlib
plt.imshow(M, cmap='gray')
plt.axis('off')  # Menyembunyikan axis
plt.show()