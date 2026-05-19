import scipy.io
import numpy as np
from viseda import HyperspectralEDA

# Step 1: load the file
mat = scipy.io.loadmat("C:/Users/Ike/Desktop/testhyper/Indian_pines_corrected.mat")

# Step 2: inspect keys to find the right one
print(mat.keys())

# Step 3: use the correct key (commonly one of these for Indian Pines)
# Try each until one works:
# cube = mat["indian_pines_corrected"].astype(np.float32)
# cube = mat["indian_pines"].astype(np.float32)
# cube = mat["data"].astype(np.float32)

cube = mat["indian_pines_corrected"].astype(np.float32)  # most likely key
print("Cube shape:", cube.shape)  # should print (145, 145, 200)

wavelengths = np.linspace(400, 2500, 200)

eda = HyperspectralEDA(verbose=True, wavelengths=wavelengths)
eda.load(cube)
eda.summary()
eda.plot()

# mat = scipy.io.loadmat("C:/Users/Ike/Desktop/testhyper/PaviaU.mat")
# cube = mat["paviaU"].astype(np.float32)  # (610, 340, 103)
#
# wavelengths = np.linspace(430, 860, 103)  # ROSIS VNIR range
#
# eda = HyperspectralEDA(verbose=True, wavelengths=wavelengths)
# eda.load(cube)
# eda.plot(rgb_bands=(60, 30, 10))  # false colour