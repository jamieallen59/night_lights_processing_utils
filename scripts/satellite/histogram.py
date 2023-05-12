#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from osgeo import gdal

import constants
import helpers

# Just grab the first file found for now
SELECTED_FILE_INDEX = 0


all_files = helpers.getAllFilesFrom(constants.OUTPUT_FOLDER, constants.OUTPUT_PREFIX)
filename = all_files[SELECTED_FILE_INDEX]
data_set = gdal.Open(filename, gdal.GA_ReadOnly)
raster_band = data_set.GetRasterBand(1)
img_array = raster_band.ReadAsArray()

plt.figure(dpi=300)
plt.imshow(img_array, cmap="gray", vmin=0, vmax=15)
plt.colorbar()

# # extract a smaller subset of the image data for India and clip values above 20
# india_data = img_array[100000:170000:5, 590000:650000:5]
# india_clip = np.clip(india_data, 0, 20)

# create a histogram of the pixel brightness values in the extracted subset
num_bins = 500
plt.title("Histogram of Brightness")
plt.ylabel("Y-axis")
plt.xlabel("X-axis")
plt.hist(img_array.flatten(), bins=num_bins)
plt.show()
