#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt 
from osgeo import gdal

# open GeoTIFF file and read raster band data into array
# geo_tiff_file = "F10199204111340.night.OIS.samples.co.tif"

geo_tiff_file = "GDNBO_npp_d20230301_t0127212_e0132598_b58760_c20230301025831459711_oeac_ops.samples.co.tif"
data_set = gdal.Open(geo_tiff_file, gdal.GA_ReadOnly)
raster_band = data_set.GetRasterBand(1)
img_array = raster_band.ReadAsArray()

# display a portion of the image for the region of India
india_region = img_array[110000:180000:2, 580000:660000:2]
plt.figure(dpi=300)
plt.imshow(india_region, cmap='gray', vmin=0, vmax=15)
plt.colorbar()

# extract a smaller subset of the image data for India and clip values above 20
india_data = img_array[100000:170000:5, 590000:650000:5]
india_clip = np.clip(india_data, 0, 20)

# create a histogram of the pixel brightness values in the extracted subset
num_bins = 500
plt.title("Histogram of Brightness")
plt.ylabel("Y-axis")
plt.xlabel("X-axis")
plt.hist(india_clip.flatten(), bins=num_bins)
plt.show()