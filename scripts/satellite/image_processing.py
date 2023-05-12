#!/usr/bin/env python3

import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
from osgeo import gdal
import os

import constants
import helpers

output_folder = f".{constants.OUTPUT_FOLDER}"


def getBand(filename):
    data_set = gdal.Open(filename, gdal.GA_ReadOnly)
    raster_band = data_set.GetRasterBand(1)
    img_array = raster_band.ReadAsArray()

    return img_array


def extract_qa_bits(qa_band, start_bit, end_bit):
    """Extracts the QA bitmask values for a specified bitmask (starting
    and ending bit).
    """
    # Initialize QA bit string/pattern to check QA band against
    qa_bits = 0
    # Add each specified QA bit flag value/string/pattern
    #  to the QA bits to check/extract
    for bit in range(start_bit, end_bit + 1):
        qa_bits += bit**2
    # Check QA band against specified QA bits to see what
    #  QA flag values are set
    qa_flags_set = qa_band & qa_bits
    # Get base-10 value that matches bitmask documentation
    #  (0-1 for single bit, 0-3 for 2 bits, or 0-2^N for N bits)
    qa_values = qa_flags_set >> start_bit

    return qa_values


# Masks
def maskForFillValues(array):
    # Mask radiance for fill value (dnb_brdf_corrected_ntl == 65535)
    return ma.masked_where(
        array == 6553.5,
        array,
        copy=True,
    )


def maskForPoorQualityAndNoRetrieval(masked_for_fill_value):
    # Mask radiance for 'poor quality' (mandatory_quality_flag == 2)
    masked_for_poor_quality = ma.masked_where(mandatory_quality_flag == 2, masked_for_fill_value, copy=True)
    # Mask radiance for 'no retrieval' (mandatory_quality_flag == 255)
    masked_for_poor_quality_and_no_retrieval = ma.masked_where(
        mandatory_quality_flag == 255, masked_for_poor_quality, copy=True
    )

    return masked_for_poor_quality_and_no_retrieval


def maskForClouds(cloud_mask, masked_for_poor_quality_and_no_retrieval):
    # Extract QF_Cloud_Mask bits 6-7 (Cloud Detection Results & Confidence Indicator)
    cloud_detection_bitmask = extract_qa_bits(qa_band=cloud_mask, start_bit=6, end_bit=7)
    # Mask radiance for 'probably cloudy' (cloud_mask == 2)
    masked_for_probably_cloudy = ma.masked_where(
        cloud_detection_bitmask == 2, masked_for_poor_quality_and_no_retrieval, copy=True
    )
    # Mask radiance for 'confident cloudy' (cloud_mask == 3)
    masked_for_probably_and_confident_cloudy = ma.masked_where(
        cloud_detection_bitmask == 3, masked_for_probably_cloudy, copy=True
    )

    return masked_for_probably_and_confident_cloudy


def maskForSeaWater(cloud_mask, masked_for_probably_and_confident_cloudy):
    # Extract QF_Cloud_Mask bits 1-3 (Land/Water Background)
    land_water_bitmask = extract_qa_bits(qa_band=cloud_mask, start_bit=1, end_bit=3)
    # Mask radiance for sea water (land_water_bitmask == 3)
    masked_for_land_water = ma.masked_where(
        land_water_bitmask == 3, masked_for_probably_and_confident_cloudy, copy=True
    )

    return masked_for_land_water


folder = constants.OUTPUT_FOLDER
# Change from root file into given folder
os.chdir(folder)

BRDF_corrected_ntl_files = helpers.getAllFilesFrom(folder, constants.BRDF_CORRECTED)
quality_flag_ntl_files = helpers.getAllFilesFrom(folder, constants.QUALITY_FLAG)
cloud_mask_ntl_files = helpers.getAllFilesFrom(folder, constants.CLOUD_MASK)

BRDF_corrected_ntl_file = getBand(BRDF_corrected_ntl_files[0])
mandatory_quality_flag = getBand(quality_flag_ntl_files[0])
cloud_mask = getBand(cloud_mask_ntl_files[0])

print("Applying scale factor...")
dnb_brdf_corrected_ntl_scaled = BRDF_corrected_ntl_file.astype("float") * 0.1
print("Masking for fill values...")
masked_for_fill_value = maskForFillValues(dnb_brdf_corrected_ntl_scaled)
print("Masking for poor quality and no retrieval...")
masked_for_poor_quality_and_no_retrieval = maskForPoorQualityAndNoRetrieval(masked_for_fill_value)
print("Masking for clouds...")
masked_for_probably_and_confident_cloudy = maskForClouds(cloud_mask, masked_for_poor_quality_and_no_retrieval)
print("Masking for sea water...")
masked_for_land_water = maskForSeaWater(cloud_mask, masked_for_probably_and_confident_cloudy)
print("masked_for_sea_water", masked_for_land_water)

# Needs to write array to a new geotiff file using helpers.export_array
# Then check file in googleearth
# Then crop pixels related to Lucknow. Should probably do this before processing begins?
# Then compare with ground truth
