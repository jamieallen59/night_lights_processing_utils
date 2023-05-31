#!/usr/bin/env python3

import numpy as np
import numpy.ma as ma

# import matplotlib.pyplot as plt
from osgeo import gdal
import os
import rasterio
from rasterio.transform import from_origin

from . import constants
from . import helpers

output_folder = f".{constants.OUTPUT_FOLDER}"
input_folder = f".{constants.INPUT_FOLDER}"
image_output_size = 512


# This file should be run after the hd5 file has already been converted to geotiff by hd5_to_geotiff (currently)
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


def create_metadata(array, transform, driver="GTiff", nodata=0, count=1, crs="epsg:4326"):
    return {
        "driver": driver,
        "dtype": array.dtype,
        "nodata": nodata,
        "width": image_output_size,
        "height": image_output_size,
        "count": count,
        "crs": crs,
        "transform": transform,
    }


def create_transform_vnp46a2(hdf5):
    """Creates a geographic transform for a VNP46A2 HDF5 file,
    based on longitude bounds, latitude bounds, and cell size.
    """
    # Extract bounding box from top-level dataset
    with rasterio.open(hdf5) as dataset:
        longitude_min = int(dataset.tags()["NC_GLOBAL#WestBoundingCoord"])
        longitude_max = int(dataset.tags()["NC_GLOBAL#EastBoundingCoord"])
        latitude_min = int(dataset.tags()["NC_GLOBAL#SouthBoundingCoord"])
        latitude_max = int(dataset.tags()["NC_GLOBAL#NorthBoundingCoord"])

        num_rows = dataset.meta.get("height")
        num_columns = dataset.meta.get("width")

    # Define transform (top-left corner, cell size)
    transform = from_origin(
        longitude_min,
        latitude_max,
        (longitude_max - longitude_min) / num_columns,
        (latitude_max - latitude_min) / num_rows,
    )
    print("transform", transform)

    return transform


# Masks
# Valid ranges and 'fill value' all defined here:
# https://viirsland.gsfc.nasa.gov/PDF/BlackMarbleUserGuide_v1.2_20220916.pdf
# The 'fill value' is a value used to show that data is missing for that pixel.
# The variables below are essentially config to adjust how your images are processed.
dnb_brdf_corrected_ntl_fill_value = 6553.5
mandatory_quality_flag_fill_value = 255
mandatory_quality_flag_poor_quality_value = 2
# Need to validate this as value in docs says '10'
cloud_mask_quality_flag_cloud_detection_results_and_confidence_indicator_probably_cloudy = 2
# Need to validate this as value in docs says '11'
cloud_mask_quality_flag_cloud_detection_results_and_confidence_indicator_confident_cloudy = 3
# Need to validate this as value in docs says '011'
cloud_mask_quality_flag_land_water_background_sea_water = 3


# array is the array of pixels for DNB_BRDF-Corrected_NTL
def removeMissingDataFrom(array):
    return ma.masked_where(
        array == dnb_brdf_corrected_ntl_fill_value,
        array,
        copy=True,
    )


# removing fill values and low quality pixels using the Mandatory_Quality_Flag1 mask
def applyMandatoryQualityFlagMask(array):
    all_mandatory_quality_flag_ntl_files = helpers.getAllFilesFrom(constants.OUTPUT_FOLDER, constants.QUALITY_FLAG)
    first_mandatory_quality_flag_ntl_file = all_mandatory_quality_flag_ntl_files[0]
    mandatory_quality_flag = getBand(first_mandatory_quality_flag_ntl_file)

    masked_for_fill_values = ma.masked_where(
        mandatory_quality_flag == mandatory_quality_flag_fill_value, array, copy=True
    )

    masked_for_poor_quality = ma.masked_where(
        mandatory_quality_flag == mandatory_quality_flag_poor_quality_value, masked_for_fill_values, copy=True
    )

    return masked_for_poor_quality


def applyCloudQualityFlagMask(array):
    cloud_mask_ntl_files = helpers.getAllFilesFrom(constants.OUTPUT_FOLDER, constants.CLOUD_MASK)
    cloud_mask = getBand(cloud_mask_ntl_files[0])

    # Cloud Detection Results & Confidence Indicator: Extract QF_Cloud_Mask bits 6-7
    cloud_detection_bitmask = extract_qa_bits(qa_band=cloud_mask, start_bit=6, end_bit=7)
    masked_for_probably_cloudy = ma.masked_where(
        cloud_detection_bitmask
        == cloud_mask_quality_flag_cloud_detection_results_and_confidence_indicator_probably_cloudy,
        array,
        copy=True,
    )
    masked_for_confident_cloudy = ma.masked_where(
        cloud_detection_bitmask
        == cloud_mask_quality_flag_cloud_detection_results_and_confidence_indicator_confident_cloudy,
        masked_for_probably_cloudy,
        copy=True,
    )

    # Land/Water Background: Extract QF_Cloud_Mask bits 1-3
    land_water_bitmask = extract_qa_bits(qa_band=cloud_mask, start_bit=1, end_bit=3)
    masked_for_sea_water = ma.masked_where(
        land_water_bitmask == cloud_mask_quality_flag_land_water_background_sea_water,
        masked_for_confident_cloudy,
        copy=True,
    )

    return masked_for_sea_water


def main():
    # Change from root file into given folder
    os.chdir(constants.OUTPUT_FOLDER)

    BRDF_corrected_ntl_files = helpers.getAllFilesFrom(constants.OUTPUT_FOLDER, constants.BRDF_CORRECTED)
    BRDF_corrected_ntl_file = getBand(BRDF_corrected_ntl_files[0])

    print("Applying scale factor...")
    dnb_brdf_corrected_ntl_scaled = BRDF_corrected_ntl_file.astype("float") * 0.1
    print("Masking for fill values...")
    masked_for_fill_value = removeMissingDataFrom(dnb_brdf_corrected_ntl_scaled)
    print("Masking for poor quality and no retrieval...")
    masked_for_poor_quality_and_no_retrieval = applyMandatoryQualityFlagMask(masked_for_fill_value)
    print("Masking for clouds...")
    result = applyCloudQualityFlagMask(masked_for_poor_quality_and_no_retrieval)
    print("Masking for sea water...")

    # Do I need to do this?
    # print("Masking for sensor problems...")
    # # Mask radiance for sensor problems (QF_DNB != 0)
    # #  (0 = no problems, any number > 0 means some kind of issue)
    # # masked_for_sensor_problems = ma.masked_where(
    # #     qf_dnb > 0, masked_for_confident_cloudy, copy=True
    # # )
    # masked_for_sensor_problems = ma.masked_where(
    #     qf_dnb > 0, masked_for_sea_water, copy=True
    # )

    print("Filling masked values...")
    # Set fill value to np.nan and fill masked values
    ma.set_fill_value(result, np.nan)
    filled_data = result.filled()

    print("revert scale factor...")
    final = filled_data.astype("float") * 10

    # Get hd5 path
    # Change from root file into given folder
    os.chdir(input_folder)
    # Get all files in the given folder
    all_files = helpers.getAllFilesFrom(input_folder, constants.FILE_TYPE)
    # Get the file in that folder based on the SELECTED_FILE_INDEX index
    hdf5_path = all_files[0]

    print("Creating metadata...")
    metadata = create_metadata(
        array=final,
        transform=create_transform_vnp46a2(hdf5_path),
        driver="GTiff",
        nodata=np.nan,
        count=1,
        crs="epsg:4326",
    )

    # Export masked array to GeoTiff (no data set to np.nan in export)
    export_name = (
        f"{os.path.basename(hdf5_path)[:-3].lower().replace('.', '-')}.tif"
    )
    helpers.export_array(
        array=final,
        output_path=os.path.join(output_folder, export_name),
        metadata=metadata,
    )


if __name__ == "__main__":
    main()
