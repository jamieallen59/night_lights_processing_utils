#!/usr/bin/env python3

# This script processes VNP46A2 datasets, exatrcting the various masks from the datasets
# and writing them to .tif files.

import argparse
import sys
import numpy as np
import numpy.ma as ma
import os
import re
import rasterio as rio
from rasterio.transform import from_origin

from . import constants
from . import helpers


DESC = "This script processes VNP46A2 datasets, exatrcting the various masks from the datasets and writing them to .tif files."
FILE_TYPE_VNP46A2 = "VNP46A2"

# Mask Variables
# Valid ranges and 'fill value' all defined here:
# https://viirsland.gsfc.nasa.gov/PDF/BlackMarbleUserGuide_v1.2_20220916.pdf
# The 'fill value' is a value used to show that data is missing for that pixel.
# The variables below are essentially config to adjust how your images are processed.
dnb_brdf_corrected_ntl_fill_value = 6553.5
MANDATORY_QUALITY_FLAG_FILL_VALUE = 255
MANDATORY_QUALITY_FLAG_POOR_QUALITY_VALUE = 2
# Need to validate this as value in docs says '10'
CLOUD_MASK_QUALITY_FLAG_CLOUD_DETECTION_RESULTS_AND_CONFIDENCE_INDICATOR_PROBABLY_CLOUDY = 2
# Need to validate this as value in docs says '11'
CLOUD_MASK_QUALITY_FLAG_CLOUD_DETECTION_RESULTS_AND_CONFIDENCE_INDICATOR_CONFIDENT_CLOUDY = 3
# Need to validate this as value in docs says '011'
CLOUD_MASK_QUALITY_FLAG_LAND_WATER_BACKGROUND_SEA_WATER = 3

################################################################################


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
        "width": array.shape[1],
        "height": array.shape[0],
        "count": count,
        "crs": crs,
        "transform": transform,
    }


def create_transform_vnp46a2(hdf5_filepath):
    """Creates a geographic transform for a VNP46A2 HDF5 file,
    based on longitude bounds, latitude bounds, and cell size.
    """
    # Extract bounding box from top-level dataset
    with rio.open(hdf5_filepath) as dataset:
        longitude_min = int(dataset.tags()["WestBoundingCoord"])
        longitude_max = int(dataset.tags()["EastBoundingCoord"])
        latitude_min = int(dataset.tags()["SouthBoundingCoord"])
        latitude_max = int(dataset.tags()["NorthBoundingCoord"])

        # Extract number of row and columns from first
        # Science Data Set (subdataset/band)
        with rio.open(dataset.subdatasets[0]) as band:
            num_rows, num_columns = (
                band.meta.get("height"),
                band.meta.get("width"),
            )

    # Define transform (top-left corner, cell size)
    transform = from_origin(
        longitude_min,
        latitude_max,
        (longitude_max - longitude_min) / num_columns,
        (latitude_max - latitude_min) / num_rows,
    )

    return transform


# array is the array of pixels for DNB_BRDF-Corrected_NTL
def removeMissingDataFrom(array):
    return ma.masked_where(
        array == dnb_brdf_corrected_ntl_fill_value,
        array,
        copy=True,
    )


# removing fill values and low quality pixels using the Mandatory_Quality_Flag1 mask
def applyMandatoryQualityFlagMask(array, mandatory_quality_flag_band):
    masked_for_fill_values = ma.masked_where(
        mandatory_quality_flag_band == MANDATORY_QUALITY_FLAG_FILL_VALUE, array, copy=True
    )

    masked_for_poor_quality = ma.masked_where(
        mandatory_quality_flag_band == MANDATORY_QUALITY_FLAG_POOR_QUALITY_VALUE, masked_for_fill_values, copy=True
    )

    return masked_for_poor_quality


def applyCloudQualityFlagMask(array, QF_cloud_mask_band):
    # Cloud Detection Results & Confidence Indicator: Extract QF_Cloud_Mask bits 6-7
    cloud_detection_bitmask = extract_qa_bits(qa_band=QF_cloud_mask_band, start_bit=6, end_bit=7)
    masked_for_probably_cloudy = ma.masked_where(
        cloud_detection_bitmask
        == CLOUD_MASK_QUALITY_FLAG_CLOUD_DETECTION_RESULTS_AND_CONFIDENCE_INDICATOR_PROBABLY_CLOUDY,
        array,
        copy=True,
    )
    masked_for_confident_cloudy = ma.masked_where(
        cloud_detection_bitmask
        == CLOUD_MASK_QUALITY_FLAG_CLOUD_DETECTION_RESULTS_AND_CONFIDENCE_INDICATOR_CONFIDENT_CLOUDY,
        masked_for_probably_cloudy,
        copy=True,
    )

    # Land/Water Background: Extract QF_Cloud_Mask bits 1-3
    land_water_bitmask = extract_qa_bits(qa_band=QF_cloud_mask_band, start_bit=1, end_bit=3)
    masked_for_sea_water = ma.masked_where(
        land_water_bitmask == CLOUD_MASK_QUALITY_FLAG_LAND_WATER_BACKGROUND_SEA_WATER,
        masked_for_confident_cloudy,
        copy=True,
    )

    return masked_for_sea_water


def extract_band(hd5_filepath, band_name):
    if band_name not in constants.BAND_NAMES:
        raise ValueError(f"Invalid band name. Must be one of the following: {constants.BAND_NAMES}")

    # Open top-level dataset, loop through Science Data Sets (subdatasets),
    #  and extract specified band
    with rio.open(hd5_filepath) as dataset:
        for science_data_set in dataset.subdatasets:
            if re.search(f"{band_name}$", science_data_set):
                with rio.open(science_data_set) as src:
                    band = src.read(1)

    return band


def process_vnp46a2(hd5_filepath, destination):
    try:
        DNB_BRDF_corrected_ntl_band = extract_band(hd5_filepath, constants.BRDF_CORRECTED)
        mandatory_quality_flag_band = extract_band(hd5_filepath, constants.QUALITY_FLAG)
        QF_cloud_mask_band = extract_band(hd5_filepath, constants.CLOUD_MASK)

        print("Applying scale factor...")
        dnb_brdf_corrected_ntl_scaled = DNB_BRDF_corrected_ntl_band.astype("float") * 0.1
        print("Masking for fill values...")
        masked_for_fill_value_array = removeMissingDataFrom(dnb_brdf_corrected_ntl_scaled)
        print("Masking for poor quality and no retrieval...")
        masked_for_poor_quality_and_no_retrieval_array = applyMandatoryQualityFlagMask(
            masked_for_fill_value_array, mandatory_quality_flag_band
        )
        print("Masking for clouds...")
        result = applyCloudQualityFlagMask(masked_for_poor_quality_and_no_retrieval_array, QF_cloud_mask_band)
        print("Masking for sea water...")

        # TODO: Do I need to do this?
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

        print("Creating metadata...")
        metadata = create_metadata(
            array=filled_data,
            transform=create_transform_vnp46a2(hd5_filepath),
            driver="GTiff",
            nodata=np.nan,
            count=1,
            crs="epsg:4326",
        )

        # Export masked array to GeoTiff (no data set to np.nan in export)
        export_name = helpers.get_hd5_to_tif_export_name(hd5_filepath)
        helpers.export_array(
            array=filled_data,
            output_path=os.path.join(destination, export_name),
            metadata=metadata,
        )
    except Exception as error:
        message = print(f"Preprocessing failed: {error}\n")
    else:
        message = print(f"Completed preprocessing: {os.path.basename(hd5_filepath)}\n")

    return message


def process_VNP46A2_images(input_folder, destination):
    # Is this better than a helper?
    # hdf5_files = glob.glob(os.path.join(hdf5_input_folder, "*.h5"))
    processed_files = 0
    all_hd5_files = helpers.getAllFilesFromFolderWithFilename(input_folder, FILE_TYPE_VNP46A2)
    total_files = len(all_hd5_files)
    print("\n")
    print(f"Total files to process: {total_files}\n")

    for filename in all_hd5_files:
        filepath = f"{input_folder}/{filename}"

        process_vnp46a2(filepath, destination)
        processed_files += 1
        print(f"\nPreprocessed file: {processed_files} of {total_files}\n")

    # -------------------------SCRIPT COMPLETION--------------------------------- #
    print("\n")
    print("-" * (18 + len(os.path.basename(__file__))))
    print(f"Completed script: {os.path.basename(__file__)}")
    print("-" * (18 + len(os.path.basename(__file__))))


################################################################################


def _main(argv):
    parser = argparse.ArgumentParser(prog=argv[0], description=DESC)
    parser.add_argument(
        "-d",
        "--destination",
        dest="destination",
        help="Store directory structure in DIR",
        required=True,
    )
    parser.add_argument("-i", "--input-folder", dest="input_folder", help="Input data directory", required=True)

    args = parser.parse_args(argv[1:])

    process_VNP46A2_images(args.input_folder, args.destination)


if __name__ == "__main__":
    try:
        sys.exit(_main(sys.argv))
    except KeyboardInterrupt:
        sys.exit(-1)
