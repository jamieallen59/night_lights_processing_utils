# This script mostly used for testing purposes to check that processing is working
# correctly on one file before moving onto using multiple_hd5_to_geotiff for larger datasets.
# Mostly taken from https://blackmarble.gsfc.nasa.gov/tools/OpenHDF5.py
from osgeo import gdal
import re

import constants
import helpers

SELECTED_DATASET = "DNB_At_Sensor_Radiance_500m"
# This is why the script is called 'single'.
# We just get the first filename found based on the index below
SELECTED_FILE = 0


def getSingleDatasetFromHd5(hd5File, subDatasetName):
    # Open HDF file
    hdflayer = gdal.Open(hd5File, gdal.GA_ReadOnly)
    # Get selected dataset from HDF file
    all_datasets = hdflayer.GetSubDatasets()
    selected_subdataset = helpers.getSubDataset(subDatasetName, all_datasets)
    sub_dataset = gdal.Open(selected_subdataset, gdal.GA_ReadOnly)
    print("selected_subdataset", selected_subdataset)

    # Get chars from position 92 to the end
    # example dataset name: HDF5:"VNP46A1.A2014001.h25v06.001.2019123191150.h5"://HDFEOS/GRIDS/VNP_Grid_DNB/Data_Fields/DNB_At_Sensor_Radiance_500m
    # example expected ouput: DNB_At_Sensor_Radiance_500m
    sub_dataset_name = selected_subdataset[92:]

    return [sub_dataset, sub_dataset_name]


def getDestinationFilename(filename, sub_dataset_ouput_name):
    # Trim the filename extension
    # (assuming it's 3 characters long as probably '.h5') from the end of the filename
    filename_without_extension = filename[:-3]
    # '.' must be removed from the filename to work with Google Earth
    first_file_main_without_periods = re.sub("\.", "-", filename_without_extension)
    # change file name
    output_name_without_prefix = (
        sub_dataset_ouput_name + first_file_main_without_periods + constants.FILE_EXTENSION_TIF
    )
    destination_filename = constants.OUTPUT_PREFIX + output_name_without_prefix

    return destination_filename


# Get all files in the given folder
all_files = helpers.getAllFilesFrom(constants.INPUT_FOLDER)
# Get the file in that folder based on the SELECTED_FILE index
filename = all_files[SELECTED_FILE]
# Extract the dataset from that file
dataset, sub_dataset_ouput_name = getSingleDatasetFromHd5(filename, SELECTED_DATASET)
# Get the destination filename
destination_filename = getDestinationFilename(filename, sub_dataset_ouput_name)
# Get command line options to pass to gdal.TranslateOptions
translate_option_text = helpers.getCommandLineTranslateOptions(dataset)
# https://gdal.org/api/python/osgeo.gdal.html#osgeo.gdal.TranslateOptions
# creates an options object to pass to gdal.Translate
translate_options = gdal.TranslateOptions(gdal.ParseCommandLine(translate_option_text))
# https://gdal.org/api/python/osgeo.gdal.html#osgeo.gdal.Translate
# converts a dataset
gdal.Translate(destination_filename, dataset, options=translate_options)
