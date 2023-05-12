#!/usr/bin/env python3

# This script mostly used for testing purposes to check that processing is working
# correctly on one file before moving onto using the multiple_hd5_to_geotiff script for larger datasets.
# Mostly taken from https://blackmarble.gsfc.nasa.gov/tools/OpenHDF5.py
from osgeo import gdal
import re
import os

import constants
import helpers

# This is why the script is called 'single'.
# We just get the first filename found based on the index below
SELECTED_FILE_INDEX = 0
# Can use gdal.GA_Update if you want to modify the file, but it takes longer.
READ_METHOD = gdal.GA_ReadOnly
# SELECTED_DATASET = "DNB_BRDF-Corrected_NTL"


def getAllSubdatasetNames(subdatasets):
    subdataset_names = []

    for subdataset in subdatasets:
        subdataset_name = subdataset[0].split("Data_Fields/", 1)[1]
        subdataset_names.append(subdataset_name)

    return subdataset_names


def getSingleDatasetFromHd5(hd5_file, dataset_name):
    # Open HDF file
    hdflayer = gdal.Open(hd5_file, READ_METHOD)
    # Get selected dataset from HDF file
    all_subdatasets = hdflayer.GetSubDatasets()

    # print("All available datasets: ", all_subdatasets)
    selected_subdataset = helpers.getSubDataset(dataset_name, all_subdatasets)
    # print("Selected subdataset: ", selected_subdataset)
    if selected_subdataset is None:
        raise RuntimeError(
            f"\nThe subdataset: '{dataset_name}' \nWas not available in subdatasets: {getAllSubdatasetNames(all_subdatasets)}"
        )

    sub_dataset = gdal.Open(selected_subdataset, READ_METHOD)

    # Print some metadata
    meta_data = hdflayer.GetMetadata()
    # print("All metadata", meta_data)
    # print("Metadata showing when the image was taken:")
    # print("RangeBeginningDate", meta_data["RangeBeginningDate"])
    # print("RangeBeginningTime", meta_data["RangeBeginningTime"])
    # print("RangeEndingDate", meta_data["RangeEndingDate"])
    # print("RangeEndingTime", meta_data["RangeEndingTime"])

    # Get chars from position 92 to the end
    # example dataset name: HDF5:"VNP46A1.A2014001.h25v06.001.2019123191150.h5"://HDFEOS/GRIDS/VNP_Grid_DNB/Data_Fields/DNB_At_Sensor_Radiance_500m
    # example expected ouput: DNB_At_Sensor_Radiance_500m
    sub_dataset_name = selected_subdataset[92:]
    print("Sub dataset name:", sub_dataset_name)

    return [sub_dataset, sub_dataset_name]


def getDatasetsFromHd5(hd5_file, dataset_names):
    datasets = []

    for dataset_name in dataset_names:
        sub_dataset, sub_dataset_ouput_name = getSingleDatasetFromHd5(hd5_file, dataset_name)

        datasets.append([sub_dataset, sub_dataset_ouput_name])

    return datasets


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


def writeDatasetsToTiff(datasets, filename):
    for dataset in datasets:
        sub_dataset, sub_dataset_ouput_name = dataset
        # Get the destination filename
        destination_filename = f".{constants.OUTPUT_FOLDER}" + getDestinationFilename(filename, sub_dataset_ouput_name)
        # Get command line options to pass to gdal.TranslateOptions
        translate_option_text = helpers.getCommandLineTranslateOptions(sub_dataset)
        # https://gdal.org/api/python/osgeo.gdal.html#osgeo.gdal.TranslateOptions
        # creates an options object to pass to gdal.Translate
        translate_options = gdal.TranslateOptions(gdal.ParseCommandLine(translate_option_text))
        # https://gdal.org/api/python/osgeo.gdal.html#osgeo.gdal.Translate
        # converts a dataset
        gdal.Translate(destination_filename, sub_dataset, options=translate_options)
        print("Writing to filename:", destination_filename)


def main():
    folder = constants.INPUT_FOLDER
    # Change from root file into given folder
    os.chdir(folder)
    # Get all files in the given folder
    all_files = helpers.getAllFilesFrom(folder, constants.FILE_TYPE)
    # Get the file in that folder based on the SELECTED_FILE_INDEX index
    filename = all_files[SELECTED_FILE_INDEX]
    # Extract the dataset from that file
    datasets = getDatasetsFromHd5(filename, constants.SELECTED_DATASETS)
    writeDatasetsToTiff(datasets, filename)


if __name__ == "__main__":
    main()
