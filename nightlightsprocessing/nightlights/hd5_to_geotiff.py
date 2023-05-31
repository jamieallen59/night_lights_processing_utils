#!/usr/bin/env python3

from osgeo import gdal
import re
import os

from nightlightsprocessing import helpers as globalHelpers

from . import constants
from . import helpers

# Can use gdal.GA_Update if you want to modify the file, but it takes longer.
READ_METHOD = gdal.GA_ReadOnly

# WARNING: VERY FRAGILE. BASED ON YOU'RE LOCAL ENVIRONMENT. 
LOCAL_ENVIRONMENT_PATH_LENGTH = 190

def _get_all_subdataset_names(subdatasets):
    subdataset_names = []

    for subdataset in subdatasets:
        subdataset_name = subdataset[0].split("Data_Fields/", 1)[1]
        subdataset_names.append(subdataset_name)

    return subdataset_names


def _get_single_dataset_from_hd5(hd5_file, dataset_name):
    # Open HDF file
    hdf5filepath = f"{os.getcwd()}{constants.H5_INPUT_FOLDER}/{hd5_file}"
    hdflayer = gdal.Open(hdf5filepath, READ_METHOD)

    # Get selected dataset from HDF file
    all_subdatasets = hdflayer.GetSubDatasets()

    # print("All available datasets: ", all_subdatasets)
    selected_subdataset = helpers.getSubDataset(dataset_name, all_subdatasets)
    # print("Selected subdataset: ", selected_subdataset)
    if selected_subdataset is None:
        raise RuntimeError(
            f"\nThe subdataset: '{dataset_name}' \nWas not available in subdatasets: {_get_all_subdataset_names(all_subdatasets)}"
        )

    sub_dataset = gdal.Open(selected_subdataset, READ_METHOD)

    # Get chars from position LOCAL_ENVIRONMENT_PATH_LENGTH to the end
    # example dataset name: HDF5:"VNP46A1.A2014001.h25v06.001.2019123191150.h5"://HDFEOS/GRIDS/VNP_Grid_DNB/Data_Fields/DNB_At_Sensor_Radiance_500m
    # example expected ouput: DNB_At_Sensor_Radiance_500m
    sub_dataset_name = selected_subdataset[LOCAL_ENVIRONMENT_PATH_LENGTH:]

    return [sub_dataset, sub_dataset_name]


def _get_datasets_from_hd5(hd5_file, dataset_names):
    datasets = []

    for dataset_name in dataset_names:
        sub_dataset, sub_dataset_ouput_name = _get_single_dataset_from_hd5(hd5_file, dataset_name)

        datasets.append([sub_dataset, sub_dataset_ouput_name])

    return datasets


def _get_destination_filename(filename, sub_dataset_ouput_name):
    # Trim the filename extension
    # (assuming it's 3 characters long as probably '.h5') from the end of the filename
    filename_without_extension = filename[:-3]

    # '.' must be removed from the filename to work with Google Earth
    filename_without_periods = re.sub("\.", "-", filename_without_extension)

    # change file name
    destination_filename = (
        sub_dataset_ouput_name + filename_without_periods + constants.FILE_EXTENSION_TIF
    )

    return destination_filename


def _write_datasets_to_tif(datasets, filename):
    for dataset in datasets:
        sub_dataset, sub_dataset_ouput_name = dataset

        # Get the destination filename
        # Writes to another input folder as becomes the input for other scripts
        destination_filename = f"{os.getcwd()}{constants.TIF_INPUT_FOLDER}/" + _get_destination_filename(filename, sub_dataset_ouput_name)
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
    # Get all files in the given folder
    all_files = globalHelpers.getAllFilesFromFolderWithFilename(constants.H5_INPUT_FOLDER, constants.FILE_TYPE)

    for filename in all_files:
      datasets = _get_datasets_from_hd5(filename, constants.SELECTED_DATASETS)

      _write_datasets_to_tif(datasets, filename)
    
    print('Completed writing files to .tif')


if __name__ == "__main__":
    main()
