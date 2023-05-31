import os

from nightlightsprocessing import helpers as globalHelpers
from osgeo import gdal
from . import constants
from . import helpers

# This file is currently used to measure the spread of time between 
# the first and last points the satellite went over an area
SELECTED_FILE_INDEX = 0
FILE_TYPE = "VNP46A1"
SELECTED_DATASETS = "UTC_Time"
READ_METHOD = gdal.GA_ReadOnly

def convert_to_normal_time(decimal_time):
    hours = int(decimal_time)
    minutes = int((decimal_time - hours) * 60)
    seconds = int(((decimal_time - hours) * 60 - minutes) * 60)
    time_string = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return time_string

def calculate_time_spread(first_time_decimal, last_time_decimal):
    spread = last_time_decimal - first_time_decimal

    hours = int(spread)
    minutes = int((spread * 60) % 60)
    seconds = int((spread * 3600) % 60)

    return hours, minutes, seconds


def getSubdatasetArray(filename):
    data_set = gdal.Open(filename, gdal.GA_ReadOnly)
    all_subdatasets = data_set.GetSubDatasets()
    selected_subdataset = helpers.getSubDataset(SELECTED_DATASETS, all_subdatasets)

    sub_dataset = gdal.Open(selected_subdataset, READ_METHOD)
    raster_band = sub_dataset.GetRasterBand(1)
    img_array = raster_band.ReadAsArray()

    return img_array

def main():
    all_files = globalHelpers.getAllFilesFromFolderWithFilename(constants.INPUT_FOLDER, FILE_TYPE)
    hdf5filename = all_files[SELECTED_FILE_INDEX]
    hdf5filepath = f"{os.getcwd()}{constants.INPUT_FOLDER}/{hdf5filename}"

    img_array = getSubdatasetArray(hdf5filepath)
    print("Image array", img_array)

    # Time decimals
    first_time_decimal = img_array[0][0]  # First time in the first array
    last_time_decimal = img_array[-1][-1]  # Last time in the last array

    # Normal times
    first_time = convert_to_normal_time(first_time_decimal)
    last_time = convert_to_normal_time(last_time_decimal)

    print('Start time: ', first_time)
    print('End time: ', last_time)

    hours, minutes, seconds = calculate_time_spread(first_time_decimal, last_time_decimal)
    print("Spread: %02d:%02d:%02d" % (hours, minutes, seconds))
    
    # TODO: read all VNP46A1 files
    # TODO: write these values to columns
      # date, start time, end time, spread


if __name__ == "__main__":
    main()


    # with rasterio.open(hdf5filename) as dataset:
    #     for science_data_set in dataset.subdatasets:
    #         if re.search(f"{SELECTED_DATASETS}$", science_data_set):
    #             with rasterio.open(science_data_set) as src:
    #                 band = src.read(1)
    #                 print("rasterio band", band)

    # print("Tags", dataset.tags())
    # print("META", dataset.meta)
    # print("dataset", dataset)
    # print("dataset.width", dataset.width)
    # print("dataset.height", dataset.height)
    # print("dataset.bounds", dataset.bounds)
    # print("dataset.transform", dataset.transform)
    # print("dataset upper left", dataset.transform * (0, 0))
    # print("dataset bottom right", dataset.transform * (dataset.width, dataset.height))
    # print("dataset.crs", dataset.crs)