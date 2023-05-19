import os
import re

import constants
import helpers
from osgeo import gdal

import rasterio

SELECTED_FILE_INDEX = 0
FILE_TYPE = "VNP46A1"
SELECTED_DATASETS = "UTC_Time"
READ_METHOD = gdal.GA_ReadOnly


def getBand(filename):
    data_set = gdal.Open(filename, gdal.GA_ReadOnly)
    raster_band = data_set.GetRasterBand(1)
    img_array = raster_band.ReadAsArray()

    return img_array


def main():
    folder = constants.INPUT_FOLDER
    # Change from root file into given folder
    os.chdir(folder)
    # Get all files in the given folder
    all_files = helpers.getAllFilesFrom(folder, FILE_TYPE)
    # Get the file in that folder based on the SELECTED_FILE_INDEX index
    hdf5filename = all_files[SELECTED_FILE_INDEX]
    print("hdf5filename", hdf5filename)

    # print("GDAL band", getBand(hdf5filename))

    # with rasterio.open(hdf5filename) as dataset:
    #     for science_data_set in dataset.subdatasets:
    #         if re.search(f"{SELECTED_DATASETS}$", science_data_set):
    #             with rasterio.open(science_data_set) as src:
    #                 band = src.read(1)
    #                 print("rasterio band", band)

    with rasterio.open(hdf5filename) as dataset:
        for science_data_set in dataset.subdatasets:
            if re.search(f"{SELECTED_DATASETS}$", science_data_set):
                # with rasterio.open(science_data_set) as src:
                print("science_data_set", science_data_set.read(1))

            # band = src.read(1)

    # with rasterio.open(hdf5filename) as dataset:
    #     # print("--- rasterio ---")
    #     band1 = dataset.read(1)
    #     print("rasterio band", band1)

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

    # print("dataset.subdatasets", dataset.subdatasets.index("UTC_Time"))

    # with rasterio.open(dataset.subdatasets[0]) as band:
    #     print("band", band)

    # hdflayer = gdal.Open(hdf5filename, READ_METHOD)
    # all_subdatasets = hdflayer.GetSubDatasets()
    # selected_subdataset = helpers.getSubDataset(SELECTED_DATASETS, all_subdatasets)
    # print("--- gdal ---")
    # print("selected_subdataset", selected_subdataset)
    # sub_dataset = gdal.Open(selected_subdataset, READ_METHOD)
    # print("sub_dataset", sub_dataset.ReadAsArray())


if __name__ == "__main__":
    main()
