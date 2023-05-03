#!/usr/bin/env python3

# Mostly taken from https://blackmarble.gsfc.nasa.gov/tools/OpenHDF5.py
from osgeo import gdal

import constants
import helpers

SELECTED_DATASET = "DNB_At_Sensor_Radiance_500m"


def filterFilesThatInclude(subString, filenames):
    filtered = []

    for filename in filenames:
        if subString in filename:
            filtered.append(filename)
    return filtered


allFiles = helpers.getAllFilesFrom(constants.INPUT_FOLDER)
# Get only the files for a specific dataset
VNP46A2Files = filterFilesThatInclude("VNP46A2", allFiles)
firstFile = VNP46A2Files[0]

# https://gdal.org/api/python/osgeo.gdal.html#osgeo.gdal.BuildVRT
# Open HDF file
hdflayer = gdal.Open(firstFile, gdal.GA_ReadOnly)

# Get selected dataset from HDF file
dataset = hdflayer.GetSubDatasets()
selected_subdataset = helpers.getSubDataset(constants.SELECTED_DATASET, dataset)
rlayer = gdal.Open(selected_subdataset, gdal.GA_ReadOnly)
# outputName = rlayer.GetMetadata_Dict()['long_name']

# Subset the Long Name
outputName = selected_subdataset[92:]

outputNameNoSpace = outputName.strip().replace(" ", "_").replace("/", "_")
# Get first file Name and trim last 3 characters (probably '.h5') from the end of the filename
rasterFilePre = firstFile[:-3]
outputNameFinal = outputNameNoSpace + rasterFilePre + constants.FILE_EXTENSION_TIF
print("outputNameFinal", outputNameFinal)


outputRaster = constants.OUTPUT_PREFIX + outputNameFinal

translateOptionText = helpers.getCommandLineTranslateOptions(rlayer)
print("translateOptionText:", translateOptionText)

# https://gdal.org/api/python/osgeo.gdal.html#osgeo.gdal.TranslateOptions
translateoptions = gdal.TranslateOptions(gdal.ParseCommandLine(translateOptionText))
# translateoptions = gdal.TranslateOptions()
gdal.Translate(outputRaster, rlayer, options=translateoptions)

# Display image in QGIS (run it within QGIS python Console) - remove comment to display
# iface.addRasterLayer(outputRaster, outputNameFinal)
