import os
from nightlightsprocessing import helpers as globalHelpers
from osgeo import gdal
import csv
from . import constants
from . import helpers

# This file is currently used to measure the spread of time between 
# the first and last points the satellite went over an area
SELECTED_FILE_INDEX = 0
FILE_TYPE = "VNP46A1"
SELECTED_DATASETS = "UTC_Time"
READ_METHOD = gdal.GA_ReadOnly

def _convert_to_normal_time(decimal_time):
    hours = int(decimal_time)
    minutes = int((decimal_time - hours) * 60)
    seconds = int(((decimal_time - hours) * 60 - minutes) * 60)
    time_string = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return time_string

def _calculate_time_spread(first_time_decimal, last_time_decimal):
    spread = last_time_decimal - first_time_decimal

    hours = int(spread)
    minutes = int((spread * 60) % 60)
    seconds = int((spread * 3600) % 60)

    return hours, minutes, seconds


def _get_subdataset_as_array(filename):
    data_set = gdal.Open(filename, gdal.GA_ReadOnly)
    all_subdatasets = data_set.GetSubDatasets()
    selected_subdataset = helpers.getSubDataset(SELECTED_DATASETS, all_subdatasets)

    sub_dataset = gdal.Open(selected_subdataset, READ_METHOD)
    raster_band = sub_dataset.GetRasterBand(1)
    img_array = raster_band.ReadAsArray()

    return img_array

def _get_date_from_filepath(hdf5filepath):
    path_from_julian_date_onwards = hdf5filepath.split(f"{FILE_TYPE}.A",1)[1]
    julian_date = path_from_julian_date_onwards.split('.')[0]

    date = helpers.get_datetime_from_julian_date(julian_date)

    return date


def _get_row_values(hdf5filepath):
    date = _get_date_from_filepath(hdf5filepath)
    img_array = _get_subdataset_as_array(hdf5filepath)

    # Time decimals
    first_time_decimal = img_array[0][0]  # First time in the first array
    last_time_decimal = img_array[-1][-1]  # Last time in the last array

    # If first_time_decimal > last_time_decimal, the satellite is travelling 
    # the other way when it takes the image, so we need to reverse the variables
    # to get the correct start/end tims and the spread between them
    if first_time_decimal > last_time_decimal:
        first_time_decimal = img_array[-1][-1]
        last_time_decimal = img_array[0][0]


    # Convert decimals to normal times
    start_time = _convert_to_normal_time(first_time_decimal)
    end_time = _convert_to_normal_time(last_time_decimal)

    # Calculate the spread between the steat and end time
    hours, minutes, seconds = _calculate_time_spread(first_time_decimal, last_time_decimal)
    
    # Round all values to 2dp
    spread = "%02d:%02d:%02d" % (hours, minutes, seconds)

    return [date, start_time, end_time, spread]

def main():
    all_files = globalHelpers.getAllFilesFromFolderWithFilename(constants.H5_INPUT_FOLDER, FILE_TYPE)

    # TODO: Should probably have the location name in the filename
    filename = f".{constants.OUTPUT_FOLDER}/vnp46a1_image_created_times.csv"

    # data = [
    #     ['date', 'start_time', 'end_time', 'spread_time'],  # Header row
    # ]
    data = []
    new_data = []


    # Get existing data if it exists
    if os.path.exists(filename):
        # Read the existing data from the file
        with open(filename, 'r') as file:
            reader = csv.reader(file)
            data = list(reader)

    # Get new data
    for file in all_files:
      hdf5filepath = f"{os.getcwd()}{constants.H5_INPUT_FOLDER}/{file}"

      date, start_time, end_time, spread_time = _get_row_values(hdf5filepath)

      new_data.append([date, start_time, end_time, spread_time])

    # Iterate over each new data item, replace it with new data if it exists,
    # otherwise add a new item to the .csv
    for new_item in new_data:
        date_to_check = new_item[0]
        item_exists = False

        # Iterate over each existing data item
        for i, existing_item in enumerate(data):
            date_index = 0
            if existing_item[date_index] == date_to_check:
                # Overwrite the existing item with the new item
                data[i] = new_item
                item_exists = True
                break

        if not item_exists:
            # Append the new item to the existing data
            data.append(new_item)

    # Write to a .csv file
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)

        if os.path.exists(filename):
            writer.writerows(data)
        else:
          writer.writerows([
              ['date', 'start_time', 'end_time', 'spread_time'],  # Header row,
              *data
          ])

        print(f"The data has been written to {filename}.")

if __name__ == "__main__":
    main()
