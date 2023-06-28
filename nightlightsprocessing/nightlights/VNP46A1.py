import os
from nightlightsprocessing import helpers as globalHelpers
from osgeo import gdal
import rasterio as rio
import re
import csv
from . import constants
from . import helpers
import datetime

# This file is currently used to measure the spread of time between
# the first and last points the satellite went over an area
SELECTED_FILE_INDEX = 0
FILE_TYPE = "VNP46A1"
SELECTED_DATASET = "UTC_Time"
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
    # data_set = gdal.Open(filename, gdal.GA_ReadOnly)
    # all_subdatasets = data_set.GetSubDatasets()
    # selected_subdataset = helpers.getSubDataset(SELECTED_DATASET, all_subdatasets)

    # sub_dataset = gdal.Open(selected_subdataset, READ_METHOD)
    # raster_band = sub_dataset.GetRasterBand(1)
    # img_array = raster_band.ReadAsArray()

    # Open top-level dataset, loop through Science Data Sets (subdatasets),
    #  and extract specified band
    with rio.open(filename) as dataset:
        for science_data_set in dataset.subdatasets:
            if re.search(f"{SELECTED_DATASET}$", science_data_set):
                with rio.open(science_data_set) as src:
                    band = src.read(1)

    return band

    # return img_array


def _get_date_from_filepath(hdf5filepath):
    path_from_julian_date_onwards = hdf5filepath.split(f"{FILE_TYPE}.A", 1)[1]
    julian_date = path_from_julian_date_onwards.split(".")[0]

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


def _get_vnp46a1_time_data():
    all_files = globalHelpers.getAllFilesFromFolderWithFilename(constants.H5_INPUT_FOLDER, FILE_TYPE)
    data = []

    for file in all_files:
        hdf5filepath = f"{os.getcwd()}{constants.H5_INPUT_FOLDER}/{file}"

        date, start_time, end_time, spread_time = _get_row_values(hdf5filepath)

        data.append([date, start_time, end_time, spread_time])

    return data


def parse_date(date_str):
    if isinstance(date_str, datetime.date):
        return date_str
    return datetime.datetime.strptime(date_str, "%Y-%m-%d").date()


# Check they're equal by their date field
def _check_items_are_equal(new_item, existing_item):
    date_column_index = 0
    new_item_date = new_item[date_column_index]
    existing_item_date = existing_item[date_column_index]

    # Necessary as dates come back as strings or datetime.date's from the .csv
    existing_item_date_parsed = parse_date(existing_item_date)

    if existing_item_date_parsed == new_item_date:
        return True
    else:
        return False


def _merge_new_and_existing_data(filename):
    new_data = _get_vnp46a1_time_data()

    # Read the existing data from the file
    with open(filename, "r") as file:
        reader = csv.reader(file)
        existing_data = list(reader)

    for new_item in new_data:
        date_column_index = 0
        item_exists = False

        # Iterate over each existing data item
        for i, existing_item in enumerate(existing_data):
            is_header = existing_item[date_column_index] == "date"

            if not is_header:
                item_exists = _check_items_are_equal(new_item, existing_item)

                if item_exists:
                    # Overwrite the existing item with the new item
                    existing_data[i] = new_item
                    break
    print("Overwriting old data with new data")
    return existing_data


def _write_to(data, filename):
    # Write to a .csv file
    with open(filename, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(data)


def main():
    # TODO: Should probably have the location name in the filename
    filename = f".{constants.OUTPUT_FOLDER}/vnp46a1_image_created_times.csv"
    header_row = ["date", "start_time", "end_time", "spread_time"]

    file_already_exists = os.path.exists(filename)

    if file_already_exists:
        new_data = _merge_new_and_existing_data(filename)
        # remove the header to allow sorting
        new_data_without_header = new_data[1:]
        sorted_data = sorted(new_data_without_header, key=lambda row: parse_date(row[0]))

        # Add the header row back to the sorted rows
        sorted_data.insert(0, header_row)
        _write_to(sorted_data, filename)
    else:
        # Write new data
        new_data = _get_vnp46a1_time_data()
        sorted_data = sorted(new_data, key=lambda row: parse_date(row[0]))

        data = [header_row, *sorted_data]
        _write_to(data, filename)

    print(f"The data has been written to {filename}.")


if __name__ == "__main__":
    main()
