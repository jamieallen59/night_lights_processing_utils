#!/usr/bin/env python3

# This file reads all the VNP46A1 files you provide it and reads the
# UTC_Time band of those images.
# See Table 3: https://viirsland.gsfc.nasa.gov/PDF/BlackMarbleUserGuide_v1.2_20220916.pdf
# It creates a .csv file wth the dates, start and end times plus the spread between those times
import argparse
import os
from osgeo import gdal
import rasterio as rio
import re
import sys
from . import helpers
from . import constants
import datetime

FILE_TYPE = "VNP46A1"  # It only works with VNP46A1, as these have the UTC_Time property
SELECTED_DATASET = "UTC_Time"
READ_METHOD = gdal.GA_ReadOnly
DESC = "This file reads all the VNP46A1 files you provide it and reads the UTC_Time band of those images"

################################################################################


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
    # Open top-level dataset, loop through Science Data Sets (subdatasets),
    # and extract specified band
    band = []

    with rio.open(filename) as dataset:
        for science_data_set in dataset.subdatasets:
            if re.search(f"{SELECTED_DATASET}$", science_data_set):
                with rio.open(science_data_set) as src:
                    band = src.read(1)

    return band


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


def _get_vnp46a1_time_data(input_folder):
    all_files = helpers.getAllFilesFromFolderWithFilename(input_folder, FILE_TYPE)
    data = []
    count = 0

    for file in all_files:
        hdf5filepath = f"{os.getcwd()}{input_folder}/{file}"
        try:
            date, start_time, end_time, spread_time = _get_row_values(hdf5filepath)

            data.append([date, start_time, end_time, spread_time])
        except:
            count = count + 1
            print(f"Error reading file: {hdf5filepath}")
    print("total skipped", count)
    return data


def parse_date(date_str):
    if isinstance(date_str, datetime.date):
        return date_str

    try:
        return datetime.datetime.strptime(date_str, constants.DATETIME_FORMAT).date()
    except ValueError as e:
        print(f"Date {date_str} is not in format {format}")


def create_image_taken_times_dataset(input_folder, destination):
    header_row = ["Date", "start_time", "end_time", "start_end_spread_time"]

    new_data = _get_vnp46a1_time_data(input_folder)
    sorted_data = sorted(new_data, key=lambda row: parse_date(row[0]))

    data = [header_row, *sorted_data]
    helpers.write_to_csv(data, destination)

    print(f"The data has been written to {destination}.")


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

    create_image_taken_times_dataset(args.input_folder, args.destination)


if __name__ == "__main__":
    try:
        sys.exit(_main(sys.argv))
    except KeyboardInterrupt:
        sys.exit(-1)
