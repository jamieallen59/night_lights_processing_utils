#!/usr/bin/env python3

# Taken and modified from:
# https://ladsweb.modaps.eosdis.nasa.gov/tools-and-services/data-download-scripts/#python
# The script can be run to download different datasets from ladsweb.modaps.eosdis.nasa.gov
import os
import glob
import argparse
import sys
import asyncio
import csv
from . import helpers
from . import constants


# you will need to replace the following line with the location of a
# python web client library that can make HTTPS requests to an IP address.
USERAGENT = "tis/download.py_1.0--" + sys.version.replace("\n", "").replace("\r", "")
# The following can be updated depending on your donwload requirements
DATASET = "VNP46A2"
DESC = "This script will recursively download all files if they don't exist from a LAADS URL and will store them to the specified path"
# IMPORTANT: This script should only be used with the source destination below:
SOURCE_URL = "https://ladsweb.modaps.eosdis.nasa.gov/archive/allData/5000"

################################################################################


def check_for_files(filepath):
    for filepath_object in glob.glob(filepath):
        if os.path.isfile(filepath_object):
            return True

    return False


# TODO: Could make an override to concatenate all files and download all A2 images
# for instances defined in whole 03_create_reliability_dataset folder.
async def _download_tile_for_days(
    destination, token, tile_descriptor, state, location, input_folder, grid_reliability
):
    filename = helpers.get_reliability_dataset_filename(state, location, grid_reliability)

    tasks = []
    # Read csv file
    groundtruth_date_and_time_instances_csvs = helpers.getAllFilesFromFolderWithFilename(input_folder, filename)
    # Should only be one file
    groundtruth_date_and_time_instances_csv = f"{input_folder}/{groundtruth_date_and_time_instances_csvs[0]}"

    with open(groundtruth_date_and_time_instances_csv, "r") as file:
        reader = csv.reader(file)
        data = list(reader)

        for _, date_and_location_instance in enumerate(data[1:]):  # [1:] to skip the header row
            date_day_integer = "{:03d}".format(
                int(date_and_location_instance[3])
            )  # .csv must have this column at position 3
            year_integer = date_and_location_instance[4]  # .csv must have this column at position 4
            print("date_and_location_instance", date_and_location_instance)

            url = f"{SOURCE_URL}/{DATASET}/{year_integer}/{date_day_integer}"

            # Check if we have the file downloaded manually first, to avoid
            # slow get_file_details_for_selected_tile step (which hits ladsweb site).
            # If any local file includes the filter options below, skip the step.
            local_filename_descriptor_check = (
                f"{destination}/{DATASET}.A{year_integer}{date_day_integer}.{tile_descriptor}*"
            )
            if check_for_files(local_filename_descriptor_check):
                print("Skipping, as already downloaded here:", local_filename_descriptor_check)
                continue

            try:
                file_details_to_download = helpers.get_file_details_for_selected_tile(url, token, tile_descriptor)
                name = file_details_to_download["name"]
                path = os.path.join(destination, name)

                if os.path.exists(path):
                    print("Skipping, as already downloaded here:", path)
                else:
                    print(f"starting task using url {url}")
                    tasks.append(asyncio.create_task(helpers.sync(url, destination, token, file_details_to_download)))
            except:
                print("Did not add task for")
                print("Url: ", url)
                print("File details: ", file_details_to_download)

    for task in tasks:
        result = await task
        print(f"finished task {result}")


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
    parser.add_argument("-t", "--token", dest="token", help="Use app token TOK to authenticate", required=True)
    parser.add_argument("-td", "--tile-descriptor", dest="tile_descriptor", help="A MODIS tile value", required=True)
    parser.add_argument(
        "-s",
        "--state",
        dest="state",
        help="State in India",
        required=True,
    )
    parser.add_argument("-l", "--location", dest="location", help="Location within State defined", required=True)
    parser.add_argument("-i", "--input-folder", dest="input_folder", help="Input data directory", required=True)

    args = parser.parse_args(argv[1:])

    for grid_reliability in constants.GRID_RELIABILITIES:
        asyncio.run(
            _download_tile_for_days(
                args.destination,
                args.token,
                args.tile_descriptor,
                args.state,
                args.location,
                args.input_folder,
                grid_reliability,
            )
        )


if __name__ == "__main__":
    try:
        sys.exit(_main(sys.argv))
    except KeyboardInterrupt:
        sys.exit(-1)
