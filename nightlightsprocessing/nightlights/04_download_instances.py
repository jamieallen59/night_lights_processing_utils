#!/usr/bin/env python3

# Taken and modified from:
# https://ladsweb.modaps.eosdis.nasa.gov/tools-and-services/data-download-scripts/#python
# The script can be run to download different datasets from ladsweb.modaps.eosdis.nasa.gov

import argparse
import os
import os.path
import sys
import asyncio
import csv
from nightlightsprocessing import helpers as globalHelpers
from . import helpers
from . import constants


################################################################################

# Variables
# you will need to replace the following line with the location of a
# python web client library that can make HTTPS requests to an IP address.
USERAGENT = "tis/download.py_1.0--" + sys.version.replace("\n", "").replace("\r", "")
# The following can be updated depending on your donwload requirements
DATASET = "VNP46A2"
TILE_DESCRIPTOR = "h26v06"
DESC = "This script will recursively download all files if they don't exist from a LAADS URL and will store them to the specified path"
OUTPUT_FOLDER = constants.OUTPUT_GROUND_TRUTH_FOLDER
STATE = "Uttar Pradesh"
LOCATION = "Lucknow"

# Constants
# IMPORTANT: This script should only be used with the source destination below:
SOURCE_URL = "https://ladsweb.modaps.eosdis.nasa.gov/archive/allData/5000"
VOLTAGE_DATA_FILENAME = "ESMI minute-wise voltage data"

################################################################################


async def _download_tile_for_days(destination, token):
    tasks = []
    filename = f"{VOLTAGE_DATA_FILENAME} - {STATE} - {LOCATION} - filtered unique.csv"
    # Read csv file
    groundtruth_date_and_time_instances_csvs = globalHelpers.getAllFilesFromFolderWithFilename(OUTPUT_FOLDER, filename)
    # Should only be one file
    groundtruth_date_and_time_instances_csv = f".{OUTPUT_FOLDER}/{groundtruth_date_and_time_instances_csvs[0]}"

    with open(groundtruth_date_and_time_instances_csv, "r") as file:
        reader = csv.reader(file)
        data = list(reader)

        for _, date_and_location_instance in enumerate(data[1:]):  # [1:] to skip the header row
            date_day_integer = date_and_location_instance[3]  # .csv must have this column at position 3
            year_integer = date_and_location_instance[4]  # .csv must have this column at position 4
            print("date_and_location_instance", date_and_location_instance)

            url = f"{SOURCE_URL}/{DATASET}/{year_integer}/{date_day_integer:03}"
            print(f"starting task using url {url}")
            file_details_to_download = helpers.get_file_details_for_selected_tile(url, token, TILE_DESCRIPTOR)

            tasks.append(asyncio.create_task(helpers.sync(url, destination, token, file_details_to_download)))

    for task in tasks:
        result = await task
        print(f"finished task {result}")


def _main(argv):
    parser = argparse.ArgumentParser(prog=argv[0], description=DESC)
    parser.add_argument(
        "-d",
        "--destination",
        dest="destination",
        metavar="DIR",
        help="Store directory structure in DIR",
        required=True,
    )
    parser.add_argument(
        "-t", "--token", dest="token", metavar="TOK", help="Use app token TOK to authenticate", required=True
    )
    args = parser.parse_args(argv[1:])
    # If the directory doesn't exist, create the path
    if not os.path.exists(args.destination):
        os.makedirs(args.destination)

    asyncio.run(_download_tile_for_days(args.destination, args.token))


if __name__ == "__main__":
    try:
        sys.exit(_main(sys.argv))
    except KeyboardInterrupt:
        sys.exit(-1)
