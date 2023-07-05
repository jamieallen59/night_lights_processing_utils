#!/usr/bin/env python3

# Taken and modified from:
# https://ladsweb.modaps.eosdis.nasa.gov/tools-and-services/data-download-scripts/#python
# The script can be run to download different datasets from ladsweb.modaps.eosdis.nasa.gov
# This could be used with other datasets, but for this project it's only used to download VNP46A1
# images.

import argparse
import sys
import asyncio
from . import helpers

# IMPORTANT: This script should only be used with the source destination below:
SOURCE_URL = "https://ladsweb.modaps.eosdis.nasa.gov/archive/allData/5000"
DATASET = "VNP46A1"
DESC = "This script will recursively download all files if they don't exist from a LAADS URL and will store them to the specified path"


################################################################################


async def _download_tile_for_days(destination, token, tile_descriptor, start_day, end_day, year):
    tasks = []

    for i in range(int(start_day), int(end_day) + 1):  # 365 = 31st Dec
        index = "{:03d}".format(int(i))
        url = f"{SOURCE_URL}/{DATASET}/{year}/{index}"
        print(f"starting task using url {url}")
        file_details_to_download = helpers.get_file_details_for_selected_tile(url, token, tile_descriptor)

        tasks.append(asyncio.create_task(helpers.sync(url, destination, token, file_details_to_download)))

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
        "-sd", "--start-day", dest="start_day", help="The start day number (between 0-364)", required=True
    )
    parser.add_argument("-ed", "--end-day", dest="end_day", help="The end day number (between 0-364)", required=True)
    parser.add_argument("-y", "--year", dest="year", help="The target download year", required=True)

    args = parser.parse_args(argv[1:])

    asyncio.run(
        _download_tile_for_days(
            args.destination, args.token, args.tile_descriptor, args.start_day, args.end_day, args.year
        )
    )


if __name__ == "__main__":
    try:
        sys.exit(_main(sys.argv))
    except KeyboardInterrupt:
        sys.exit(-1)
