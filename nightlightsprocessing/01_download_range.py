#!/usr/bin/env python3

# Taken and modified from:
# https://ladsweb.modaps.eosdis.nasa.gov/tools-and-services/data-download-scripts/#python
# The script can be run to download different datasets from ladsweb.modaps.eosdis.nasa.gov

import argparse
import os
import os.path
import sys
import asyncio
from . import helpers

################################################################################

# Variables
# The following can be updated depending on your donwload requirements
DATASET = "VNP46A1"
TILE_DESCRIPTOR = "h26v06"
YEAR = 2019
START_DAY_NUMBER = 0  # 0 = 1st Jan
END_DAY_NUMBER = 365  # 365 = 31st Dec
DESC = "This script will recursively download all files if they don't exist from a LAADS URL and will store them to the specified path"

# Constants
# IMPORTANT: This script should only be used with the source destination below:
SOURCE_URL = "https://ladsweb.modaps.eosdis.nasa.gov/archive/allData/5000"

################################################################################


async def _download_tile_for_days(destination, token):
    tasks = []

    for i in range(START_DAY_NUMBER, END_DAY_NUMBER + 1):  # 365 = 31st Dec
        url = f"{SOURCE_URL}/{DATASET}/{YEAR}/{i:03}"
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
