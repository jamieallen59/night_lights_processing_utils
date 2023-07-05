#!/usr/bin/env python3

import argparse
import sys
import geopandas as gpd
from shapely.geometry import Point

DESC = "This script creates a shapefile based on the name, logitude and latitude defined."

################################################################################

# name = "Aishbagh-Lucknow"
# longitude = 80.9039054
# latitude = 26.8412685

# name = "Alambagh- Lucknow"
# longitude = 80.8770509
# latitude = 26.8081419

# name = "Aliganj- Lucknow"
# longitude = 80.9362831
# latitude = 26.8975983

# name = "Ashiana-Lucknow [Offline]"
# longitude = 80.9047309
# latitude = 26.7869682

name = "Vikas Nagar- Lucknow [Offline]"
longitude = 80.9540233
latitude = 26.8976618


def create_shapefile(destination):
    point = Point(longitude, latitude)
    data = gpd.GeoDataFrame(geometry=[point])
    data.crs = "EPSG:4326"  # For example, using WGS84 CRS

    output_file = f"{destination}/{name}.shp"
    data.to_file(output_file)


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

    create_shapefile(args.destination)


if __name__ == "__main__":
    try:
        sys.exit(_main(sys.argv))
    except KeyboardInterrupt:
        sys.exit(-1)
