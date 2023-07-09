#!/usr/bin/env python3

import argparse
import sys
import geopandas as gpd
from shapely.geometry import Point

DESC = "This script creates a shapefile based on the name, logitude and latitude defined."

################################################################################

# --- Lucknow ---
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

# name = "Vikas Nagar- Lucknow [Offline]"
# longitude = 80.9540233
# latitude = 26.8976618

# --- Bahraich ---
# name = "Bahadurpur-Bahraich"
# longitude = 81.5933001
# latitude = 27.546632
# name = "Bardaha-Bahraich"
# longitude = 81.3921704
# latitude = 27.8080485
# name = "Huzurpur-Bahraich"
# longitude = 81.6586352
# latitude = 27.3481097
# name = "Jagatapur-Bahraich"
# longitude = 81.5969203
# latitude = 27.5452547
# name = "Kanjaya-Bahraich"
# longitude = 81.6582102
# latitude = 27.2606896
# name = "Kurmaura-Bahraich"
# longitude = 81.5929152
# latitude = 27.1746395
# name = "Mahasi-Bahraich" # can't find on map
# longitude = 00
# latitude = 00
# name = "Mihinpurwa-Bahraich"
# longitude = 81.2478366
# latitude = 28.0676455
name = "Puraini-Bahraich"
longitude = 81.5782502
latitude = 27.2847546
# name = "Samariya-Bahraich [Offline]" # can't find on map
# longitude = 00
# latitude = 00


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

    args = parser.parse_args(argv[1:])

    create_shapefile(args.destination)


if __name__ == "__main__":
    try:
        sys.exit(_main(sys.argv))
    except KeyboardInterrupt:
        sys.exit(-1)
