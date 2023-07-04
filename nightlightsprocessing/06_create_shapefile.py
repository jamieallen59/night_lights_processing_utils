#!/usr/bin/env python3

import geopandas as gpd
from shapely.geometry import Point

from . import constants

################################################################################
# Variables

OUTPUT_FOLDER = constants.O6_LOCATION_SHAPEFILES
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


def main():
    point = Point(longitude, latitude)
    data = gpd.GeoDataFrame(geometry=[point])
    data.crs = "EPSG:4326"  # For example, using WGS84 CRS

    output_file = f".{OUTPUT_FOLDER}/{name}.shp"
    data.to_file(output_file)


if __name__ == "__main__":
    main()
