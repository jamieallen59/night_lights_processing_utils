#!/usr/bin/env python3

import argparse
import sys
import geopandas as gpd
from shapely.geometry import Point

DESC = "This script creates a shapefile based on the name, logitude and latitude defined."

################################################################################

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
# name = "Mihinpurwa-Bahraich"
# longitude = 81.2478366
# latitude = 28.0676455
# name = "Puraini-Bahraich"
# longitude = 81.5782502
# latitude = 27.2847546

# Couldn't find the below Bahraich locations on Google Maps
# name = "Samariya-Bahraich [Offline]" # can't find on map
# name = "Mahasi-Bahraich" # can't find on map


# --- Sitapur ---
# name = "Anand Nagar Sitapur"
# longitude = 80.699816
# latitude = 27.567880
# name = "Biswan-Sitapur"
# longitude = 80.9592375
# latitude = 27.4929634
# name = "Devtapur-Sitapur[Offline]"
# longitude = 80.5151169
# latitude = 27.7362708
# name = "Dharampur"
# longitude = 80.9526403
# latitude = 27.5566647
# name = "Jhauwa Khurd- Sitapur"
# longitude = 81.0514732
# latitude = 27.5358702
# name = "Jyotishah Alampur- Sitapur [Offline]"
# longitude = 80.6193454
# latitude = 27.4523863
# name = "Kamhira Kathura - Sitapur"
# longitude = 81.1595053
# latitude = 27.6664147
# name = "Kankari- Sitapur"
# longitude = 81.1454263
# latitude = 27.6410101
# name = "Mukimpur-Sitapur [Offline]"
# longitude = 80.8442303
# latitude = 27.2507346
# name = "Muradpur -Sitapur"
# longitude = 80.5103262
# latitude = 27.6067621
# name = "Pahadpur-Sitapur"
# longitude = 80.9189902
# latitude = 27.2192997
# name = "Sahdipur-Sitapur"
# longitude = 80.7414252
# latitude = 27.6667996
# name = "Sidhauli Town"
# longitude = 80.8025674
# latitude = 27.2752201
# name = "Tarinpur Sitapur"
# longitude = 80.5727284
# latitude = 27.5878485
# name = "Thangaon-Sitapur"
# longitude = 81.2267153
# latitude = 27.4566985
name = "Vijay Laxmi Nagar Sitapur"
longitude = 80.6698714
latitude = 27.5655893

# Couldn't find the below Sitapur locations on Google Maps
# name = "Bhadupur Sidhauli"
# name = "Bhattha Mehmoodabad"
# name = "Ichauli- Sitapur"
# name = "Khindaura- Sitapur"
# name = "Manwan- Sitapur"
# name = "Ramuapur Mahmoodabad"
# name = "Sikandarabad - Sitapur"
# name = "Tedwadih- Sitapur"


# Could do:
# Sitapur -
# Barabanki -
# Based on https://energy.prayaspune.org/our-work/article-and-blog/esmi-in-uttar-pradesh

# Others
# Chandauli -
# Faizabad -
# Sultanpur -
# Fatehpur -

# Lucknow discarded due to minimal low reliability instances
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
