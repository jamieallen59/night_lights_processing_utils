#!/usr/bin/env python3

import argparse
import sys
import os
import csv
import geopandas as gpd
from shapely.geometry import Point
from .get_ESMI_location_information import get_ESMI_location_information
import requests
from . import helpers

DESC = "This script creates a shapefile based on the name, logitude and latitude defined."

# TODO: use centralised column names
LOCATION_NAME_COLUMN = "Location name"

################################################################################

# Based on https://energy.prayaspune.org/our-work/article-and-blog/esmi-in-uttar-pradesh
# Do:
# Sitapur
# Barabanki
# Bahraich

# Others potential
# Chandauli -
# Faizabad -
# Sultanpur -
# Fatehpur -


def get_location_coordinates_already_exist(csv_destination, location_name):
    location_row = None
    file_exists = os.path.exists(csv_destination)
    print(f"Location '{location_name}' already exists: ", file_exists)
    if file_exists:
        with open(csv_destination, "r", newline="") as file:
            data = list(csv.reader(file))

            for row in data[1:]:  # Skip header row
                if row[0] == location_name:
                    location_row = row

    return location_row


def get_location_coordinates_from_api(location_name, location, state, google_maps_geocoding_api_key):
    print("Getting location from API")
    location_name_edit = location_name.replace("[Offline]", "")
    location_name_edit = location_name_edit.replace(location, "")
    location_name_edit = location_name_edit.replace("-", "")
    search_location_name = f"{location_name_edit}, {location}, {state}"

    url = f"https://maps.googleapis.com/maps/api/geocode/json?address={search_location_name}&key={google_maps_geocoding_api_key}"
    response = requests.get(url)
    data = response.json()

    # Extract the coordinates from the API response
    if data["status"] == "OK":
        results = data["results"]
        if results:
            first_result = results[0]
            result_geometry = first_result["geometry"]
            result_location = result_geometry["location"]
            latitude = result_location["lat"]
            longitude = result_location["lng"]
            print(f"Latitude: {latitude}")
            print(f"Longitude: {longitude}")
            return [location_name, longitude, latitude]
    else:
        print("Geocoding failed. Error:", data["status"])


def write_shapefiles(destination, location_coordinate_data):
    for location_coordinate_row in location_coordinate_data:
        print("location_coordinate_row", location_coordinate_row)
        new_location_name = location_coordinate_row[0]
        new_logitude = location_coordinate_row[1]
        new_latitude = location_coordinate_row[2]

        point = Point(new_logitude, new_latitude)
        data = gpd.GeoDataFrame(geometry=[point])
        data.crs = "EPSG:4326"  # For example, using WGS84 CRS

        output_file = f"{destination}/{new_location_name}.shp"
        data.to_file(output_file)


def create_shapefiles(destination, google_maps_geocoding_api_key, ground_truth_input_folder, state, location):
    header_row = [LOCATION_NAME_COLUMN, "longitude", "latitude"]
    location_coordinate_data = []
    csv_destination = f"{destination}/{location}_coordinate_data.csv"

    location_names = get_ESMI_location_information(ground_truth_input_folder, state, location)[LOCATION_NAME_COLUMN]
    print("Finding location names...")
    for location_name in location_names:
        # Check if exists alreadu in csv
        location_row = get_location_coordinates_already_exist(csv_destination, location_name)
        location_row_already_exists = location_row is not None

        if location_row_already_exists:
            location_coordinate_data.append(location_row)
        else:
            location_coordinates = get_location_coordinates_from_api(
                location_name, location, state, google_maps_geocoding_api_key
            )
            location_coordinate_data.append(location_coordinates)

    # Write to csv to avoid needing API calls every time these are generated
    data = [header_row, *location_coordinate_data]
    helpers.write_to_csv(data, csv_destination)

    print(f"The location coordinate csv has been written to {csv_destination}.")
    print(f"The data has been written to {destination}.")

    write_shapefiles(destination, location_coordinate_data)


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
    parser.add_argument(
        "-g",
        "--google-maps-geocoding-api-key",
        dest="google_maps_geocoding_api_key",
        help="Google Maps API key",
        required=True,
    )
    parser.add_argument(
        "-i",
        "--ground-truth-input-folder",
        dest="ground_truth_input_folder",
        help="Ground truth input data directory",
        required=True,
    )

    parser.add_argument(
        "-s",
        "--state",
        dest="state",
        help="State in India",
        required=True,
    )
    parser.add_argument("-l", "--location", dest="location", help="Location within State defined", required=True)

    args = parser.parse_args(argv[1:])

    create_shapefiles(
        args.destination, args.google_maps_geocoding_api_key, args.ground_truth_input_folder, args.state, args.location
    )


if __name__ == "__main__":
    try:
        sys.exit(_main(sys.argv))
    except KeyboardInterrupt:
        sys.exit(-1)
