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
# name = "Vijay Laxmi Nagar Sitapur"
# longitude = 80.6698714
# latitude = 27.5655893

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


# GET https://maps.googleapis.com/maps/api/geocode/json?address=<YOUR_LOCATION>&key=<YOUR_API_KEY>


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


def create_shapefile(destination, google_maps_geocoding_api_key, ground_truth_input_folder, state, location):
    # Pass in state and location
    # Get location info from ESMI location information
    # Use geocoding api to create a dataframe with locations + coordinates
    # Write it to CSV
    # Check for locations in CSV before trying to get them again
    header_row = ["Location name", "longitude", "latitude"]
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

    create_shapefile(
        args.destination, args.google_maps_geocoding_api_key, args.ground_truth_input_folder, args.state, args.location
    )


if __name__ == "__main__":
    try:
        sys.exit(_main(sys.argv))
    except KeyboardInterrupt:
        sys.exit(-1)

# AFTER API
# location_coordinate_row ['Anand Nagar Sitapur', 80.6789519, 27.5680156]
# location_coordinate_row ['Biswan-Sitapur', 80.9964806, 27.4937578]
# location_coordinate_row ['Devtapur-Sitapur[Offline]', 81.0947808, 27.5656198]
# location_coordinate_row ['Ichauli- Sitapur', 80.14425399999999, 25.5707042]
# location_coordinate_row ['Jhauwa Khurd- Sitapur', 81.07684549999999, 27.5368173]
# location_coordinate_row ['Jyotishah Alampur- Sitapur [Offline]', 80.8599193, 27.3727844]
# location_coordinate_row ['Kamhira Kathura - Sitapur', 81.1719936, 27.6636778]
# location_coordinate_row ['Kankari- Sitapur', 81.1830178, 27.6411061]
# location_coordinate_row ['Khindaura- Sitapur', 80.6789519, 27.5680156]
# location_coordinate_row ['Manwan- Sitapur', 80.6789519, 27.5680156]
# location_coordinate_row ['Mukimpur-Sitapur [Offline]', 80.8571521, 27.2531845]
# location_coordinate_row ['Muradpur -Sitapur', 80.6396287, 27.6039423]
# location_coordinate_row ['Pahadpur-Sitapur', 80.59244489999999, 27.7227898]
# location_coordinate_row ['Sahdipur-Sitapur', 80.7533126, 27.6668271]
# location_coordinate_row ['Sikandarabad - Sitapur', 77.6954889, 28.4511246]
# location_coordinate_row ['Tarinpur Sitapur', 80.6665045, 27.5745025]
# location_coordinate_row ['Tedwadih- Sitapur', 80.6789519, 27.5680156]
# location_coordinate_row ['Thangaon-Sitapur', 81.2436227, 27.4586356]
# location_coordinate_row ['Vijay Laxmi Nagar Sitapur', 80.67933939999999, 27.5660019]

# AFTER csv
# location_coordinate_row {'Location name': 'Anand Nagar Sitapur', 'longitude': '80.6789519', 'latitude': '27.5680156'}
# location_coordinate_row {'Location name': 'Biswan-Sitapur', 'longitude': '80.9964806', 'latitude': '27.4937578'}
# location_coordinate_row {'Location name': 'Devtapur-Sitapur[Offline]', 'longitude': '81.0947808', 'latitude': '27.5656198'}
# location_coordinate_row {'Location name': 'Ichauli- Sitapur', 'longitude': '80.14425399999999', 'latitude': '25.5707042'}
# location_coordinate_row {'Location name': 'Jhauwa Khurd- Sitapur', 'longitude': '81.07684549999999', 'latitude': '27.5368173'}
# location_coordinate_row {'Location name': 'Jyotishah Alampur- Sitapur [Offline]', 'longitude': '80.8599193', 'latitude': '27.3727844'}
# location_coordinate_row {'Location name': 'Kamhira Kathura - Sitapur', 'longitude': '81.1719936', 'latitude': '27.6636778'}
# location_coordinate_row {'Location name': 'Kankari- Sitapur', 'longitude': '81.1830178', 'latitude': '27.6411061'}
# location_coordinate_row {'Location name': 'Khindaura- Sitapur', 'longitude': '80.6789519', 'latitude': '27.5680156'}
# location_coordinate_row {'Location name': 'Manwan- Sitapur', 'longitude': '80.6789519', 'latitude': '27.5680156'}
# location_coordinate_row {'Location name': 'Mukimpur-Sitapur [Offline]', 'longitude': '80.8571521', 'latitude': '27.2531845'}
# location_coordinate_row {'Location name': 'Muradpur -Sitapur', 'longitude': '80.6396287', 'latitude': '27.6039423'}
# location_coordinate_row {'Location name': 'Pahadpur-Sitapur', 'longitude': '80.59244489999999', 'latitude': '27.7227898'}
# location_coordinate_row {'Location name': 'Sahdipur-Sitapur', 'longitude': '80.7533126', 'latitude': '27.6668271'}
# location_coordinate_row {'Location name': 'Sikandarabad - Sitapur', 'longitude': '77.6954889', 'latitude': '28.4511246'}
# location_coordinate_row {'Location name': 'Tarinpur Sitapur', 'longitude': '80.6665045', 'latitude': '27.5745025'}
# location_coordinate_row {'Location name': 'Tedwadih- Sitapur', 'longitude': '80.6789519', 'latitude': '27.5680156'}
# location_coordinate_row {'Location name': 'Thangaon-Sitapur', 'longitude': '81.2436227', 'latitude': '27.4586356'}
# location_coordinate_row {'Location name': 'Vijay Laxmi Nagar Sitapur', 'longitude': '80.67933939999999', 'latitude': '27.5660019'}
