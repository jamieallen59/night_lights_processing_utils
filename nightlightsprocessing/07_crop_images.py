#!/usr/bin/env python3

import os
import pandas as pd
import rasterio
import csv
import sys
import argparse
import fiona
import numpy as np
import earthpy.spatial as es
import geopandas as gpd
from . import helpers
from datetime import datetime

# Need to run this file once for each reliability setting e.g. once for LOW and once for HIGH
# https://earthpy.readthedocs.io/en/latest/api/earthpy.spatial.html#earthpy.spatial.crop_image
DESC = "This script crops all images found in the given VNP46A2 file, based on data from previous script outputs"

ALLOWED_NAN_PERCENTAGE = 60
OVER_PERCENTAGE_OF_VALUES_NAN_ERROR = f"Over {ALLOWED_NAN_PERCENTAGE}% of image was NaN, so discarding image"
GRID_RELIABILITIES = ["LOW", "HIGH"]

################################################################################


def get_clipped_vnp46a2(geotiff_path, clip_boundary, location_name, output_folder, grid_reliability):
    print(f"Started clipping: Clip {geotiff_path} " f"to {location_name} boundary")
    print("Clipping image...")
    # Clip image (return clipped array and new metadata)
    with rasterio.open(geotiff_path) as src:
        cropped_image, cropped_metadata = es.crop_image(raster=src, geoms=clip_boundary)

    export_name = helpers.get_tif_to_clipped_export_name(
        filepath=geotiff_path, location_name=location_name, reliability=grid_reliability
    )

    print(f"Completed clipping: Clip {os.path.basename(geotiff_path)} " f"to {location_name} boundary\n")
    return cropped_image[0], cropped_metadata, export_name


def crop_images(
    reliability_dataset_input_folder,
    vnp46a2_tif_input_folder,
    shapefile_input_folder,
    destination,
    buffer,
    state,
    location,
):
    failed_clippings = 0
    low_over_percentage_nan_images = 0
    high_over_percentage_nan_images = 0
    low_successful_clippings = 0
    high_successful_clippings = 0
    images_count = 0

    for grid_reliability in GRID_RELIABILITIES:
        filename = helpers.get_reliability_dataset_filename(state, location, grid_reliability)
        # Loop through date and time instances
        # Read csv file
        groundtruth_date_and_time_instances_csvs = helpers.getAllFilesFromFolderWithFilename(
            reliability_dataset_input_folder, filename
        )
        # Should only be one file
        groundtruth_date_and_time_instances_csv = (
            f"{reliability_dataset_input_folder}/{groundtruth_date_and_time_instances_csvs[0]}"
        )

        with open(groundtruth_date_and_time_instances_csv, "r") as file:
            reader = csv.reader(file)
            data = list(reader)

            for _, date_and_location_instance in enumerate(data[1:]):  # [1:] to skip the header row
                images_count += 1
                location_name = date_and_location_instance[1]
                date = date_and_location_instance[2]
                date_day_integer = "{:03d}".format(
                    int(date_and_location_instance[3])
                )  # .csv must have this column at position 3
                year_integer = date_and_location_instance[4]  # .csv must have this column at position 4
                print("date_and_location_instance", date_and_location_instance)

                filename_filter = f"vnp46a2-a{year_integer}{date_day_integer}"

                try:
                    vnp46a2_tif_file_paths = helpers.getAllFilesFromFolderWithFilename(
                        vnp46a2_tif_input_folder, filename_filter
                    )
                    vnp46a2_tif_file_path = f"{vnp46a2_tif_input_folder}/{vnp46a2_tif_file_paths[0]}"  # Should only be one .tif, so take the first one

                    print("vnp46a2_tif_file_path", vnp46a2_tif_file_path)
                    print("location_name", location_name)
                    print("date", date)
                    # Get shape file by location name
                    location_path = f"{shapefile_input_folder}/{location_name}.shp"
                    location_raw = gpd.read_file(location_path)

                    # Change the crs system to 32634 to allow us to set a buffer in metres
                    # https://epsg.io/32634
                    lucknow_unit_metres = location_raw.to_crs(crs=32634)

                    buffer_distance_metres = int(buffer) * 1609.34  # Convert miles to meters

                    # For if you don't want a circular buffer.
                    # # Note cap_style: round = 1, flat = 2, square = 3
                    # buffer = lucknow_unit_metres.buffer(buffer_distance_metres, cap_style = 3)

                    buffered_location = lucknow_unit_metres.buffer(buffer_distance_metres)

                    # Then change it back to allow the crop images
                    # https://epsg.io/4326
                    clip_boundary = buffered_location.to_crs(crs=4326)

                    cropped_image, cropped_metadata, export_name = get_clipped_vnp46a2(
                        vnp46a2_tif_file_path, clip_boundary, location_name, destination, grid_reliability
                    )

                    # Filtering for over a % NaN values
                    nan_count = np.isnan(cropped_image).sum()
                    non_nan_count = cropped_image.size - nan_count
                    nan_percentage = (np.isnan(cropped_image).sum() / cropped_image.size) * 100

                    print("Non-nan count", non_nan_count)
                    print("Nan count", nan_count)
                    print("Nan %", nan_percentage)

                    if nan_percentage > ALLOWED_NAN_PERCENTAGE:
                        raise ValueError(OVER_PERCENTAGE_OF_VALUES_NAN_ERROR)

                    print("Exporting to GeoTiff...", export_name)
                    print("destination: ", destination)
                    print("export_name: ", export_name)
                    print("location_name: ", location_name)

                    # Create the main directory if it doesn't exist
                    if not os.path.exists(destination):
                        os.mkdir(destination)

                    # Create the location sub-directory if it doesn't exist
                    location_sub_directory_pathname = location_name.replace(" ", "-").lower()
                    location_sub_directory_path = os.path.join(destination, location_sub_directory_pathname)
                    if not os.path.exists(location_sub_directory_path):
                        os.mkdir(location_sub_directory_path)

                    output_path = os.path.join(location_sub_directory_path, export_name)
                    print("Output path: ", output_path)

                    helpers.export_array(
                        array=cropped_image,
                        output_path=output_path,
                        metadata=cropped_metadata,
                    )

                    if grid_reliability == "LOW":
                        low_successful_clippings += 1
                    elif grid_reliability == "HIGH":
                        high_successful_clippings += 1

                except RuntimeError as e:
                    failed_clippings += 1
                    print("Failed to read image:", e)
                except fiona.errors.DriverError as e:
                    failed_clippings += 1
                    print("Failed to read shapefile image", e)
                except ValueError as e:
                    if str(e) == OVER_PERCENTAGE_OF_VALUES_NAN_ERROR:
                        if grid_reliability == "LOW":
                            low_over_percentage_nan_images += 1
                        elif grid_reliability == "HIGH":
                            high_over_percentage_nan_images += 1
                        print("ValueError", e)
                    else:
                        # Should never hit this
                        failed_clippings += 1
                        print("WARNING: Other Value Error", e)
                except Exception as error:
                    failed_clippings += 1
                    print(f"Clipping failed: {error}\n")

    print("----- End clippings -----")
    print("All images count: ", len(data[1:]))
    print("Failed clippings count: ", failed_clippings)
    print("Successful LOW clippings count: ", low_successful_clippings)
    print("Successful HIGH clippings count: ", high_successful_clippings)
    print(f"LOW over >{ALLOWED_NAN_PERCENTAGE}% nan count: ", low_over_percentage_nan_images)
    print(f"HIGH over >{ALLOWED_NAN_PERCENTAGE}% nan count: ", high_over_percentage_nan_images)


def non_filtered_sub_location_crop_images(
    reliability_dataset_input_folder,
    vnp46a2_tif_input_folder,
    shapefile_input_folder,
    buffer,
    state,
    location,
    sub_location,
    sub_location_dataset_path,
):
    data = pd.read_csv(sub_location_dataset_path)

    # Get all unique dates from the 'Date' column
    # data["Date"] = pd.to_datetime(data["Date"])
    unique_dates = pd.unique(data["Date"])

    for date in unique_dates:
        date_datetime = datetime.strptime(date, "%Y-%m-%d")

        date_day_integer = "{:03d}".format(date_datetime.timetuple().tm_yday)
        year_integer = date_datetime.year

        filename_filter = f"vnp46a2-a{year_integer}{date_day_integer}"

        try:
            vnp46a2_tif_file_paths = helpers.getAllFilesFromFolderWithFilename(
                vnp46a2_tif_input_folder, filename_filter
            )
            vnp46a2_tif_file_path = f"{vnp46a2_tif_input_folder}/{vnp46a2_tif_file_paths[0]}"  # Should only be one .tif, so take the first one
            location_name = sub_location
            print("Reading from: ", vnp46a2_tif_file_path)
            print("At:", location_name)
            print("On:", date)
            # Get shape file by location name
            location_path = f"{shapefile_input_folder}/{location_name}.shp"
            location_raw = gpd.read_file(location_path)
            # Change the crs system to 32634 to allow us to set a buffer in metres
            # https://epsg.io/32634
            lucknow_unit_metres = location_raw.to_crs(crs=32634)

            buffer_distance_metres = int(buffer) * 1609.34  # Convert miles to meters

            # For if you don't want a circular buffer.
            # # Note cap_style: round = 1, flat = 2, square = 3
            # buffer = lucknow_unit_metres.buffer(buffer_distance_metres, cap_style = 3)

            buffered_location = lucknow_unit_metres.buffer(buffer_distance_metres)

            # Then change it back to allow the crop images
            # https://epsg.io/4326
            clip_boundary = buffered_location.to_crs(crs=4326)

            def _get_base_filepath(filepath):
                return f"{os.path.basename(filepath)[:-3].lower().replace('.', '-')}"

            base_filepath = _get_base_filepath(vnp46a2_tif_file_path)
            export_name = f"{_get_base_filepath(vnp46a2_tif_file_path)}clipped-{location_name}.tif"

            with rasterio.open(vnp46a2_tif_file_path) as src:
                cropped_image, cropped_metadata = es.crop_image(raster=src, geoms=clip_boundary)

            cropped_image = cropped_image[0]

            # Nan filtering
            nan_count = np.isnan(cropped_image).sum()
            non_nan_count = cropped_image.size - nan_count
            nan_percentage = (np.isnan(cropped_image).sum() / cropped_image.size) * 100

            print(cropped_image)
            print("Non-nan count", non_nan_count)
            print("Nan count", nan_count)
            print("Nan %", nan_percentage)

            if nan_percentage > ALLOWED_NAN_PERCENTAGE:
                raise ValueError(OVER_PERCENTAGE_OF_VALUES_NAN_ERROR)

            destination = f"./data/07-cropped-images/{location_name}-all-buffer-{buffer}-miles"

            if not os.path.exists(destination):
                os.mkdir(destination)

            output_path = os.path.join(destination, export_name)
            print("Output path: ", output_path)

            helpers.export_array(
                array=cropped_image,
                output_path=output_path,
                metadata=cropped_metadata,
            )

            print(
                f"Completed clipping: Clip {os.path.basename(vnp46a2_tif_file_path)} " f"to {location_name} boundary\n"
            )

        except ValueError as e:
            if str(e) == OVER_PERCENTAGE_OF_VALUES_NAN_ERROR:
                print("ValueError", e)
            else:
                # Should never hit this
                print("WARNING: Other Value Error", e)
        except Exception as error:
            print(f"Clipping failed: {error}\n")


################################################################################


def _main(argv):
    parser = argparse.ArgumentParser(prog=argv[0], description=DESC)
    parser.add_argument(
        "-r",
        "--reliability-dataset-input-folder",
        dest="reliability_dataset_input_folder",
        help="The directory of your reliability datasets",
        required=True,
    )
    parser.add_argument(
        "-t",
        "--vnp46a2-tif-input-folder",
        dest="vnp46a2_tif_input_folder",
        help="The directory of your VNP46A2 processed .tif files",
        required=True,
    )
    parser.add_argument(
        "-si",
        "--shapefile-input-folder",
        dest="shapefile_input_folder",
        help="The directory of your shapefiles",
        required=True,
    )
    parser.add_argument(
        "-d",
        "--destination",
        dest="destination",
        help="Store directory structure in DIR",
        required=True,
    )
    parser.add_argument(
        "-b",
        "--buffer",
        dest="buffer",
        help="Distance in miles of the buffer around locations",
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

    crop_images(
        args.reliability_dataset_input_folder,
        args.vnp46a2_tif_input_folder,
        args.shapefile_input_folder,
        args.destination,
        args.buffer,
        args.state,
        args.location,
    )

    # sub_location = "Huzurpur-Bahraich"
    # sub_location_dataset_path = "./data/03-reliability-datasets/Huzurpur-Bahraich-all.csv"
    # non_filtered_sub_location_crop_images(
    #     args.reliability_dataset_input_folder,
    #     args.vnp46a2_tif_input_folder,
    #     args.shapefile_input_folder,
    #     args.buffer,
    #     args.state,
    #     args.location,
    #     sub_location,
    #     sub_location_dataset_path,
    # )


if __name__ == "__main__":
    try:
        sys.exit(_main(sys.argv))
    except KeyboardInterrupt:
        sys.exit(-1)
