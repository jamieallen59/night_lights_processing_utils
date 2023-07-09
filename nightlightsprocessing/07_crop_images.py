#!/usr/bin/env python3

import os
import rasterio
import csv
import sys
import argparse

import earthpy.spatial as es
import fiona
import geopandas as gpd
from . import helpers

# https://earthpy.readthedocs.io/en/latest/api/earthpy.spatial.html#earthpy.spatial.crop_image

DESC = "This script crops all images found in the given VNP46A2 file, based on data from previous script outputs"

################################################################################


def clip_vnp46a2(geotiff_path, clip_boundary, location_name, output_folder, grid_reliability):
    print(f"Started clipping: Clip {geotiff_path} " f"to {location_name} boundary")
    try:
        print("Clipping image...")
        # Clip image (return clipped array and new metadata)
        with rasterio.open(geotiff_path) as src:
            cropped_image, cropped_metadata = es.crop_image(raster=src, geoms=clip_boundary)

        print("Setting export name...")
        export_name = helpers.get_tif_to_clipped_export_name(
            filepath=geotiff_path, location_name=location_name, reliability=grid_reliability
        )

        print("Exporting to GeoTiff...", export_name)
        output_path = os.path.join(output_folder, export_name)
        print("Output path: ", output_path)
        helpers.export_array(
            array=cropped_image[0],
            output_path=output_path,
            metadata=cropped_metadata,
        )
    except Exception as error:
        message = print(f"Clipping failed: {error}\n")
    else:
        message = print(f"Completed clipping: Clip {os.path.basename(geotiff_path)} " f"to {location_name} boundary\n")

    return message


def crop_images(
    reliability_dataset_input_folder,
    vnp46a2_tif_input_folder,
    shapefile_input_folder,
    destination,
    buffer,
    state,
    location,
    grid_reliability,
):
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
        failed_clippings = 0

        for _, date_and_location_instance in enumerate(data[1:]):  # [1:] to skip the header row
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
                buffered_location = lucknow_unit_metres.buffer(buffer_distance_metres)

                # Then change it back to allow the crop images
                # https://epsg.io/4326
                location = buffered_location.to_crs(crs=4326)

                clip_boundary = location
                clip_vnp46a2(vnp46a2_tif_file_path, clip_boundary, location_name, destination, grid_reliability)
            except RuntimeError as e:
                failed_clippings += 1
                print("Failed to read image:", e)
            except fiona.errors.DriverError as e:
                failed_clippings += 1
                print("Failed to read shapefile image", e)

        print("All images count: ", len(data[1:]))
        print("Failed clippings count: ", failed_clippings)


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
    parser.add_argument(
        "-gr",
        "--grid-reliability",
        dest="grid_reliability",
        help="A value either LOW or HIGH to represent the reliability of the grid",
        required=True,
    )
    args = parser.parse_args(argv[1:])

    crop_images(
        args.reliability_dataset_input_folder,
        args.vnp46a2_tif_input_folder,
        args.shapefile_input_folder,
        args.destination,
        args.buffer,
        args.state,
        args.location,
        args.grid_reliability,
    )


if __name__ == "__main__":
    try:
        sys.exit(_main(sys.argv))
    except KeyboardInterrupt:
        sys.exit(-1)
