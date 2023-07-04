import os
import rasterio

import csv
import earthpy.spatial as es
import geopandas as gpd
from . import constants
from . import helpers

# https://earthpy.readthedocs.io/en/latest/api/earthpy.spatial.html#earthpy.spatial.crop_image

################################################################################
# Variables
STATE = "Uttar Pradesh"
LOCATION = "Lucknow"
BUFFER_DISTANCE_MILES = 5

################################################################################


def clip_vnp46a2(geotiff_path, clip_boundary, location_name, output_folder):
    print(f"Started clipping: Clip {os.path.basename(geotiff_path)} " f"to {location_name} boundary")
    try:
        print("Clipping image...")
        # Clip image (return clipped array and new metadata)
        with rasterio.open(geotiff_path) as src:
            cropped_image, cropped_metadata = es.crop_image(raster=src, geoms=clip_boundary)

        print("Setting export name...")
        export_name = helpers.get_tif_to_clipped_export_name(filepath=geotiff_path, location_name=location_name)

        print("Exporting to GeoTiff...")
        helpers.export_array(
            array=cropped_image[0],
            output_path=os.path.join(output_folder, export_name),
            metadata=cropped_metadata,
        )
    except Exception as error:
        message = print(f"Clipping failed: {error}\n")
    else:
        message = print(f"Completed clipping: Clip {os.path.basename(geotiff_path)} " f"to {location_name} boundary\n")

    return message


def main():
    # Loop through date and time instances
    filename = f"{constants.VOLTAGE_DATA_FILENAME} - {STATE} - {LOCATION} - filtered unique.csv"
    # Read csv file
    groundtruth_date_and_time_instances_csvs = helpers.getAllFilesFromFolderWithFilename(
        constants.OUTPUT_GROUND_TRUTH_FOLDER, filename
    )
    # Should only be one file
    groundtruth_date_and_time_instances_csv = (
        f".{constants.OUTPUT_GROUND_TRUTH_FOLDER}/{groundtruth_date_and_time_instances_csvs[0]}"
    )

    with open(groundtruth_date_and_time_instances_csv, "r") as file:
        reader = csv.reader(file)
        data = list(reader)

        for _, date_and_location_instance in enumerate(data[1:]):  # [1:] to skip the header row
            location_name = date_and_location_instance[1]
            date = date_and_location_instance[2]
            date_day_integer = "{:03d}".format(
                int(date_and_location_instance[3])
            )  # .csv must have this column at position 3
            year_integer = date_and_location_instance[4]  # .csv must have this column at position 4
            print("date_and_location_instance", date_and_location_instance)

            filename_filter = f"vnp46a2-a{year_integer}{date_day_integer}"

            vnp46a2_tif_file_paths = helpers.getAllFilesFromFolderWithFilename(
                constants.OUTPUT_FOLDER, filename_filter
            )
            vnp46a2_tif_file_path = vnp46a2_tif_file_paths[0]  # Should only be one .tif, so take the first one
            print("vnp46a2_tif_file_path", vnp46a2_tif_file_path)
            print("location_name", location_name)
            print("date", date)

            # Get shape file by location name
            location_path = f"{os.getcwd()}{constants.LOCATION_INPUT_FOLDER}/{location_name}.shp"
            location_raw = gpd.read_file(location_path)

            # Change the crs system to 32634 to allow us to set a buffer in metres
            # https://epsg.io/32634
            lucknow_unit_metres = location_raw.to_crs(crs=32634)

            buffer_distance_metres = BUFFER_DISTANCE_MILES * 1609.34  # Convert miles to meters
            buffered_location = lucknow_unit_metres.buffer(buffer_distance_metres)

            # Then change it back to allow the crop images
            # https://epsg.io/4326
            location = buffered_location.to_crs(crs=4326)

            clip_boundary = location
            output_folder = f"{os.getcwd()}{constants.OUTPUT_FOLDER}/clipped"
            clip_vnp46a2(vnp46a2_tif_file_path, clip_boundary, location_name, output_folder)

    # try:
    #     location_path = f"{os.getcwd()}{constants.LOCATION_INPUT_FOLDER}/lucknow-point.shp"
    #     lucknowRaw = gpd.read_file(location_path)

    #     # here we change the crs system to 32634 to allow us to set a buffer in metres
    #     # https://epsg.io/32634
    #     lucknow_unit_metres = lucknowRaw.to_crs(crs=32634)
    #     buffer_distance_miles = 15
    #     buffer_distance_metres = buffer_distance_miles * 1609.34  # Convert miles to meters
    #     buffered_lucknow = lucknow_unit_metres.buffer(buffer_distance_metres)

    #     # Then change it back to allow the crop images
    #     # https://epsg.io/4326
    #     lucknow = buffered_lucknow.to_crs(crs=4326)

    #     geotiff_path = f"{os.getcwd()}{constants.OUTPUT_FOLDER}/vnp46a2-a2014305-h26v06-001-2020214163912.tif"

    #     clip_boundary = lucknow
    #     location_name = "Lucknow"
    #     output_folder = f"{os.getcwd()}{constants.OUTPUT_FOLDER}"
    #     clip_vnp46a2(geotiff_path, clip_boundary, location_name, output_folder)

    # except Exception as error:
    #     message = print(f"Failed: {error}\n")
    # else:
    #     message = print(f"Completed")

    # return message


# TODO: use crop_all in future so done all at once?
# https://earthpy.readthedocs.io/en/latest/api/earthpy.spatial.html#earthpy.spatial.crop_all
if __name__ == "__main__":
    main()

    # input("Press ENTER to exit")
