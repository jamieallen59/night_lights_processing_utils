import os
import rasterio
from datetime import datetime


# https://ladsweb.modaps.eosdis.nasa.gov/learn/how-to-use-laads-daac-post-processing-tools/
# SDS (subdataset processing)
def getSubDataset(name, dataset):
    for subdataset in dataset:
        if name in subdataset[0]:
            return subdataset[0]


def export_array(array, output_path, metadata):
    # Write numpy array to GeoTiff
    try:
        with rasterio.open(output_path, "w", **metadata) as dst:
            dst.write(array, 1)
    except Exception as error:
        output_message = print(f"ERROR: {error}")
    else:
        output_message = print(f"Exported: {os.path.split(output_path)[-1]}")

    return output_message


def _get_base_filepath(filepath):
    return f"{os.path.basename(filepath)[:-3].lower().replace('.', '-')}"


def get_hd5_to_tif_export_name(filepath):
    return f"{_get_base_filepath(filepath)}.tif"


def get_tif_to_clipped_export_name(filepath, location_name):
    image_country = location_name.replace(" ", "-").lower()

    export_name = f"{_get_base_filepath(filepath)}clipped-{image_country}.tif"
    return export_name


def get_datetime_from_julian_date(julian_date):
    year = int(julian_date[:4])
    day_of_year = int(julian_date[4:])

    # Create a datetime object using the year and day of year
    full_datetime = datetime.strptime(f"{year}-{day_of_year}", "%Y-%j")

    date_only = full_datetime.date()

    return date_only
