import os
import rasterio
from datetime import datetime


# https://ladsweb.modaps.eosdis.nasa.gov/learn/how-to-use-laads-daac-post-processing-tools/
# SDS (subdataset processing)
def getSubDataset(name, dataset):
    for subdataset in dataset:
        if name in subdataset[0]:
            return subdataset[0]


# Should probably move out of here
def getCommandLineTranslateOptions(dataset):
    # collect bounding box coordinates
    # Tile numbers are MODIS grid numbers:
    # https://modis-land.gsfc.nasa.gov/MODLAND_grid.html
    HorizontalTileNumber = int(dataset.GetMetadata_Dict()["HorizontalTileNumber"])
    VerticalTileNumber = int(dataset.GetMetadata_Dict()["VerticalTileNumber"])

    print("HorizontalTileNumber:", HorizontalTileNumber)
    print("VerticalTileNumber:", VerticalTileNumber)

    WestBoundCoord = (10 * HorizontalTileNumber) - 180
    NorthBoundCoord = 90 - (10 * VerticalTileNumber)
    EastBoundCoord = WestBoundCoord + 10
    SouthBoundCoord = NorthBoundCoord - 10

    print("WestBoundCoord:", WestBoundCoord)
    print("NorthBoundCoord:", NorthBoundCoord)
    print("EastBoundCoord:", EastBoundCoord)
    print("SouthBoundCoord:", SouthBoundCoord)

    WestBoundCoord = 74.33
    NorthBoundCoord = 30
    EastBoundCoord = 92
    SouthBoundCoord = 20

    # Must be specified to work with Google Earth
    # https://gdal.org/programs/gdal_translate.html#cmdoption-gdal_translate-a_srs
    EPSG = "-a_srs EPSG:4326"  # WGS84

    # Must be specified to work with Google Earth
    # https://gdal.org/programs/gdal_translate.html#cmdoption-gdal_translate-a_ullr
    GEOREFERENCED_BOUNDS = (
        " -a_ullr "
        + str(WestBoundCoord)
        + " "
        + str(NorthBoundCoord)
        + " "
        + str(EastBoundCoord)
        + " "
        + str(SouthBoundCoord)
    )

    return EPSG + GEOREFERENCED_BOUNDS


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


def get_datetime_from_julian_date(julian_date):
    year = int(julian_date[:4])
    day_of_year = int(julian_date[4:])

    # Create a datetime object using the year and day of year
    full_datetime = datetime.strptime(f"{year}-{day_of_year}", "%Y-%j")

    date_only = full_datetime.date()

    return date_only
