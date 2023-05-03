import os


# https://ladsweb.modaps.eosdis.nasa.gov/learn/how-to-use-laads-daac-post-processing-tools/
# SDS (subdataset processing)
def getSubDataset(name, dataset):
    for subdataset in dataset:
        if name in subdataset[0]:
            return subdataset[0]


def getCommandLineTranslateOptions(rlayer):
    # collect bounding box coordinates
    # Tile numbers are MODIS grid numbers:
    # https://modis-land.gsfc.nasa.gov/MODLAND_grid.html
    HorizontalTileNumber = int(rlayer.GetMetadata_Dict()["HorizontalTileNumber"])
    VerticalTileNumber = int(rlayer.GetMetadata_Dict()["VerticalTileNumber"])

    print("HorizontalTileNumber:", HorizontalTileNumber)
    print("VerticalTileNumber:", VerticalTileNumber)

    WestBoundCoord = (10 * HorizontalTileNumber) - 180
    NorthBoundCoord = 90 - (10 * VerticalTileNumber)
    EastBoundCoord = WestBoundCoord + 10
    SouthBoundCoord = NorthBoundCoord - 10

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


def getAllFilesFrom(folder):
    # Change from root file into given folder
    os.chdir(folder)
    # Get all files in that folder
    allFiles = os.listdir(os.getcwd())
    return allFiles
