from nightlightsprocessing import helpers

import os
import rasterio as rio

from . import constants

INPUT_FOLDER = constants.O7_CROPPED_IMAGES


def main():
    clipped_files = helpers.getAllFilesFromFolderWithFilename(INPUT_FOLDER, "clipped")
    # Just one for testing atm
    first_clipped_file = clipped_files[0]
    clipped_file_path = f"{os.getcwd()}{INPUT_FOLDER}/{first_clipped_file}"

    with rio.open(clipped_file_path) as src:
        arr = src.read()  # read all raster values
        print("arr", arr)
        print("arr.shape", arr.shape)


if __name__ == "__main__":
    main()
