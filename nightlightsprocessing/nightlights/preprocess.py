import os

# import warnings
# import glob
from nightlightsprocessing import helpers as globalHelpers
import sys

from . import constants
from . import process_vnp46a2

# -------------------------DATA PREPROCESSING-------------------------------- #
# Preprocess each HDF5 file (extract bands, mask for fill values,
#  poor-quality, no retrieval, clouds, sea water, fill masked values
#  with NaN, export to GeoTiff)


def _main():
    # Is this better than a helper?
    # hdf5_files = glob.glob(os.path.join(hdf5_input_folder, "*.h5"))
    processed_files = 0
    all_hd5_files = globalHelpers.getAllFilesFromFolderWithFilename(
        constants.H5_INPUT_FOLDER, constants.FILE_TYPE_VNP46A2
    )
    total_files = len(all_hd5_files)
    print("\n")
    print(f"Total files to process: {total_files}\n")

    for filename in all_hd5_files:
        filepath = f"{os.getcwd()}{constants.H5_INPUT_FOLDER}/{filename}"

        process_vnp46a2.process_vnp46a2(filepath)
        processed_files += 1
        print(f"\nPreprocessed file: {processed_files} of {total_files}\n")

    # -------------------------SCRIPT COMPLETION--------------------------------- #
    print("\n")
    print("-" * (18 + len(os.path.basename(__file__))))
    print(f"Completed script: {os.path.basename(__file__)}")
    print("-" * (18 + len(os.path.basename(__file__))))


if __name__ == "__main__":
    try:
        sys.exit(_main())
    except KeyboardInterrupt:
        sys.exit(-1)
