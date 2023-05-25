import os

# TODO: TIDY UP and make relevant for this proj
# import warnings
import glob
import viirs

# Set options
# warnings.simplefilter("ignore")

# -------------------------USER-DEFINED VARIABLES---------------------------- #
# Define path to folder containing input VNP46A2 HDF5 files
hdf5_input_folder = os.path.join("02-raw-data", "hdf", "south-korea", "vnp46a2")

# Defne path to output folder to store exported GeoTiff files
geotiff_output_folder = os.path.join("03-processed-data", "raster", "south-korea", "vnp46a2-grid")

# -------------------------DATA PREPROCESSING-------------------------------- #
# Preprocess each HDF5 file (extract bands, mask for fill values,
#  poor-quality, no retrieval, clouds, sea water, fill masked values
#  with NaN, export to GeoTiff)
hdf5_files = glob.glob(os.path.join(hdf5_input_folder, "*.h5"))
processed_files = 0
total_files = len(hdf5_files)
for hdf5 in hdf5_files:
    viirs.preprocess_vnp46a2(hdf5_path=hdf5, output_folder=geotiff_output_folder)
    processed_files += 1
    print(f"Preprocessed file: {processed_files} of {total_files}\n\n")

# -------------------------SCRIPT COMPLETION--------------------------------- #
print("\n")
print("-" * (18 + len(os.path.basename(__file__))))
print(f"Completed script: {os.path.basename(__file__)}")
print("-" * (18 + len(os.path.basename(__file__))))
