# Variables
STATE = "Uttar Pradesh"
LOCATION = "Lucknow"
TOKEN = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJBUFMgT0F1dGgyIEF1dGhlbnRpY2F0b3IiLCJpYXQiOjE2ODU1MzQ3MjUsIm5iZiI6MTY4NTUzNDcyNSwiZXhwIjoxNzAxMDg2NzI1LCJ1aWQiOiJqYW1pZWFsbGVuNTkiLCJlbWFpbF9hZGRyZXNzIjoiamFtaWVhbGxlbjU5QGdtYWlsLmNvbSIsInRva2VuQ3JlYXRvciI6ImphbWllYWxsZW41OSJ9.Hh5uHl3N5TWKblonqNT1-UwsdIgYNbwCYLmPTme_wxw"
# Data file paths
OO_GROUND_TRUTH_PATH = "./data/00-ground-truth"
O1_VNP46A1_H5_PATH = "./data/01-VNP46A1-h5"
O2_VNP46A1_IMAGE_CREATED_TIMES_PATH = "./data/02-VNP46A1-image-created-times"
O3_RELIABILITY_DATASETS_PATH = "./data/03-reliability-datasets"
O4_VNP46A2_H5_PATH = "./data/04-VNP46A2-h5"
O5_PROCESSED_VNP46A2_IMAGES = "./data/05-processed-VNP46A2-images"
O6_LOCATION_SHAPEFILES = "./data/06-location-shapefiles"
O7_CROPPED_IMAGES = "./data/07-cropped-images"

TILE_DESCRIPTOR = "h26v06" # A MODIS tile descriptor
GRID_RELIABILITY = "HIGH" # Either LOW or HIGH
BUFFER_DISTANCE_MILES = "1"


################################################################################

setup:
	pip install -r requirements.txt

test:
	python3 -m unittest discover tests

clean:
	rm -rf __pycache__
	rm -rf venv

.PHONY: test

################################################################################

# Main functional scripts

# TODO: pass e.g. location in as a variable here to all scripts that need it.
# TODO: extract token to and env variable
# 01 download range of A1 images to find time spreads in question
01-download-range:
		python3 -m nightlightsprocessing.01_download_VNP46A1_range --destination "${O1_VNP46A1_H5_PATH}" --token "${TOKEN}" --tile-descriptor ${TILE_DESCRIPTOR} --start-day "0" --end-day "364" --year "2019"

# 02 create time of capture spreads dataset based on downloaded A1 images UTC_time band
02-create-image-taken-times-dataset:
		python3 -m nightlightsprocessing.02_create_image_taken_times_dataset --input_folder "${O1_VNP46A1_H5_PATH}" --destination "${O2_VNP46A1_IMAGE_CREATED_TIMES_PATH}/vnp46a1_image_created_times" 

# 03 Create ground truth dataset of low reliability date instances based on time of capture spreads
# Note: may want to do this for multiple locations in your Area of Interest
03-create-reliability-dataset:
	python3 -m nightlightsprocessing.03_create_reliability_dataset --state "${STATE}" --location "${LOCATION}" --input_folder "${OO_GROUND_TRUTH_PATH}" --destination "${O3_RELIABILITY_DATASETS_PATH}"

# 04 dowload A2 images based on low reliability ground truth dataset
04-download-VNP46A2-instances:
		python3 -m nightlightsprocessing.04_download_VNP46A2_instances --destination "${O4_VNP46A2_H5_PATH}" --token "${TOKEN}" --tile-descriptor ${TILE_DESCRIPTOR} --state "${STATE}" --location "${LOCATION}" --grid-reliability "${GRID_RELIABILITY}" --input-folder "${O3_RELIABILITY_DATASETS_PATH}"

# 05 process all A2 images with the given masks
05-process-VNP46A2-images:
		python3 -m nightlightsprocessing.05_process_VNP46A2_images --input_folder "${O4_VNP46A2_H5_PATH}" --destination "${O3_RELIABILITY_DATASETS_PATH}"

# 06 create .shp files for your areas
06-create-shapefile:
		python3 -m nightlightsprocessing.06_create_shapefile --destination "${O6_LOCATION_SHAPEFILES}"

# 07 crop processed images based on ground truth data
07-crop-images:
		python3 -m nightlightsprocessing.07_crop_images --reliability-dataset-input-folder "${O3_RELIABILITY_DATASETS_PATH}" --vnp46a2-tif-input-folder "${O5_PROCESSED_VNP46A2_IMAGES}" --shapefile-input-folder "${O6_LOCATION_SHAPEFILES}" --destination "${O7_CROPPED_IMAGES}/buffer-{BUFFER_DISTANCE_MILES}-miles" --buffer "${BUFFER_DISTANCE_MILES}" --state "${STATE}" --location "${LOCATION}" --grid-reliability "${GRID_RELIABILITY}"


# TODO: should turn all this to one entry point with script arguements (assuming it stays as one package)
# https://stackoverflow.com/questions/57744466/how-to-properly-structure-internal-scripts-in-a-python-project


