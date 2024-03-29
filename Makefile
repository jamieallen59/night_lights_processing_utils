# TOKEN comes from .env. It is your token from the https://urs.earthdata.nasa.gov/ website.
include .env

# Full process
# 03-create-reliability-dataset
# 04-download-VNP46A2-instances
# 05-process-VNP46A2-images
# 06-create-shapefiles
# 07-crop-images
# 08-run-model

# // Then ready to run model

# Variables
# Data file paths
OO_GROUND_TRUTH_PATH = "./data/00-ground-truth"
O1_VNP46A1_H5_PATH = "./data/01-VNP46A1-h5"
O2_VNP46A1_IMAGE_CREATED_TIMES_PATH = "./data/02-VNP46A1-image-created-times"
O3_RELIABILITY_DATASETS_PATH = "./data/03-reliability-datasets"
O4_VNP46A2_H5_PATH = "./data/04-VNP46A2-h5"
O5_PROCESSED_VNP46A2_IMAGES = "./data/05-processed-VNP46A2-images"
O6_LOCATION_SHAPEFILES = "./data/06-location-shapefiles"
O7_CROPPED_IMAGES = "./data/07-cropped-images"

LOW_RELIABILITY_VOLTAGE = 130
HIGH_RELIABILITY_VOLTAGE = 205
STATE = "Uttar Pradesh"
LOCATION = "Varanasi"
TILE_DESCRIPTOR = "h26v06" # A MODIS tile descriptor
GRID_RELIABILITY = "HIGH" # Either LOW or HIGH
BUFFER_DISTANCE_MILES = "5"

# May need:
# - Bahraich
# - Barabanki
# - Kanpur
# - Sitapur
# - Varanasi


################################################################################

# Main functional scripts

# 01 download range of A1 images to find time spreads in question
01-download-range:
		python3 -m nightlightsprocessing.01_download_VNP46A1_range --destination ${O1_VNP46A1_H5_PATH} --token ${TOKEN} --tile-descriptor ${TILE_DESCRIPTOR} --start-day 0 --end-day 364 --year 2019

# 02 create time of capture spreads dataset based on downloaded A1 images UTC_time band
02-create-image-taken-times-dataset:
		python3 -m nightlightsprocessing.02_create_image_taken_times_dataset --input-folder ${O1_VNP46A1_H5_PATH} --destination ${O2_VNP46A1_IMAGE_CREATED_TIMES_PATH}/vnp46a1_image_created_times

# 03 Create ground truth dataset of low reliability date instances based on time of capture spreads
# Note: may want to do this for multiple locations in your Area of Interest
03-create-reliability-dataset:
	python3 -m nightlightsprocessing.03_create_reliability_dataset --state ${STATE} --location ${LOCATION} --input-folder ${OO_GROUND_TRUTH_PATH} --destination ${O3_RELIABILITY_DATASETS_PATH} --low-reliability-voltage ${LOW_RELIABILITY_VOLTAGE} --high-reliability-voltage ${HIGH_RELIABILITY_VOLTAGE}

# 04 dowload A2 images based on low reliability ground truth dataset
04-download-VNP46A2-instances:
		python3 -m nightlightsprocessing.04_download_VNP46A2_instances --destination ${O4_VNP46A2_H5_PATH} --token ${TOKEN} --tile-descriptor ${TILE_DESCRIPTOR} --state ${STATE} --location ${LOCATION} --input-folder ${O3_RELIABILITY_DATASETS_PATH}

# 05 process all A2 images with the given masks
05-process-VNP46A2-images:
		python3 -m nightlightsprocessing.05_process_VNP46A2_images --input-folder ${O4_VNP46A2_H5_PATH} --destination ${O5_PROCESSED_VNP46A2_IMAGES}

# 06 create .shp files for your areas
06-create-shapefiles:
		python3 -m nightlightsprocessing.06_create_shapefiles --destination ${O6_LOCATION_SHAPEFILES} --google-maps-geocoding-api-key ${GOOGLE_MAPS_GEOCODING_API_KEY} --ground-truth-input-folder ${OO_GROUND_TRUTH_PATH} --state ${STATE} --location ${LOCATION}

# 07 crop processed images based on ground truth data
07-crop-images:
		python3 -m nightlightsprocessing.07_crop_images --reliability-dataset-input-folder ${O3_RELIABILITY_DATASETS_PATH} --vnp46a2-tif-input-folder ${O5_PROCESSED_VNP46A2_IMAGES} --shapefile-input-folder ${O6_LOCATION_SHAPEFILES} --destination ${O7_CROPPED_IMAGES}/${LOCATION}-buffer-${BUFFER_DISTANCE_MILES}-miles --buffer ${BUFFER_DISTANCE_MILES} --state ${STATE} --location ${LOCATION}

08-regression-models:
		python3 -m nightlightsprocessing.08_regression_models

09-ml-model:
		python3 -m nightlightsprocessing.09_ml_model

10-plotting:
		python3 -m nightlightsprocessing.10_plotting plot_picket_fence


# TODO: should turn all this to one entry point with script arguements (assuming it stays as one package)
# https://stackoverflow.com/questions/57744466/how-to-properly-structure-internal-scripts-in-a-python-project

# Ancillary

clean-cropped-images:
		python3 -m nightlightsprocessing.clean_cropped_images

################################################################################

setup:
	pip install -r requirements.txt

test:
	python3 -m unittest discover tests

clean:
	rm -rf __pycache__
	rm -rf venv

.PHONY: test
