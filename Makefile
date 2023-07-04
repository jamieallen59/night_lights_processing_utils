setup:
	pip install -r requirements.txt

test:
	python3 -m unittest discover tests

clean:
	rm -rf __pycache__
	rm -rf venv

# 01 download range of A1 images to find time spreads in question
01-download-range:
		python3 -m nightlightsprocessing.01_download_VNP46A1_range -d "./data/01-VNP46A1-h5" -t "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJBUFMgT0F1dGgyIEF1dGhlbnRpY2F0b3IiLCJpYXQiOjE2ODU1MzQ3MjUsIm5iZiI6MTY4NTUzNDcyNSwiZXhwIjoxNzAxMDg2NzI1LCJ1aWQiOiJqYW1pZWFsbGVuNTkiLCJlbWFpbF9hZGRyZXNzIjoiamFtaWVhbGxlbjU5QGdtYWlsLmNvbSIsInRva2VuQ3JlYXRvciI6ImphbWllYWxsZW41OSJ9.Hh5uHl3N5TWKblonqNT1-UwsdIgYNbwCYLmPTme_wxw"

# 02 create time of capture spreads dataset based on downloaded A1 images UTC_time band
02-create-image-taken-times-dataset:
		python3 -m nightlightsprocessing.02_create_image_taken_times_dataset

# 03 Create ground truth dataset of low reliability date instances based on time of capture spreads
# Note: may want to do this for multiple locations in your Area of Interest
03-create-reliability-dataset:
	python3 -m nightlightsprocessing.03_create_reliability_dataset -s "Uttar Pradesh" -l "Bahraich"

# 04 dowload A2 images based on low reliability ground truth dataset
04-download-A2-based-on-date-and-location-instances:
		python3 -m nightlightsprocessing.04_download_VNP46A2_instances -d "./input-data/night-lights/h5" -t "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJBUFMgT0F1dGgyIEF1dGhlbnRpY2F0b3IiLCJpYXQiOjE2ODU1MzQ3MjUsIm5iZiI6MTY4NTUzNDcyNSwiZXhwIjoxNzAxMDg2NzI1LCJ1aWQiOiJqYW1pZWFsbGVuNTkiLCJlbWFpbF9hZGRyZXNzIjoiamFtaWVhbGxlbjU5QGdtYWlsLmNvbSIsInRva2VuQ3JlYXRvciI6ImphbWllYWxsZW41OSJ9.Hh5uHl3N5TWKblonqNT1-UwsdIgYNbwCYLmPTme_wxw"

# 05 process all A2 images with the given masks
05-process-all-vnp46a2-images:
		python3 -m nightlightsprocessing.05_process_VNP46A2_images

# 06 create .shp files for your areas
06-create-shapefile:
		python3 -m nightlightsprocessing.06_create_shapefile

# 07 crop processed images based on ground truth data
07-crop-images:
		python3 -m nightlightsprocessing.07_crop_images


# Ancillary
read-clipped-files:
		python3 -m nightlightsprocessing.read_clipped_files





# venv/bin/activate: requirements.txt
# 	python3 -m $(VENV) $(VENV)
# 	./$(PIP) install -r requirements.txt


# run: venv/bin/activate
# 	$(PYTHON) app.py

.PHONY: test

# TODO: should turn all this to one entry point with script arguements (assuming it stays as one package)
# https://stackoverflow.com/questions/57744466/how-to-properly-structure-internal-scripts-in-a-python-project


