setup:
	pip install -r requirements.txt

test:
	python3 -m unittest discover tests

clean:
	rm -rf __pycache__
	rm -rf venv

# ground truth data
# Unused atm. Might need functions inside it though for filtering by minute/hour in create-training-dataset-of-low-reliability-grids.
# grid-reliability:
# 	python3 -m nightlightsprocessing.groundtruth.grid_reliability -s 2014-11-01T23:45:00.000Z -e 2014-11-02T00:15:00.000Z

create-dataset-of-low-reliability-dates:
	python3 -m nightlightsprocessing.groundtruth.create_dataset_of_low_reliability_dates -s "Uttar Pradesh" -l "Lucknow"


# night lights
download:
		python3 -m nightlightsprocessing.nightlights.download -d "./input-data/night-lights/h5" -t "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJBUFMgT0F1dGgyIEF1dGhlbnRpY2F0b3IiLCJpYXQiOjE2ODU1MzQ3MjUsIm5iZiI6MTY4NTUzNDcyNSwiZXhwIjoxNzAxMDg2NzI1LCJ1aWQiOiJqYW1pZWFsbGVuNTkiLCJlbWFpbF9hZGRyZXNzIjoiamFtaWVhbGxlbjU5QGdtYWlsLmNvbSIsInRva2VuQ3JlYXRvciI6ImphbWllYWxsZW41OSJ9.Hh5uHl3N5TWKblonqNT1-UwsdIgYNbwCYLmPTme_wxw"

create-VNP46A1-UTC-Time-dataset:
		python3 -m nightlightsprocessing.nightlights.create_VNP46A1_UTC_Time_dataset

process-vnp46a2-images:
		python3 -m nightlightsprocessing.nightlights.process_VNP46A2_images

test-clip-image:
		python3 -m nightlightsprocessing.nightlights.clip_image

read-clipped-files:
		python3 -m nightlightsprocessing.nightlights.read_clipped_files




# venv/bin/activate: requirements.txt
# 	python3 -m $(VENV) $(VENV)
# 	./$(PIP) install -r requirements.txt


# run: venv/bin/activate
# 	$(PYTHON) app.py

.PHONY: test

# TODO: should turn all this to one entry point with script arguements (assuming it stays as one package)
# https://stackoverflow.com/questions/57744466/how-to-properly-structure-internal-scripts-in-a-python-project


