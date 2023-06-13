setup:
	pip install -r requirements.txt

test:
	python3 -m unittest discover tests

clean:
	rm -rf __pycache__
	rm -rf venv

# ground truth data
grid-reliability:
	python3 -m nightlightsprocessing.groundtruth.grid_reliability -s 2014-11-01T23:45:00.000Z -e 2014-11-02T00:15:00.000Z

create-groundtruth:
	python3 -m nightlightsprocessing.groundtruth.create_location_dataset -s "Uttar Pradesh"


# night lights
download:
		python3 -m nightlightsprocessing.nightlights.download -s "https://ladsweb.modaps.eosdis.nasa.gov/archive/allData/5000/VNP46A1/2014/305" -d "./input-data/night-lights/h5" -t "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJBUFMgT0F1dGgyIEF1dGhlbnRpY2F0b3IiLCJpYXQiOjE2ODU1MzQ3MjUsIm5iZiI6MTY4NTUzNDcyNSwiZXhwIjoxNzAxMDg2NzI1LCJ1aWQiOiJqYW1pZWFsbGVuNTkiLCJlbWFpbF9hZGRyZXNzIjoiamFtaWVhbGxlbjU5QGdtYWlsLmNvbSIsInRva2VuQ3JlYXRvciI6ImphbWllYWxsZW41OSJ9.Hh5uHl3N5TWKblonqNT1-UwsdIgYNbwCYLmPTme_wxw"

convert-hd5-to-geotiff:
		python3 -m nightlightsprocessing.nightlights.hd5_to_geotiff

image-processing:
		python3 -m nightlightsprocessing.nightlights.image_processing

get-vnp46a1-time-spread:
		python3 -m nightlightsprocessing.nightlights.VNP46A1




# venv/bin/activate: requirements.txt
# 	python3 -m $(VENV) $(VENV)
# 	./$(PIP) install -r requirements.txt


# run: venv/bin/activate
# 	$(PYTHON) app.py

.PHONY: test

# TODO: should turn all this to one entry point with script arguements (assuming it stays as one package)
# https://stackoverflow.com/questions/57744466/how-to-properly-structure-internal-scripts-in-a-python-project