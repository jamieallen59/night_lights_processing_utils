setup:
	pip install -r requirements.txt

test:
	python3 -m unittest discover tests

clean:
	rm -rf __pycache__
	rm -rf venv

grid-reliability:
	python3 -m nightlightsprocessing.groundtruth.grid_reliability

# venv/bin/activate: requirements.txt
# 	python3 -m $(VENV) $(VENV)
# 	./$(PIP) install -r requirements.txt


# run: venv/bin/activate
# 	$(PYTHON) app.py

.PHONY: test
