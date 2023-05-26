VENV = venv
PYTHON = $(VENV)/bin/python3
PIP = $(VENV)/bin/pip

setup:
	$(PIP) install -r requirements.txt

test:
	python -m unittest

clean:
	rm -rf __pycache__
	rm -rf $(VENV)

venv/bin/activate: requirements.txt
	python3 -m $(VENV) $(VENV)
	./$(PIP) install -r requirements.txt


run: venv/bin/activate
	$(PYTHON) app.py

.PHONY: test
