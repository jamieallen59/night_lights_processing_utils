VENV = venv
PIP = $(VENV)/bin/pip

setup:
	$(PIP) install -r requirements.txt

test:
	python3 -m unittest discover tests

clean:
	rm -rf __pycache__
	rm -rf $(VENV)

venv/bin/activate: requirements.txt
	python3 -m $(VENV) $(VENV)
	./$(PIP) install -r requirements.txt


# run: venv/bin/activate
# 	$(PYTHON) app.py

.PHONY: test
