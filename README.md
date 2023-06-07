# Night Lights Processing Utils

`nightlights_processing_utils` is a set of python scripts for managing night lights satellite data.

For use on:

- The data this package is for is from the Suomi National Polar-orbiting Partnership (SNPP) Visible Infrared Imaging Radiometer Suite (VIIRS).
- Specifically, the **VNP46A2** product:https://ladsweb.modaps.eosdis.nasa.gov/missions-and-measurements/products/VNP46A2/
- Relevant data can be downloaded from https://ladsweb.modaps.eosdis.nasa.gov/search/
- Many of these post-processing steps are defined here: https://ladsweb.modaps.eosdis.nasa.gov/learn/how-to-use-laads-daac-post-processing-tools/
- I would strongly recommend reading page 16-17 of the [Black Marble guide](https://viirsland.gsfc.nasa.gov/PDF/BlackMarbleUserGuide_v1.2_20220916.pdf) before continuing.

---

## Prerequisites

You will need [python 3](https://www.python.org/downloads/).
Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the scripts requirements.

Activate a virtual environment (https://realpython.com/python-virtual-environments-a-primer/):

```bash
python3 -m venv venv
. venv/bin/activate
```

```bash
make setup
```

Troubleshooting:

- if you have the M1 chip mac you may find clashes between gdal versions. To download gdal correctly for your computer run:

```bash
ARCHFLAGS="-arch arm64" pip install gdal==$(gdal-config --version)  --compile --no-cache-dir
```

Source: https://stackoverflow.com/questions/75902777/error-running-gdal-scripts-with-python3-on-macos-12-monterrey

In the project structure there are directories:

    .
    ├── docs                    # Documentation files
    ├── input-data
          ├── ground-truth      # folder to put the input ground truth data
          └── night-lights      # folder to put the input satellight night light data in (VNP46A1 + VNP46A2)
    ├── output-data
          ├── ground-truth      # output folder for ground truth scripts
          └── night-lights      # output folder for night lights scripts
    ├── nightlightsprocessing   # Main package folder containing python scripts
          ├── groundtruth      # scripts for ground truth input data
          ├── nightlights      # scripts for night lights input data
          └── mlmodel          # scripts for training and running machine learning analysis models
    ├── tests                   # Automated tests
    ├── Makefile                # defines all project interactions
    ├── LICENSE
    └── README.md

- [./input](./input)
- [./output](./output)

The idea is to:

- download the files you want from the links above, and put the .h5 files in the [input](./input) directory. That's where the various scripts exepct them to be.
- There is a set of constants defined in [scripts/constants](scripts/constants). Please check these suit your requirements.

## Usage

TODO

## Testing

To run a test file e.g:

```bash
make test
```

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License

[GNU General Public Licence v3.0](https://choosealicense.com/licenses/gpl-3.0/)
