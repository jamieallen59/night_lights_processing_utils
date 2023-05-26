# Night Lights Processing Utils

`night_lights_processing_utils` is a set of python scripts for managing night lights satellite data.

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
python -m venv venv
. venv/bin/activate
```

```bash
make setup
```

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
          ├── ground_truth      # scripts for ground truth input data
          ├── night_lights      # scripts for night lights input data
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

Once you have the files you want in the input directory, there are a set of scripts available to use:

1. Convert single hd5 files to geotiff: [single_hd5_to_geotiff](./scripts//satellite//single_hd5_to_geotiff.py). To run:

```bash
python scripts/satellite/single_hd5_to_geotiff.py
```

2. Convert multiple hd5 files to geotiff: [multiple_hd5_to_geotiff](./scripts//satellite//multiple_hd5_to_geotiff.py). To run:

```bash
python scripts/satellite/multiple_hd5_to_geotiff.py
```

## Testing

To run a test file e.g:

```bash
 python -m unittest scripts.VNP46A2.tests.image_processing
```

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License

[GNU General Public Licence v3.0](https://choosealicense.com/licenses/gpl-3.0/)
