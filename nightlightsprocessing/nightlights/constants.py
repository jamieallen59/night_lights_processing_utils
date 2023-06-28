# This file assumes scripts are run from the root of the project
INPUT_FOLDER = "/input-data"
NIGHT_LIGHTS_INPUT_FOLDER = f"{INPUT_FOLDER}/night-lights"
H5_INPUT_FOLDER = f"{NIGHT_LIGHTS_INPUT_FOLDER}/h5"
LOCATION_INPUT_FOLDER = f"{INPUT_FOLDER}/locations"

OUTPUT_FOLDER = "/output-data/night-lights"

FILE_EXTENSION_TIF = ".tif"
FILE_TYPE_VNP46A2 = "VNP46A2"


BRDF_CORRECTED = "DNB_BRDF-Corrected_NTL"
QUALITY_FLAG = "Mandatory_Quality_Flag"
CLOUD_MASK = "QF_Cloud_Mask"
# The datasets used by this project
SELECTED_DATASETS = [BRDF_CORRECTED, QUALITY_FLAG, CLOUD_MASK]
# From available:
BAND_NAMES = [
    BRDF_CORRECTED,
    "DNB_Lunar_Irradiance",
    "Gap_Filled_DNB_BRDF-Corrected_NTL",
    "Latest_High_Quality_Retrieval",
    QUALITY_FLAG,
    CLOUD_MASK,
    "Snow_Flag",
]
# All available datasets for VNP46A2 listed on page 16 here:
# https://viirsland.gsfc.nasa.gov/PDF/BlackMarbleUserGuide_v1.2_20220916.pdf

DATETIME_FORMAT = "%Y-%m-%d"
