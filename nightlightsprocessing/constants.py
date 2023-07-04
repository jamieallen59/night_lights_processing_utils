VOLTAGE_DATA_FILENAME = "ESMI minute-wise voltage data"
DATETIME_FORMAT = "%Y-%m-%d"

DATA_PATH = "/data"
OO_GROUND_TRUTH_PATH = f"{DATA_PATH}/00-ground-truth"
O1_VNP46A1_H5_PATH = f"{DATA_PATH}/01-VNP46A1-h5"
O2_VNP46A1_IMAGE_CREATED_TIMES_PATH = f"{DATA_PATH}/02-VNP46A1-image-created-times"
O3_RELIABILITY_DATASETS_PATH = f"{DATA_PATH}/03-reliability-datasets"
O4_VNP46A2_H5_PATH = f"{DATA_PATH}/04-VNP46A2-h5"
O5_PROCESSED_VNP46A2_IMAGES = f"{DATA_PATH}/05-processed-VNP46A2-images"
O6_LOCATION_SHAPEFILES = f"{DATA_PATH}/06-location-shapefiles"
O7_CROPPED_IMAGES = f"{DATA_PATH}/07-cropped-images"


FILE_TYPE_VNP46A2 = "VNP46A2"

# VNP46A2 mask properties
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
