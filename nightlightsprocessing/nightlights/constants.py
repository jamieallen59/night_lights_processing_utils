# This file assumes scripts are run from the root of the project
INPUT_FOLDER = "/input-data/night-lights"
H5_INPUT_FOLDER = f"{INPUT_FOLDER}/h5"
TIF_INPUT_FOLDER = f"{INPUT_FOLDER}/tif"
OUTPUT_FOLDER = "/output-data/night-lights"

FILE_EXTENSION_TIF = ".tif"
FILE_TYPE = "VNP46A2"


BRDF_CORRECTED = "DNB_BRDF-Corrected_NTL"
QUALITY_FLAG = "Mandatory_Quality_Flag"
CLOUD_MASK = "QF_Cloud_Mask"
SELECTED_DATASETS = [BRDF_CORRECTED, QUALITY_FLAG, CLOUD_MASK]
# From available:
# [
# 'DNB_BRDF-Corrected_NTL',
# 'DNB_Lunar_Irradiance',
# 'Gap_Filled_DNB_BRDF-Corrected_NTL',
# 'Latest_High_Quality_Retrieval',
# 'Mandatory_Quality_Flag',
# 'QF_Cloud_Mask',
# 'Snow_Flag'
# ]
# All available datasets for VNP46A2 listed on page 16 here:
# https://viirsland.gsfc.nasa.gov/PDF/BlackMarbleUserGuide_v1.2_20220916.pdf
