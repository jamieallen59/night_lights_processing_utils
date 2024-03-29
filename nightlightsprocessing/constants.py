VOLTAGE_DATA_FILENAME = "ESMI minute-wise voltage data"
DATETIME_FORMAT = "%d/%m/%Y"

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

GRID_RELIABILITIES = ["LOW", "HIGH"]

# All available datasets for VNP46A2 listed on page 16 here:
# https://viirsland.gsfc.nasa.gov/PDF/BlackMarbleUserGuide_v1.2_20220916.pdf

CLASS_MAPPING = {"HIGH": 1, "LOW": 0}
INVERSE_CLASS_MAPPING = {1: "HIGH", 0: "LOW"}

COLOURS = {
    "blue": "#1f77b4",
    "red": "#d62728",
    "grey": "grey",
    "green": "green",
    "orange": "orange",
    "black": "black",
}
