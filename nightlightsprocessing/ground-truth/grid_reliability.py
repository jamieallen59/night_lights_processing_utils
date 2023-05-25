from nightlightsprocessing import helpers
import constants
from location_information import get_location_information
import pandas as pd
import os

folder = constants.GROUND_TRUTH_FOLDER

# file info
VOLTAGE_DATA_FILENAME = "ESMI minute-wise voltage data 2014"

# Column names
LOCATION_NAME_COLUMN = "Location name"
DATE_COLUMN = "Date"


def read_voltage_data_csv():
    voltage_data_files = helpers.getAllFilesFromFolderWithFilename(folder, VOLTAGE_DATA_FILENAME)
    voltage_data_file = voltage_data_files[0]
    voltage_data_file_path = f"{os.getcwd()}{folder}/{voltage_data_file}"

    voltage_dataframe = pd.read_csv(voltage_data_file_path)

    return voltage_dataframe


def filter_voltage_data_by_location_names(voltage_data, location_names):
    filtered_df = voltage_data[voltage_data[LOCATION_NAME_COLUMN].isin(location_names)]
    filtered_df = helpers.drop_filtered_table_index(filtered_df)

    return filtered_df


def main():
    location_names = get_location_information()[LOCATION_NAME_COLUMN]
    voltage_data = read_voltage_data_csv()

    # Filtering
    filtered_df = filter_voltage_data_by_location_names(voltage_data, location_names)

    print("filtered_df", filtered_df)


if __name__ == "__main__":
    main()
