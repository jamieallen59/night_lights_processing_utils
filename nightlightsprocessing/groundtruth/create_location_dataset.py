import sys, getopt, os
from nightlightsprocessing import helpers
from . import constants
import pandas as pd
from .location_information import get_location_information
import csv

# this is a one time script used to take all the csv files for different years
# and combine them into a single csv for a given state.

VOLTAGE_DATA_FILENAME = "voltage data"
input_folder = constants.INPUT_GROUND_TRUTH_FOLDER
output_folder = constants.OUTPUT_GROUND_TRUTH_FOLDER
LOCATION_NAME_COLUMN = "Location name"
DATE_COLUMN = "Date"


def get_all_groundtruth_files():
    all_groundtruth_files = helpers.getAllFilesFromFolderWithFilename(input_folder, VOLTAGE_DATA_FILENAME)

    return all_groundtruth_files


# TODO duplicated
def filter_by_state(dataframe, location_names):
    filtered_df = dataframe[dataframe[LOCATION_NAME_COLUMN].isin(location_names)]
    # filtered_df = helpers.drop_filtered_table_index(filtered_df)

    return filtered_df


def get_groundtruth_csvs_filtered_by(location_names):
    groundtruth_files = get_all_groundtruth_files()

    # Keys are created to allow knowing which original dataset each row came from
    frames = []

    for filename in groundtruth_files:
        file_path = f"{os.getcwd()}{input_folder}/{filename}"

        date_format = "%d-%m-%Y"
        dataframe = pd.read_csv(file_path, parse_dates=[DATE_COLUMN], date_format=date_format, dayfirst=True)
        # Force converting date fields to dates
        dataframe[DATE_COLUMN] = pd.to_datetime(dataframe[DATE_COLUMN], errors="coerce")
        # Remove any fields that cannot be changed to dates
        dataframe.dropna(subset=[DATE_COLUMN], inplace=True)

        filtered_dataframe = filter_by_state(dataframe, location_names)

        frames.append(filtered_dataframe)

    return frames


def get_filtered_by_zeros(dataframe, amount):
    zeros_columns = dataframe.columns[3:]  # Select the columns starting from Min 0
    filtered_df = dataframe[dataframe[zeros_columns].eq(0).sum(axis=1) >= amount]

    return filtered_df


def get_filtered_by_hours(df):
    seven_pm = 19
    eight_pm = 20

    filtered_df = df[(df["Hour"] >= seven_pm) & (df["Hour"] <= eight_pm)]
    return filtered_df


def main(argv):
    indian_state = None
    indian_state_location = None

    try:
        opts, args = getopt.getopt(argv, "hs:l:", ["indian_state="])
    except getopt.GetoptError:
        print("Possible options: -s <indian_state> -l <indian_state_location>")
        sys.exit(2)

    for opt, arg in opts:
        if opt == "-h":
            print("Possible options: -s <indian_state> -l <indian_state_location>")
            sys.exit()
        elif opt in ("-s", "--indian_state"):
            indian_state = arg
        elif opt in ("-l", "--indian_state_location"):
            indian_state_location = arg

    location_names = get_location_information(indian_state, indian_state_location)[LOCATION_NAME_COLUMN]
    print("location_names", location_names)
    # get all csv files
    # loop over them and filter them all by the given state
    dataframes = get_groundtruth_csvs_filtered_by(location_names)
    # combine them all into one spreadsheet
    result = pd.concat(dataframes)
    # Sort new dataframe by date
    result = result.sort_values("Date", ascending=True, kind="stable", na_position="first", ignore_index=True)
    # print(type(result.iloc[1, 5]))

    # May want to change
    CONTINUOUS_ZEROS_NUMBER = 59
    result_filtered_by_zeros_included = get_filtered_by_zeros(result, CONTINUOUS_ZEROS_NUMBER)
    result_filtered_by_certain_hours = get_filtered_by_hours(result_filtered_by_zeros_included)
    # Save to new csv called
    write_file_path = (
        f"{os.getcwd()}{output_folder}/ESMI minute-wise voltage data - Uttar Pradesh - Lucknow - filtered.csv"
    )
    print("Writing new data to: ", write_file_path)
    result_filtered_by_certain_hours.to_csv(write_file_path)


if __name__ == "__main__":
    main(sys.argv[1:])
