#!/usr/bin/env python3

# This script takes in 'state' and 'location' args and creates a .csv of those
# locations where there is no voltage for the amount of minuted defined in any given hour

import sys, getopt, os
from nightlightsprocessing import helpers
from . import constants
import pandas as pd
from .get_ESMI_location_information import get_ESMI_location_information

################################################################################

# Variables
INPUT_FOLDER = constants.INPUT_GROUND_TRUTH_FOLDER
OUTPUT_FOLDER = constants.OUTPUT_GROUND_TRUTH_FOLDER
# For any given hour. So if you want the full hour where there is no voltage, you'd use '60'
CONTINUOUS_MINUTES_WITH_NO_VOLTAGE_PER_HOUR = 60

# Constants
# As the filenames are e.g. 'ESMI minute-wise voltage data 2014.csv', 'voltage data' will get all .csv's
# and concatenate them together to use to check against.
VOLTAGE_DATA_FILENAME = "ESMI minute-wise voltage data"
LOCATION_NAME_COLUMN = "Location name"
DATE_COLUMN = "Date"

################################################################################


def _get_groundtruth_csvs_filtered_by(location_names):
    groundtruth_files = helpers.getAllFilesFromFolderWithFilename(INPUT_FOLDER, VOLTAGE_DATA_FILENAME)

    # Keys are created to allow knowing which original dataset each row came from
    frames = []

    for filename in groundtruth_files:
        file_path = f"{os.getcwd()}{INPUT_FOLDER}/{filename}"

        date_format = "%d-%m-%Y"
        dataframe = pd.read_csv(file_path, parse_dates=[DATE_COLUMN], date_format=date_format, dayfirst=True)
        # Force converting date fields to dates
        dataframe[DATE_COLUMN] = pd.to_datetime(dataframe[DATE_COLUMN], errors="coerce")
        # Remove any fields that cannot be changed to dates
        dataframe.dropna(subset=[DATE_COLUMN], inplace=True)

        filtered_dataframe = dataframe[dataframe[LOCATION_NAME_COLUMN].isin(location_names)]

        frames.append(filtered_dataframe)

    return frames


def _get_filtered_by_zeros(dataframe, amount):
    zeros_columns = dataframe.columns[3:]  # Select the columns starting from Min 0
    filtered_df = dataframe[dataframe[zeros_columns].eq(0).sum(axis=1) >= amount]

    return filtered_df


# Different years have different spreads of when images were taken.
# The spread come from the UTC_time property of the VNP46A1 images for that year.
def _get_filtered_by_hours(df):
    filtered_rows = []  # List to store filtered rows

    # Apply a boolean mask depending
    for index, row in df.iterrows():
        date = pd.to_datetime(row[DATE_COLUMN], errors="coerce")
        year = date.year

        # Spread results: 19:44:12 - 20:31:18
        if year == 2014:
            seven_pm = 19
            eight_pm = 20
            filter_condition = (row["Hour"] >= seven_pm) & (row["Hour"] <= eight_pm)
        elif year == 2015:
            seven_pm = 19
            eight_pm = 20
            filter_condition = (row["Hour"] >= seven_pm) & (row["Hour"] <= eight_pm)
        elif year == 2016:
            seven_pm = 19
            eight_pm = 20
            filter_condition = (row["Hour"] >= seven_pm) & (row["Hour"] <= eight_pm)
        elif year == 2017:
            seven_pm = 19
            eight_pm = 20
            filter_condition = (row["Hour"] >= seven_pm) & (row["Hour"] <= eight_pm)
        # Spread results: 18:48:25 - 21:19:11
        elif year == 2018:
            six_pm = 18
            nine_pm = 21
            filter_condition = (row["Hour"] >= six_pm) & (row["Hour"] <= nine_pm)
        elif year == 2019:
            seven_pm = 19
            eight_pm = 20
            filter_condition = (row["Hour"] >= seven_pm) & (row["Hour"] <= eight_pm)
        # Should never get here. If it does, we use the most restrictive times.
        else:
            print(f"No year found. Year value: {year}. Date value: {date}")
            seven_pm = 19
            eight_pm = 20
            print("Using most restrictive times between {seven_pm} and {eight_pm}")
            filter_condition = (row["Hour"] >= seven_pm) & (row["Hour"] <= eight_pm)

        if filter_condition:
            filtered_rows.append(row)

    return pd.DataFrame(filtered_rows)


################################################################################


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

    location_names = get_ESMI_location_information(indian_state, indian_state_location)[LOCATION_NAME_COLUMN]
    print("location_names", location_names)
    # get all csv files
    # loop over them and filter them all by the given state
    dataframes = _get_groundtruth_csvs_filtered_by(location_names)
    # combine them all into one spreadsheet
    result = pd.concat(dataframes)
    # Sort new dataframe by date
    result = result.sort_values("Date", ascending=True, kind="stable", na_position="first", ignore_index=True)
    # print(type(result.iloc[1, 5]))

    # Each row is an hour, so '60' means the lights were entirely off
    result_filtered_by_zeros_included = _get_filtered_by_zeros(result, CONTINUOUS_MINUTES_WITH_NO_VOLTAGE_PER_HOUR)
    result_filtered_by_certain_hours = _get_filtered_by_hours(result_filtered_by_zeros_included)
    # Save to new csv called
    write_file_path = (
        f"{os.getcwd()}{OUTPUT_FOLDER}/ESMI minute-wise voltage data - Uttar Pradesh - Lucknow - filtered test.csv"
    )
    print("Writing new data to: ", write_file_path)
    result_filtered_by_certain_hours.to_csv(write_file_path)


if __name__ == "__main__":
    main(sys.argv[1:])
