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

# Constants
# As the filenames are e.g. 'ESMI minute-wise voltage data 2014.csv', 'voltage data' will get all .csv's
# and concatenate them together to use to check against.
VOLTAGE_DATA_FILENAME = "ESMI minute-wise voltage data"
LOCATION_NAME_COLUMN = "Location name"
DATE_COLUMN = "Date"
DATETIME_FORMAT = "%Y-%m-%d"

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


# filters out rows that don't have the required number of 0's
# defined by the 'amount' arg.
# Low = 19:36:07
# High = 20:35:26
def _get_filtered_by_zeros(dataframe):
    low_spread_hour_value = 19
    low_spread_minute_value = 36
    high_spread_hour_value = 20
    high_spread_minute_value = 35
    filtered_rows = []  # List to store filtered rows

    for index, row in dataframe.iterrows():
        hour = row["Hour"]

        if hour == low_spread_hour_value:
            if all(row[f"Min {minute}"] == 0 for minute in range(low_spread_minute_value, 60)):
                filtered_rows.append(row)
        elif hour == high_spread_hour_value:
            # +1 as range with a single value starts at 0
            if all(row[f"Min {minute}"] == 0 for minute in range(0, high_spread_minute_value)):
                filtered_rows.append(row)

    # Create a new DataFrame with the filtered rows
    filtered_df = pd.DataFrame(filtered_rows)

    return filtered_df


# Low = 19:36:07
# High = 20:35:26
def _get_filtered_by_hours(df):
    low_spread_hour_value = 19
    high_spread_hour_value = 20
    filtered_df = df[(df["Hour"] >= low_spread_hour_value) & (df["Hour"] <= high_spread_hour_value)]
    return filtered_df


# This is done to ensure the date instances we find cover both hours.
def _get_filtered_by_both_hours(df):
    date_counts = df["Date"].value_counts()

    # Filter the DataFrame to include only dates with two rows
    filtered_dates = date_counts[date_counts == 2].index.tolist()
    filtered_df = df[df["Date"].isin(filtered_dates)]

    return filtered_df


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
    print("Finding location names...")
    print(location_names)
    # get all csv files
    # loop over them and filter them all by the given state
    dataframes = _get_groundtruth_csvs_filtered_by(location_names)
    # combine them all into one spreadsheet
    result = pd.concat(dataframes)
    # Sort new dataframe by date
    result = result.sort_values("Date", ascending=True, kind="stable", na_position="first", ignore_index=True)
    # print(type(result.iloc[1, 5]))

    result_filtered_by_certain_hours = _get_filtered_by_hours(result)
    result_filtered_by_zeros = _get_filtered_by_zeros(result_filtered_by_certain_hours)
    result_filtered_by_outages_over_both_hours = _get_filtered_by_both_hours(result_filtered_by_zeros)
    # Save to new csv called
    # This isn't the final dataset, but is the data for all locations filtered to only include
    # days when there are outages. It's written to csv to allow inspection.
    write_full_location_results_file_path = f"{os.getcwd()}{OUTPUT_FOLDER}/ESMI minute-wise voltage data - {indian_state} - {indian_state_location} - filtered.csv"
    print("Writing full filtered data to to allow inspection at: ", write_full_location_results_file_path)
    result_filtered_by_outages_over_both_hours.to_csv(write_full_location_results_file_path)

    # Then here we work on the data further to get individual locations and dates from the data.
    # This is what we need to know which dates for VNP46A2 images to download + the locations needed to crop on that date.

    date_location_counts = result_filtered_by_outages_over_both_hours.groupby(["Date", "Location name"]).size()
    # Filter the DataFrame to include only rows with at least two instances of the date
    filtered_dates = date_location_counts[date_location_counts >= 2].reset_index()
    # Extract the unique date + location combinations as the final result
    result = filtered_dates[["Location name", "Date"]].drop_duplicates()
    # Add a 'Date day integer' which can be used for the download script
    result["Date day integer"] = result["Date"].dt.dayofyear
    result["Date year integer"] = result["Date"].dt.year

    write_unique_date_location_combinations_path = f"{os.getcwd()}{OUTPUT_FOLDER}/ESMI minute-wise voltage data - {indian_state} - {indian_state_location} - filtered unique.csv"
    print("Writing unique locations and dates to: ", write_full_location_results_file_path)
    result.to_csv(write_unique_date_location_combinations_path)


if __name__ == "__main__":
    main(sys.argv[1:])
