#!/usr/bin/env python3

# This script takes in 'state' and 'location' args and creates a .csv of those
# locations where there is no voltage for the amount of minuted defined in any given hour
import argparse
import sys
from nightlightsprocessing import helpers
from . import constants
import pandas as pd
from .get_ESMI_location_information import get_ESMI_location_information

COMMON_GROUND_TRUTH_FILENAME = constants.VOLTAGE_DATA_FILENAME
DESC = "This script will create a dataset from the ground truth data provided, and based on the state and location provided."
# TODO: use centralised column names
LOCATION_NAME_COLUMN = "Location name"
DATE_COLUMN = "Date"
# TODO: move to Makefile?
LOW_RELIABILITY_VOLTAGE = 0
HIGH_RELIABILITY_VOLTAGE = 170
DEBUG = False

################################################################################


def _get_groundtruth_csvs_filtered_by(input_folder, location_names):
    groundtruth_files = helpers.getAllFilesFromFolderWithFilename(input_folder, COMMON_GROUND_TRUTH_FILENAME)

    # Keys are created to allow knowing which original dataset each row came from
    frames = []

    for filename in groundtruth_files:
        file_path = f"{input_folder}/{filename}"

        dataframe = pd.read_csv(
            file_path, parse_dates=[DATE_COLUMN], date_format=constants.DATETIME_FORMAT, dayfirst=True
        )
        # Force converting date fields to dates
        dataframe[DATE_COLUMN] = pd.to_datetime(arg=dataframe[DATE_COLUMN], errors="coerce", dayfirst=True)
        # Remove any fields that cannot be changed to dates
        dataframe.dropna(subset=[DATE_COLUMN], inplace=True)

        filtered_dataframe = dataframe[dataframe[LOCATION_NAME_COLUMN].isin(location_names)]

        frames.append(filtered_dataframe)

    return frames


# filters out rows that don't have the required number of 0's
# defined by the 'amount' arg.
# Low = 19:36:07
# High = 20:35:26
def _get_filtered_by_zeros(dataframe, grid_reliability):
    # TODO: move to Makefile?
    low_spread_hour_value = 19
    low_spread_minute_value = 36
    high_spread_hour_value = 20
    high_spread_minute_value = 35
    filtered_rows = []  # List to store filtered rows

    for index, row in dataframe.iterrows():
        hour = row["Hour"]

        if hour == low_spread_hour_value:
            conditional = (
                all(
                    int(row[f"Min {minute}"]) >= HIGH_RELIABILITY_VOLTAGE
                    for minute in range(low_spread_minute_value, 60)
                )
                if grid_reliability == "HIGH"
                else all(
                    int(row[f"Min {minute}"]) == LOW_RELIABILITY_VOLTAGE
                    for minute in range(low_spread_minute_value, 60)
                )
            )

            if conditional:
                filtered_rows.append(row)
        elif hour == high_spread_hour_value:
            conditional = (
                all(
                    int(row[f"Min {minute}"]) >= HIGH_RELIABILITY_VOLTAGE
                    for minute in range(0, high_spread_minute_value)
                )
                if grid_reliability == "HIGH"
                else all(
                    int(row[f"Min {minute}"]) == LOW_RELIABILITY_VOLTAGE
                    for minute in range(0, high_spread_minute_value)
                )
            )

            if conditional:
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
def _get_filtered_when_a_location_has_2_hours_together(df):
    location_date_counts = df.groupby(["Location name", "Date"]).size()

    # Filter the DataFrame to include only locations with two dates
    filtered_locations_dates = location_date_counts[location_date_counts == 2].index.tolist()
    filtered_df = df[df.set_index(["Location name", "Date"]).index.isin(filtered_locations_dates)]

    return filtered_df


def _get_concatenated_groundtruth_sorted_by_date(input_folder, state, location):
    location_names = get_ESMI_location_information(input_folder, state, location)[LOCATION_NAME_COLUMN]
    print("Finding location names...")
    print(location_names)
    # get all csv files
    # loop over them and filter them all by the given state
    dataframes = _get_groundtruth_csvs_filtered_by(input_folder, location_names)
    # combine them all into one spreadsheet
    result = pd.concat(dataframes)
    # Sort new dataframe by date
    return result.sort_values("Date", ascending=True, kind="stable", na_position="first", ignore_index=True)


def create_reliability_dataset(input_folder, destination, state, location, grid_reliability):
    concatenated_groundtruth_sorted_by_date = _get_concatenated_groundtruth_sorted_by_date(
        input_folder, state, location
    )

    result_filtered_by_certain_hours = _get_filtered_by_hours(concatenated_groundtruth_sorted_by_date)

    if DEBUG:
        write_full_location_results_file_path = f"{destination}/ESMI minute-wise voltage data - {state} - {location} - filtered debug 1 {grid_reliability}.csv"
        print("Writing non filtered data to to allow inspection at: ", write_full_location_results_file_path)
        result_filtered_by_certain_hours.to_csv(write_full_location_results_file_path)

    result_filtered_by_zeros = _get_filtered_by_zeros(result_filtered_by_certain_hours, grid_reliability)

    if DEBUG:
        write_full_location_results_file_path = f"{destination}/ESMI minute-wise voltage data - {state} - {location} - filtered debug 2 {grid_reliability}.csv"
        print("Writing non filtered data to to allow inspection at: ", write_full_location_results_file_path)
        result_filtered_by_zeros.to_csv(write_full_location_results_file_path)

    result_filtered_by_outages_over_both_hours = _get_filtered_when_a_location_has_2_hours_together(
        result_filtered_by_zeros
    )

    if DEBUG:
        # Saves to new csv. This isn't the final dataset, but is the data for all locations filtered to only include
        # days when there are outages. It's written to csv to allow inspection.
        write_filtered_location_results_file_path = f"{destination}/ESMI minute-wise voltage data - {state} - {location} - filtered debug 3 {grid_reliability}.csv"

        print("Writing full filtered data to to allow inspection at: ", write_filtered_location_results_file_path)
        result_filtered_by_outages_over_both_hours.to_csv(write_filtered_location_results_file_path)

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

    write_unique_date_location_combinations_path = (
        f"{destination}/{helpers.get_reliability_dataset_filename(state, location, grid_reliability)}"
    )

    print("Writing unique locations and dates to: ", write_unique_date_location_combinations_path)
    result.to_csv(write_unique_date_location_combinations_path)


################################################################################


def _main(argv):
    parser = argparse.ArgumentParser(prog=argv[0], description=DESC)
    parser.add_argument(
        "-s",
        "--state",
        dest="state",
        help="State in India",
        required=True,
    )
    parser.add_argument("-l", "--location", dest="location", help="Location within State defined", required=True)
    parser.add_argument(
        "-d",
        "--destination",
        dest="destination",
        help="Store directory structure in DIR",
        required=True,
    )
    parser.add_argument(
        "-gr",
        "--grid-reliability",
        dest="grid_reliability",
        help="A value either LOW or HIGH to represent the reliability of the grid",
        required=True,
    )
    parser.add_argument("-i", "--input-folder", dest="input_folder", help="Input data directory", required=True)

    args = parser.parse_args(argv[1:])

    create_reliability_dataset(args.input_folder, args.destination, args.state, args.location, args.grid_reliability)


if __name__ == "__main__":
    try:
        sys.exit(_main(sys.argv))
    except KeyboardInterrupt:
        sys.exit(-1)
