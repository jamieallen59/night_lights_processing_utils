#!/usr/bin/env python3

# This file is used to read the "ESMI location information.csv"
# It exports get_location_information, which takes a state and
# location string, and returns the locations present in the spreadsheet.
# These locations can then be used to extract data rom the various ESMI minute-wise voltage data .csv's


from . import helpers
import pandas as pd


LOCATION_INFORMATION_FILENAME = "ESMI location information.csv"

# Column names
LOCATION_NAME_COLUMN = "Location name"
FROM_DATE_COLUMN = "From date"
TO_DATE_COLUMN = "To date"
# DISTRICT_COLUMN = "District" # Unused
STATE_COLUMN = "State"
# CATEGORY_COLUMN = "Category" # Unused
# CONNECTION_TYPE_COLUMN = "Connection Type" # Unused

################################################################################


# Private
def _read_ESMI_location_information_csv(input_folder):
    location_information_files = helpers.getAllFilesFromFolderWithFilename(input_folder, LOCATION_INFORMATION_FILENAME)
    location_information_file = location_information_files[0]
    location_information_file_path = f"{input_folder}/{location_information_file}"

    location_information_dataframe = pd.read_csv(
        location_information_file_path, parse_dates=[FROM_DATE_COLUMN, TO_DATE_COLUMN], dayfirst=True
    )

    return location_information_dataframe


def _get_ESMI_location_information_filtered_by_state(location_information_dataframe, state):
    filtered_df = location_information_dataframe[location_information_dataframe[STATE_COLUMN] == state]
    filtered_df = helpers.drop_filtered_table_index(filtered_df)

    return filtered_df


def _get_ESMI_location_information_filtered_by_location(location_information_dataframe, indian_state_location):
    filtered_df = location_information_dataframe[
        location_information_dataframe[LOCATION_NAME_COLUMN].str.contains(indian_state_location)
    ]
    filtered_df = helpers.drop_filtered_table_index(filtered_df)

    return filtered_df


################################################################################


# Public
def get_ESMI_location_information(input_folder, indian_state, indian_state_location):
    location_information_dataframe = _read_ESMI_location_information_csv(input_folder)
    # Filtering
    filtered_df = _get_ESMI_location_information_filtered_by_state(location_information_dataframe, indian_state)
    filtered_df = _get_ESMI_location_information_filtered_by_location(
        location_information_dataframe, indian_state_location
    )
    # filtered_df = _get_locations_that_started_in(filtered_df, STARTED_BEFORE_DATE)

    return filtered_df
