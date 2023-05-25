from nightlightsprocessing import helpers
import constants
import pandas as pd

folder = constants.GROUND_TRUTH_FOLDER
# file info
LOCATION_INFORMATION = "location information"
# Column names
LOCATION_NAME_COLUMN = "Location name"
STATE_COLUMN = "State"
FROM_DATE_COLUMN = "From date"
TO_DATE_COLUMN = "To date"

# Filter requirements
STATE = "Uttar Pradesh"
STARTED_BEFORE_DATE = "2014-12-31"  # starting with just the ones that started before in 2014


# Drops a filtered table's index back to zero
# https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.reset_index.html
def drop_filtered_table_index(filtered_table):
    return filtered_table.reset_index(drop=True)


def read_location_information_csv():
    location_information_files = helpers.getAllFilesFrom(folder, LOCATION_INFORMATION)
    location_information_file = location_information_files[0]

    location_information_dataframe = pd.read_csv(
        location_information_file, parse_dates=[FROM_DATE_COLUMN, TO_DATE_COLUMN], dayfirst=True
    )

    return location_information_dataframe


def get_location_information_filtered_by(location_information_dataframe, state):
    filtered_df = location_information_dataframe[location_information_dataframe[STATE_COLUMN] == state]
    filtered_df = drop_filtered_table_index(filtered_df)

    return filtered_df


# # get_location_information_for(constants.LOCATIONS_IN_UTTAR_PRADESH)
def get_locations_that_started_in(location_information_dataframe, started_before_date):
    # Convert from and to date columns to datetime types
    location_information_dataframe[FROM_DATE_COLUMN] = pd.to_datetime(location_information_dataframe[FROM_DATE_COLUMN])
    location_information_dataframe[TO_DATE_COLUMN] = pd.to_datetime(location_information_dataframe[TO_DATE_COLUMN])

    filtered_df = location_information_dataframe[
        (location_information_dataframe[FROM_DATE_COLUMN] <= started_before_date)
    ]

    filtered_df = drop_filtered_table_index(filtered_df)

    return filtered_df


def get_location_information():
    location_information_dataframe = read_location_information_csv()

    filtered_df = get_location_information_filtered_by(location_information_dataframe, STATE)
    filtered_df = get_locations_that_started_in(filtered_df, STARTED_BEFORE_DATE)

    return filtered_df


get_location_information()
