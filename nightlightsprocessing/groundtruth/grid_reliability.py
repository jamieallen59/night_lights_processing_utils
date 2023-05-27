import sys, getopt, os

from nightlightsprocessing import helpers
from . import constants
from .location_information import get_location_information
import pandas as pd

folder = constants.GROUND_TRUTH_FOLDER

# file info
VOLTAGE_DATA_FILENAME = "ESMI minute-wise voltage data"

# Column names
LOCATION_NAME_COLUMN = "Location name"
DATE_COLUMN = "Date"


def get_hours_between(start_hours, end_hours):
    hours = []

    if start_hours <= end_hours:
        for hour in range(start_hours, end_hours + 1):
            hours.append(hour)
    else:
        for hour in range(start_hours, 24):
            hours.append(hour)
        for hour in range(0, end_hours + 1):
            hours.append(hour)

    return hours

def get_minutes_between(start_minutes, end_minutes):
    minutes = []

    if start_minutes <= end_minutes:
        for minute in range(start_minutes, end_minutes + 1):
            minutes.append(minute)
    else:
        for minute in range(start_minutes, 60):
            minutes.append(minute)
        for minute in range(0, end_minutes + 1):
            minutes.append(minute)

    return minutes

def get_hours_query_string(hours):
    query_parts = []

    for hour in hours:
        query_part = "Hour == " + str(hour)
        query_parts.append(query_part)

    query_string = " | ".join(query_parts)

    return query_string

def get_minutes_query_columns(minutes):
    query_array = []

    for minute in minutes:
        query = f"Min {minute}"
        query_array.append(query)
   
    return query_array


def read_voltage_data_csv(year, start_datetime, end_datetime):
     
    filename = f"{VOLTAGE_DATA_FILENAME} {year}"
    print('Looking for file: ', filename)
    voltage_data_files = helpers.getAllFilesFromFolderWithFilename(folder, filename)
    voltage_data_file = voltage_data_files[0]
    voltage_data_file_path = f"{os.getcwd()}{folder}/{voltage_data_file}"
    print('Reading CSV file: ', voltage_data_file_path)

    # Convert the dates to datetime objects
    voltage_dataframe = pd.read_csv(voltage_data_file_path, parse_dates=['Date'], dayfirst=True)

    if start_datetime is not None and end_datetime is not None:
      print(f'Filter data between {start_datetime} and {end_datetime}')

      start_hours = start_datetime.time().hour
      start_minute = start_datetime.time().minute
      end_hours = end_datetime.time().hour
      end_minute = end_datetime.time().minute

      all_hours_between_start_and_end = get_hours_between(start_hours, end_hours)
      all_minutes_between_start_and_end = get_minutes_between(start_minute, end_minute)

      # hours and minutes filtering done differently as hours is a column with a 
      # set of row values whereas minutes are their own column labels.

      # filter the table by the hours selected
      hours_query_string = get_hours_query_string(all_hours_between_start_and_end)
      voltage_dataframe = voltage_dataframe.query(hours_query_string)

      # filter the table by the minutes
      default_columns = ['Location name', 'Date', 'Hour']
      minutes_query_columns = get_minutes_query_columns(all_minutes_between_start_and_end)
      minutes_query_columns = default_columns + minutes_query_columns
      voltage_dataframe = voltage_dataframe[minutes_query_columns]

    return voltage_dataframe


# TODO duplicated
def filter_by_state(data, location_names):
    filtered_df = data[data[LOCATION_NAME_COLUMN].isin(location_names)]
    filtered_df = helpers.drop_filtered_table_index(filtered_df)

    return filtered_df


def main(argv):
    # TODO remove hardcoding
    year = '2014'
    start_datetime = None
    end_datetime = None

    try:
      opts, args = getopt.getopt(argv,"hs:e:",["startdatetime=","enddatetime="])
    except getopt.GetoptError:
      print ('Possible options: -s <startdatetime> -e <enddatetime>')
      sys.exit(2)

    for opt, arg in opts:
      if opt == '-h':
        print ('Possible options: -s <startdatetime> -e <enddatetime>')
        sys.exit()
      elif opt in ("-s", "--startdatetime"):
        start_datetime = pd.to_datetime(arg)
      elif opt in ("-e", "--enddatetime"):
        end_datetime = pd.to_datetime(arg)

    # Maybe should come from Command line?
    indian_state = "Uttar pradesh"
    location_names = get_location_information(indian_state)[LOCATION_NAME_COLUMN]
    voltage_data = read_voltage_data_csv(year, start_datetime, end_datetime)

    # Filtering
    filtered_df = filter_by_state(voltage_data, location_names)
    
    # TODO
    # split the data by area
    # split the data by days
    # scan cell by cell. Count the cells + add them up. Divide by count to get an average.
    # Return true or false based on set number.

    print("RESULTS: ", filtered_df)


if __name__ == "__main__":
    main(sys.argv[1:])
