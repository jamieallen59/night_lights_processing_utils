
import pandas as pd
import numpy as np
import datetime

def fill_data():

    date_range = "NovDec18"
    input_file = pd.read_csv("~/Library/Mobile Documents/com~apple~CloudDocs/India_grid_data/"
                             "Data_for_paper/Nanpara/New_data/" + date_range + ".csv")

    hour_column = input_file["Hour"]
    supply_column = input_file["No Supply"]
    first_datetime = hour_column[0]
    start_date_dt = datetime.datetime.strptime(first_datetime, "%Y-%m-%d %H:%M:%S")

    one_hour_dt = datetime.timedelta(hours=1)
    count_hour_dt = start_date_dt

    for row in range(0, len(input_file)):

        cell_hours_dt = datetime.datetime.strptime(hour_column[row], "%Y-%m-%d %H:%M:%S")
        #print(cell_hours_dt, count_hour_dt, row)
        if cell_hours_dt == count_hour_dt:
            count_hour_dt += one_hour_dt

        elif cell_hours_dt != count_hour_dt:

            #number_of_hours = cell_hours_dt.hour - count_hour_dt.hour
            number_of_hours = cell_hours_dt - count_hour_dt #calculates the amount of time missing
            print("number of hours", number_of_hours)
            time_in_hours = int(((number_of_hours.seconds) / 3600) + ((number_of_hours.days) * 24))
            print("the total hours missing are", time_in_hours)
            new_value = get_this_hour_month_average(hour_column, supply_column, one_hour_dt, time_in_hours, count_hour_dt)
            time_now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S:%f')
            new_value.to_csv("~/Library/Mobile Documents/com~apple~CloudDocs/India_grid_data/Data_for_paper/"
                             "Nanpara/New_data/newdata"+time_now+".csv")

            # string_current_date = count_hour_dt.strftime("%Y-%m-%d %H:%M:%S")
            while cell_hours_dt != count_hour_dt:
                count_hour_dt += one_hour_dt
            count_hour_dt += one_hour_dt


def get_this_hour_month_average(hour_column, supply, one_hour, number_of_hours, current_hour_dt):

    hour_working_on = current_hour_dt
    dates_array = np.array([])
    means_array = np.array([])

    for j in range(0, number_of_hours):

        hour = hour_working_on.hour
        month = hour_working_on.month

        hour_values = np.array([])
        first_hour = hour_column[0]
        dt_first_hour = datetime.datetime.strptime(first_hour, "%Y-%m-%d %H:%M:%S")

        for i in range(0, (len(hour_column))):

            this_hour = dt_first_hour
            if month == this_hour.month:
                if hour == this_hour.hour:
                    hour_values = np.append(hour_values, supply[i])
                else:
                    pass
            else:
                pass
            dt_first_hour += one_hour

        mean_value = float(np.mean(hour_values))
        rounded = round(mean_value, 1)
        str_this_hour_working_on = hour_working_on.strftime("%Y-%m-%d %H:%M:%S")
        dates_array = np.append(dates_array, str_this_hour_working_on)
        means_array = np.append(means_array, rounded)
        hour_working_on += one_hour

    means_df = pd.Series(means_array)
    dates_df = pd.Series(dates_array)
    new_rows = pd.concat([dates_df, means_df], axis=1)
    print(new_rows)
    return new_rows


def generate_grid_profile():

    grid_file = pd.read_csv("~/Library/Mobile Documents/com~apple~CloudDocs/India_grid_data/Data_for_paper/"
                             "Nanpara/Nanpara_whole_year.csv")
    no_supply = grid_file['No Supply']

    n = 1
    #years = 1
    #all_data = pd.DataFrame()
    # for year in range(0,years):

    year_array = np.array([])
    print("Producing grid profile")

    for row in range(0, len(grid_file)):
        outage_time = float(no_supply.iloc[row])
        percentage_of_hour_on = (60 - outage_time) / 60
        year_array = np.append(year_array, percentage_of_hour_on)

    # for row in range(0, len(grid_file)):
    #     outage_time = float(no_supply.iloc[row])
    #     probability_on = (60 - outage_time) / 60
    #     grid_on = np.random.binomial(n, probability_on)
    #     year_array = np.append(year_array, grid_on)

    grid_profile = pd.DataFrame(year_array)
    # year_string = str(year)
    grid_profile.to_csv("~/Library/Mobile Documents/com~apple~CloudDocs/India_grid_data/Data_for_paper/"
                             "Nanpara/percentage_total_nanpara.csv")




def makefilesasone():

    total_file = pd.DataFrame()
    for i in range(1,7):
        istr = str(i)
        input_file = pd.read_csv("~/Library/Mobile Documents/com~apple~CloudDocs/India_grid_data/Data_for_paper/"
                             "Nanpara/New_data/"+istr+".csv")

        total_file = total_file.append(input_file, ignore_index=True)
        print(i)

    total_file.to_csv("~/Library/Mobile Documents/com~apple~CloudDocs/India_grid_data/Data_for_paper/"
                             "Nanpara/New_data/complete_profile.csv")


def calculate_grid_energy():

    """
    This function takes the necessary profiles and examines gives the amount of energy supplied in each hour,
    allowing the user to properly assess the reliability profile against the demand profile
    :return:
    """

    grid_file = pd.read_csv("~/Library/Mobile Documents/com~apple~CloudDocs/India_grid_data/Data_for_paper/"
                             "Nanpara/nanpara_masterfile.csv")
    domestic = grid_file['Domestic']
    phc = grid_file['PHC']
    pathlab = grid_file['Pathlab']
    percentage_profile = grid_file['Percentage profile']
    on_off_profile = grid_file['OO Profile']
    domestic_phc = domestic + phc
    domestic_pathlab = domestic + pathlab
    scenarios = np.array([domestic, phc, pathlab, domestic_phc, domestic_pathlab])
    scenario_list = ["domestic", "phc", "pathlab", "domestic_phc", "domestic_pathlab"]
    for scenario in range(0, len(scenarios)):

        str_scenario = scenario_list[scenario]
        input_file = scenarios[scenario]
        percentage_array = np.array([])
        on_off_array = np.array([])

        for row in range(0, len(grid_file)):

            percentage_value = percentage_profile[row] * input_file[row]
            on_off_value = on_off_profile[row] * input_file[row]
            percentage_array = np.append(percentage_array, percentage_value)
            on_off_array = np.append(on_off_array, on_off_value)

        percentage_df = pd.DataFrame(percentage_array)
        on_off_df = pd.DataFrame(on_off_array)
        scenario_df = pd.concat([percentage_df, on_off_df], axis=1)

        scenario_df.to_csv("~/Library/Mobile Documents/com~apple~CloudDocs/India_grid_data/Data_for_paper/"
                           "Nanpara/" + str_scenario + "_final_output.csv")

        #scenario_df.to_csv("~/Documents/" + str_scenario + "_final_output.csv")


#fill_data()
#makefilesasone()
#generate_grid_profile()
#calculate_grid_energy()

