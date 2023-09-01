import numpy
import rasterio
import sys
import numpy as np
from sklearn.linear_model import LinearRegression
from . import helpers
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from matplotlib.dates import DateFormatter, MonthLocator, YearLocator
from sklearn.model_selection import train_test_split
import pandas as pd


grid_reliabilities = ["LOW", "HIGH"]
TRAINING_DATASET_FOLDERNAMES = [
    "Bahraich-buffer-5-miles",
    "Barabanki-buffer-5-miles",
    "Kanpur-buffer-5-miles",
    "Sitapur-buffer-5-miles",
    "Varanasi-buffer-5-miles",
]

blue = "#1f77b4"
red = "#d62728"
grey = "grey"
green = "green"


def cosine_similarity(array1, array2):
    """
    Calculates the cosine similarity between two arrays.

    Args:
      array1 (np.ndarray): The first array.
      array2 (np.ndarray): The second array.

    Returns:
      float: The cosine similarity between the two arrays.
    """

    # Calculate the norms of the two arrays.
    norm1 = numpy.linalg.norm(array1, axis=1)
    norm2 = numpy.linalg.norm(array2, axis=1)

    # Calculate the dot product of the two arrays.
    dot_product = numpy.matmul(array1, array2.T)

    # Return the cosine similarity.
    return dot_product / (norm1 * norm2)


def read_tifs_to_low_high_arrays(input_folder):
    all_array = []
    low_array = []
    high_array = []

    for reliability in grid_reliabilities:
        # filter = f"f{location}-barabanki-{reliability}.ti"
        filter = f"{reliability}.tif"
        files = helpers.getAllFilesFromFolderWithFilename(input_folder, filter)

        for filename in files:
            filepath = f"{input_folder}/{filename}"

            with rasterio.open(filepath) as src:
                array = src.read()
                inner_array = array[0]
                all_array.append(inner_array)

                if reliability == "LOW":
                    low_array.append(inner_array)

                elif reliability == "HIGH":
                    high_array.append(inner_array)

    return all_array, high_array, low_array


# Maybe find the next lowest value above 0 for the LIGHT ON cases?
def spreads_of_means_and_stds():
    # input_folder = "./data/07-cropped-images/Bahraich-buffer-1-miles"
    input_folder = "./data/07-cropped-images/Sitapur-buffer-1-miles"
    # input_folder = "./data/07-cropped-images/Varanasi-buffer-1-miles"
    # input_folder = "./data/07-cropped-images/Kanpur-buffer-1-miles"
    # input_folder = "./data/07-cropped-images/Barabanki-buffer-1-miles"

    # Initialize variables to store min and max values
    low_minimum_value, low_maximum_value = float("inf"), float("-inf")
    high_minimum_value, high_maximum_value = float("inf"), float("-inf")

    all_array, high_array, low_array = read_tifs_to_low_high_arrays(input_folder)

    print("len(low_array)", len(low_array))
    print("len(high_array)", len(high_array))

    # Need to make sure all the data is the same shape
    all_array = numpy.array(helpers.get_padded_array(all_array))
    low_array = numpy.array(helpers.get_padded_array(low_array))
    high_array = numpy.array(helpers.get_padded_array(high_array))

    # I.e. an array with the mean values for every pixel
    average_array = numpy.nanmean(all_array, axis=0)
    print("average_array", average_array)
    # I.e. an array with the SD values for every pixel
    std_array = numpy.nanstd(all_array, axis=0)
    print("std_array", std_array)

    low_minimum_value = min(low_minimum_value, numpy.nanmin(low_array))
    low_maximum_value = max(low_maximum_value, numpy.nanmax(low_array))
    high_minimum_value = min(high_minimum_value, numpy.nanmin(high_array))
    high_maximum_value = max(high_maximum_value, numpy.nanmax(high_array))

    # Calculate spreads
    low_min_max_spread = low_maximum_value - low_minimum_value
    high_min_max_spread = high_maximum_value - high_minimum_value

    # Means
    low_mean = numpy.nanmean(low_array, axis=0)
    print("low_mean", low_mean)
    low_minimum_mean_value = numpy.nanmin(low_mean)
    low_maximum_mean_value = numpy.nanmax(low_mean)
    low_mean_min_max_spread = low_maximum_mean_value - low_minimum_mean_value
    high_mean = numpy.nanmean(high_array, axis=0)
    print("high_mean", high_mean)
    high_minimum_mean_value = numpy.nanmin(high_mean)
    high_maximum_mean_value = numpy.nanmax(high_mean)
    high_mean_min_max_spread = high_maximum_mean_value - high_minimum_mean_value

    mean_min_spread = high_minimum_mean_value - low_minimum_mean_value
    mean_max_spread = high_maximum_mean_value - low_maximum_mean_value

    # Standard deviations
    low_std = numpy.nanstd(low_array, axis=0)
    low_minimum_std_value = numpy.nanmin(low_std)
    low_maximum_std_value = numpy.nanmax(low_std)
    low_std_min_max_spread = low_maximum_std_value - low_minimum_std_value
    high_std = numpy.nanstd(high_array, axis=0)
    high_minimum_std_value = numpy.nanmin(high_std)
    high_maximum_std_value = numpy.nanmax(high_std)
    high_std_min_max_spread = high_maximum_std_value - high_minimum_std_value

    std_min_spread = high_minimum_std_value - low_minimum_std_value
    std_max_spread = high_maximum_std_value - low_maximum_std_value

    # Determine spread status'
    mean_min_spread_status = "Positive" if mean_min_spread >= 0 else "Negative"
    mean_max_spread_status = "Positive" if mean_max_spread >= 0 else "Negative"
    std_min_spread_status = "Positive" if std_min_spread >= 0 else "Negative"
    std_max_spread_status = "Positive" if std_max_spread >= 0 else "Negative"

    # Print the comparison for the current location
    print("--- LIGHTS OFF - LOW ---")
    print("Image count", len(low_array))
    print("Actual minimum value: ", low_minimum_value)
    print("Actual maximum value: ", low_maximum_value)
    print("Min-max spread: ", low_min_max_spread)
    print("Mean min, max: ", low_minimum_mean_value, low_maximum_mean_value)
    print("Mean min-max spread: ", low_mean_min_max_spread)
    print("Std min, max: ", low_minimum_std_value, low_maximum_std_value)
    print("Std min-max spread: ", low_std_min_max_spread)

    print()
    print("--- LIGHTS ON - HIGH ---")
    print("Image count", len(high_array))
    print("Actual minimum value: ", high_minimum_value)
    print("Actual maximum value: ", high_maximum_value)
    print("Min-max spread: ", high_min_max_spread)
    print("Mean min, max: ", high_minimum_mean_value, high_maximum_mean_value)
    print("Mean min-max spread: ", high_mean_min_max_spread)
    print("Std min, max: ", high_minimum_std_value, high_maximum_std_value)
    print("Std min-max spread: ", high_std_min_max_spread)
    print()

    print("--- OVERALL ---")
    print("Image count", len(all_array))
    print("Cosign similarity between means of HIGH and LOW", cosine_similarity(low_mean, high_mean))
    print("Spread between mean of LOW min and HIGH min: ", mean_min_spread, f"= {mean_min_spread_status}")
    print("Spread between mean of LOW max and HIGH max: ", mean_max_spread, f"= {mean_max_spread_status}")
    print("Spread between std of LOW min and HIGH min: ", std_min_spread, f"= {std_min_spread_status}")
    print("Spread between std of LOW min and HIGH min: ", std_max_spread, f"= {std_max_spread_status}")

    # Create a list to store all the pixel values.
    # pixel_values = []
    # for data in list(low_array.flat):
    #     pixel_values += data

    # fig, ax = plt.subplots()

    # ax.plot(list(low_array.flat))

    # ax.set_title("LOW values")
    # ax.set_xlabel("Days")
    # ax.set_ylabel("Pixel value")

    # plt.show()

    # --- Uncomment to allow visual plotting ----
    # # Calculate the average value per coordinate across all arrays
    # average_array = numpy.nanmean(stacked_array, axis=0)
    # std_array = numpy.nanstd(stacked_array, axis=0)
    # # Plotting pixel values on a greyscale image
    # plt.imshow(average_array[0], cmap="gray")
    # plt.colorbar()
    # plt.show()
    # # To plot the standard deviation array as a heatmap
    # plt.imshow(std_array[0], cmap="hot", vmin=numpy.nanmin(std_array), vmax=numpy.nanmax(std_array))
    # plt.colorbar()
    # plt.title("Standard Deviation")
    # plt.show()


# ----------------------------

CLASS_MAPPING = {"HIGH": 1, "LOW": 0}
INVERSE_CLASS_MAPPING = {1: "HIGH", 0: "LOW"}


def reshape_original_data(data):
    new_data = None
    # print(f"All data reshaped to {max_size_1st}, {max_size_2nd}")
    # Use same max sizes for training and test to make sure all arrays are the same size.
    new_data = np.array(helpers.get_padded_array(data))
    new_data = helpers.replace_image_nan_with_means(new_data)

    return new_data


def get_y_encoded(ydata):
    # Make sure HIGH is 1 and LOW is 0
    # Create array of 0's and 1's for my dataset
    encoded_Y = [CLASS_MAPPING[label] for label in ydata]
    encoded_Y = np.array(encoded_Y, dtype=int)

    return encoded_Y


def get_all_data():
    training_data_input_folder = "./data/07-cropped-images"

    # Get training dataset
    X_train, y_train, all_items = helpers.get_training_dataset(
        training_data_input_folder, TRAINING_DATASET_FOLDERNAMES
    )

    original = [item["original"] for item in all_items]
    original = reshape_original_data(original)

    encoded_Y = get_y_encoded(y_train)

    means = [item["mean"] for item in all_items]
    medians = [item["median"] for item in all_items]
    scaled = [item["scaled"] for item in all_items]
    scaled_means = [item["scaled_mean"] for item in all_items]
    scaled_medians = [item["scaled_median"] for item in all_items]
    dates = [item["date"] for item in all_items]
    locations = [item["location"] for item in all_items]
    sub_locations = [item["sub-location"] for item in all_items]
    z_scores = [item["z_score"] for item in all_items]
    classifications = [item["classification"] for item in all_items]

    return {
        "scaled": scaled,
        "scaled_means": scaled_means,
        "scaled_medians": scaled_medians,
        "original": original,
        "means": means,
        "medians": medians,
        "dates": dates,
        "locations": locations,
        "sub_locations": sub_locations,
        "encoded_Y": encoded_Y,
        "all_items": all_items,
        "z_scores": z_scores,
        "classifications": classifications,
    }


# -------- PLOTTING -----------


# TODO year value seems a year out
def plot_mean_timeseries():
    data = get_all_data()
    means = data["means"]
    dates = data["dates"]
    # -----------------

    plt.figure(figsize=(10, 6))
    plt.plot(dates, means, marker="o", linestyle="")

    # Set Axis labels
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.title("Values Over Time")

    # Format the x-axis date ticks to display nicely
    date_format = DateFormatter("%Y")
    plt.gca().xaxis.set_major_formatter(date_format)
    years = YearLocator(base=1, month=1, day=1)
    plt.gca().xaxis.set_major_locator(years)

    # Rotate the x-axis date labels for better readability
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()  # Ensures the plot elements fit within the figure area
    plt.show()


# Comparison before/after means/scaled
def plot_overall_vs_scaled():
    data = get_all_data()
    scaled_medians = data["scaled_medians"]
    medians = data["medians"]

    # -----------------

    values = medians
    scaled_values = scaled_medians

    print("values", values[:10])
    print("total", len(values))
    nans_percentage = np.isnan(values).sum() / len(values)
    print("nans_percentage", nans_percentage)
    print("Zero values", sum(i == 0 for i in values))
    print("Values below 1 = ", sum(i < 1 for i in values))
    print("Values below 3 = ", sum(i < 3 for i in values))

    fig, ax = plt.subplots(1, 2, figsize=(15, 3))
    sns.histplot(values, ax=ax[0], kde=True, legend=False)
    ax[0].set_title("Values (original data)")
    sns.histplot(scaled_values, ax=ax[1], kde=True, legend=False)
    ax[1].set_title("Values (scaled)")
    fig.suptitle(f"Overall comparison of median values")
    plt.show()


def get_best_midpoint(data):
    print("data: ", data[:10])

    # Sort the data array based on the values
    sorted_data = data[np.argsort(data[:, 0], kind="stable")]
    print("sorted_data: ", sorted_data[:10])

    # Initialize variables to keep track of the best threshold and counts
    best_threshold = None
    max_difference = 0

    # Iterate through the sorted data and find the best threshold
    for i in range(1, len(sorted_data)):
        # Potential threshold
        threshold = (sorted_data[i - 1, 0] + sorted_data[i, 0]) / 2
        print("threshold: ", threshold)

        # Count true positive (HIGH) values above and true negative (LOW) values below the threshold
        true_positive_above = np.sum(sorted_data[i:, 1] == 1)
        print("true_positive_above: ", true_positive_above)
        true_negative_below = np.sum(sorted_data[:i, 1] == 0)
        print("true_negative_below: ", true_negative_below)

        # Calculate the difference between true positive and true negative counts
        difference = abs(true_positive_above - true_negative_below)
        print("difference: ", difference)

        # Update best threshold if the difference is greater
        if difference > max_difference:
            best_threshold = threshold
            max_difference = difference

    print("Best threshold:", best_threshold)
    print("Max max_difference:", max_difference)


def plot_radiance_histogram_medians():
    data = get_all_data()

    # If using whole dataset
    medians = data["medians"]
    encoded_Y = data["encoded_Y"]

    # If filtering for one sub_location
    # all_items = data["all_items"]
    # sub_location_data = [item for item in all_items if item["sub-location"] == "dashehrabagh-barabanki"]
    # classifications = [item["classification"] for item in sub_location_data]
    # encoded_Y = get_y_encoded(classifications)
    # medians = [item["median"] for item in sub_location_data]

    # -----------------
    values = medians
    print("values: ", values[:10])
    data = np.array(list(zip(values, encoded_Y)))

    get_best_midpoint(data)
    return

    high_medians = [median for median, label in zip(medians, encoded_Y) if INVERSE_CLASS_MAPPING[label] == "HIGH"]
    low_medians = [median for median, label in zip(medians, encoded_Y) if INVERSE_CLASS_MAPPING[label] == "LOW"]

    overall_high_median = np.nanmedian(high_medians)
    overall_low_median = np.nanmedian(low_medians)

    print("overall_high_median", overall_high_median)
    print("overall_low_median", overall_low_median)

    colors = [blue, red]
    cmap = mcolors.ListedColormap(colors)
    x_values = np.arange(len(values))

    # Create the scatter plot
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(x_values, values, c=encoded_Y, cmap=cmap, marker="o", alpha=0.7)
    plt.xlabel("Sample Index")
    plt.ylabel("Radiance")
    plt.title("Scatter Radiance All Locations (Medians)")

    # Draw horizontal lines for overall average values
    plt.axhline(y=overall_high_median, color="orange", linestyle="--", label="Overall HIGH Median")
    plt.axhline(y=overall_low_median, color="black", linestyle="--", label="Overall LOW Median")

    # a, b = np.polyfit(x, y, 1)

    # Create a custom colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label("Class Label")
    cbar.set_ticks([0.25, 0.75])
    cbar.set_ticklabels(["LOW", "HIGH"])

    plt.legend()  # Show legend for lines
    plt.grid(True)
    plt.show()


# TODO: Could give plots colours for locations?
def plot_radiance_histogram_means():
    data = get_all_data()
    means = data["means"]
    encoded_Y = data["encoded_Y"]
    # -----------------
    values = means

    high_means = [mean for mean, label in zip(means, encoded_Y) if INVERSE_CLASS_MAPPING[label] == "HIGH"]
    low_means = [mean for mean, label in zip(means, encoded_Y) if INVERSE_CLASS_MAPPING[label] == "LOW"]

    overall_high_avg = np.nanmean(high_means)
    overall_low_avg = np.nanmean(low_means)

    print("overall_high_avg", overall_high_avg)
    print("overall_low_avg", overall_low_avg)

    colors = [blue, red]
    cmap = mcolors.ListedColormap(colors)
    x_values = np.arange(len(values))

    # Create the scatter plot
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(x_values, values, c=encoded_Y, cmap=cmap, marker="o", alpha=0.7)
    plt.xlabel("Sample Index")
    plt.ylabel("Radiance")
    plt.title("Scatter Radiance All Locations (Means)")

    # Draw horizontal lines for overall average values
    plt.axhline(y=overall_high_avg, color="orange", linestyle="--", label="Overall HIGH Avg")
    plt.axhline(y=overall_low_avg, color="black", linestyle="--", label="Overall LOW Avg")

    # Create a custom colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label("Class Label")
    cbar.set_ticks([0.25, 0.75])
    cbar.set_ticklabels(["LOW", "HIGH"])

    plt.legend()  # Show legend for lines
    plt.grid(True)
    plt.show()


def plot_picket_fence():
    data = get_all_data()
    locations = data["locations"]
    all_items = data["all_items"]
    item_type = "scaled_median"
    # -----------------

    # Build year and location based dictionary
    location_data = {}
    for item in all_items:
        location = item["location"]
        year = item["date"].year

        if year not in location_data:
            location_data[year] = {}
        if location not in location_data[year]:
            location_data[year][location] = []

        # Scaled mean vs mean
        location_data[year][location].append(item[item_type])
        # location_data[year][location].append(item["mean"])

    fig, ax = plt.subplots(figsize=(10, 6))

    years = sorted(set(item["date"].year for item in all_items))
    locations = set()
    for year_locations in location_data.values():
        locations.update(year_locations.keys())

    group_width = 0.4  # Total width for each group of bars
    bar_width = group_width / len(years)  # Width for each individual bar

    for i, year in enumerate(years):
        # Mean per location
        mean_values = [np.mean(location_data[year].get(location, [])) for location in locations]

        x = np.arange(len(locations)) + (i - len(years) / 2) * bar_width  # Adjust the x positions
        ax.bar(x, mean_values, bar_width, label=year, edgecolor="black")

    locations = [location.replace("-", "") for location in locations]
    ax.set_xticks(np.arange(len(locations)))

    ax.set_xticklabels(locations, rotation=45)
    ax.tick_params(axis="x", which="major", labelsize=10)

    ax.set_xlabel("Location")
    ax.set_ylabel("Average Radiance (scaled)")
    ax.set_title("Per location scaled radiance 2014-2019")
    ax.legend()
    plt.tight_layout()
    plt.show()

    # TODO: Not sure mean radiance is valuable as such different data per location?
    # TODO: DO picket fence per location for the sub-locations?


FROM_DATE_COLUMN = "From date"
TO_DATE_COLUMN = "To date"


def plot_ground_truth_vs_real():
    # --- GROUND TRUTH ---
    path = "./data/03-reliability-datasets/Huzurpur-Bahraich-all.csv"
    data = pd.read_csv(path)
    # Convert 'Date' column to datetime
    data["Date"] = pd.to_datetime(data["Date"])
    # Calculate the mean of each row (hour)
    data["Hour Mean"] = data.iloc[:, 3:].mean(axis=1)

    # --- To plot mean of every hour ---
    # # Group the data by 'Hour' and 'Date', and calculate the mean of 'Hour Mean'
    # grouped_data = data.groupby(["Hour", "Date"])["Hour Mean"].mean().reset_index()
    # # Plotting
    # plt.figure(figsize=(10, 6))
    # for hour in grouped_data["Hour"].unique():
    #     hour_data = grouped_data[grouped_data["Hour"] == hour]
    #     plt.scatter(hour_data["Date"], hour_data["Hour Mean"], label=f"Hour {hour}", alpha=0.7)

    # --- To plot pf every day (with regression line) ---
    # Group the data by 'Date' and calculate the mean of 'Hour Mean'
    grouped_data = data.groupby("Date")["Hour Mean"].mean().reset_index()

    # Plotting the "real" data mean values
    fig, ax1 = plt.subplots(figsize=(10, 6))

    colors = [red if value < 130 else blue for value in grouped_data["Hour Mean"]]
    ax1.scatter(grouped_data["Date"], grouped_data["Hour Mean"], c=colors, label="Daily Mean", alpha=0.7)

    # Fit linear regression model
    X = np.array(range(len(grouped_data))).reshape(-1, 1)
    y = grouped_data["Hour Mean"].values
    model = LinearRegression()
    model.fit(X, y)

    # Plot regression line
    plt.plot(grouped_data["Date"], model.predict(X), color=grey, linestyle="--", label="Trend Line")
    # Set labels and title for the primary y-axis (left)
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Ground Truth Voltage", color=blue)
    ax1.tick_params(axis="y", labelcolor=blue)

    # --- REAL DATA ---

    input_folder = "./data/07-cropped-images"
    foldername = "Huzurpur-Bahraich-all-buffer-1-miles"
    real_path = f"{input_folder}/{foldername}"
    filepaths = helpers.getAllFilesFromFolderWithFilename(real_path, "")

    real_data_means = []
    real_data_medians = []
    real_data_dates = []

    for filename in filepaths:
        filepath = f"{real_path}/{filename}"

        with rasterio.open(filepath) as src:
            array = src.read()
            array = array[0]

            # Fill NaN's
            image_mean = np.nanmean(array)
            array[np.isnan(array)] = image_mean  # Replace NaN with image mean

            # Recalculate means and medians after filling nans
            image_mean = np.nanmean(array)
            image_median = np.nanmedian(array)
            print("image_mean: ", image_mean)
            print("image_median: ", image_median)

            # Handle dates
            path_from_julian_date_onwards = filepath.split(f"vnp46a2-a", 1)[1]
            julian_date = path_from_julian_date_onwards.split("-")[0]
            date = helpers.get_datetime_from_julian_date(julian_date)
            print("date: ", date)

            real_data_means.append(image_mean)
            real_data_medians.append(image_median)
            real_data_dates.append(date)

    # Create a secondary y-axis (right)
    ax2 = ax1.twinx()
    radiance_scaled = np.interp(real_data_means, (0, 7), (0, 250))

    # Plot the "real" data mean values on the secondary y-axis
    ax2.scatter(real_data_dates, radiance_scaled, c=green, label="Satellite Radiance Mean", alpha=0.7)
    ax2.set_ylabel("Satellite Radiance Mean", color=green)
    ax2.tick_params(axis="y", labelcolor=green)

    # --- Overall final plotting ---
    # plt.xlabel("Date")
    plt.title("Daily Voltage Mean with Trend Line. Huzurpur, Bahraich.")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# RESULTS
def straight_line_analysis_means():
    data = get_all_data()
    means = data["means"]
    encoded_Y = data["encoded_Y"]
    # -----------------
    values = means

    high_means = [mean for mean, label in zip(values, encoded_Y) if INVERSE_CLASS_MAPPING[label] == "HIGH"]
    low_means = [mean for mean, label in zip(values, encoded_Y) if INVERSE_CLASS_MAPPING[label] == "LOW"]

    overall_high_avg = np.nanmean(high_means)
    overall_low_avg = np.nanmean(low_means)

    print("overall_high_avg", overall_high_avg)
    print("overall_low_avg", overall_low_avg)
    middle = (overall_high_avg + overall_low_avg) / 2
    print("middle", middle)

    # Results
    high_items_count = sum(i > overall_high_avg for i in values)
    low_items_count = sum(i < overall_low_avg for i in values)
    print("high_items_count", high_items_count)
    print("low_items_count", low_items_count)

    print("High correct = ", sum(i > middle for i in high_means))
    print("Low correct = ", sum(i < middle for i in low_means))


def straight_line_analysis_medians():
    data = get_all_data()
    medians = data["medians"]
    encoded_Y = data["encoded_Y"]
    # -----------------
    values = medians

    high_medians = [median for median, label in zip(values, encoded_Y) if INVERSE_CLASS_MAPPING[label] == "HIGH"]
    low_medians = [median for median, label in zip(values, encoded_Y) if INVERSE_CLASS_MAPPING[label] == "LOW"]

    # For overall analysis
    overall_high_median = np.nanmedian(high_medians)
    overall_low_median = np.nanmedian(low_medians)
    print("overall_high_median", overall_high_median)
    print("overall_low_median", overall_low_median)
    middle = (overall_high_median + overall_low_median) / 2
    print("middle", middle)
    # Results
    high_items_count = sum(i > overall_high_median for i in values)
    low_items_count = sum(i < overall_low_median for i in values)
    print("high_items_count", high_items_count)
    print("low_items_count", low_items_count)

    print("High correct = ", sum(i > middle for i in high_medians))
    print("Low correct = ", sum(i < middle for i in low_medians))


def low_value_analysis():
    data = get_all_data()
    medians = data["medians"]
    encoded_Y = data["encoded_Y"]
    # -----------------

    values = medians
    combined = zip(values, encoded_Y)
    count = 0
    low_count = 0

    for item in tuple(combined):
        value, y_value = item
        is_low = value < 2
        is_low_class = INVERSE_CLASS_MAPPING[y_value] == "LOW"

        if is_low:
            count = count + 1

            if is_low_class:
                low_count = low_count + 1

    print("Count", count)
    print("LOW count", low_count)
    # Count 2923
    # LOW count 1597


def get_sub_locations_counts():
    data = get_all_data()
    sub_locations = data["sub_locations"]
    classifications = data["classifications"]

    sub_location_counts = {}

    for sub_location, label in zip(sub_locations, classifications):
        if sub_location not in sub_location_counts:
            sub_location_counts[sub_location] = {"count": 0, "high-count": 0, "low-count": 0}

        sub_location_counts[sub_location]["count"] += 1
        sub_location_counts[sub_location][f"{label.lower()}-count"] += 1

    return sub_location_counts


def get_valuable_locations():
    data = get_sub_locations_counts()
    valuable_overall_count = 100
    valuable_ratio = 0.3

    filtered_data = {}

    for location, counts in data.items():
        count = counts["count"]
        low_count = counts["low-count"]
        high_count = counts["high-count"]

        has_good_low_ratio = low_count / (count) > valuable_ratio
        has_good_high_ratio = high_count / (count) > valuable_ratio

        if count > valuable_overall_count and has_good_low_ratio and has_good_high_ratio:
            filtered_data[location] = counts

    print("filtered_data", filtered_data)


def historical_z_score_analysis():
    data = get_all_data()
    all_items = data["all_items"]

    # valuable locations from 'get_valuable_locations' above
    sub_location_names = [
        "jagatapur-bahraich",
        "huzurpur-bahraich",
        "dashehrabagh-barabanki",
        "devtapur-sitapur[offline]",
        "tedwadih--sitapur",
        "sahdipur-sitapur",
        "jhauwa-khurd--sitapur",
        "khindaura--sitapur",
    ]

    for sub_location_name in sub_location_names:
        sub_location_data = [item for item in all_items if item["sub-location"] == sub_location_name]
        classifications = [item["classification"] for item in sub_location_data]
        encoded_Y = get_y_encoded(classifications)

        # z_scores = data["z_scores"]
        # locations = data["locations"]
        # sub_location = data["sub_locations"]

        # print("len sub_locations", len(sub_location_data))
        # print("len encoded_Y", len(encoded_Y))

        # reandom_state key allows predictable results
        trainX, testX, trainY, testY = train_test_split(sub_location_data, encoded_Y, train_size=0.7, random_state=42)

        trainX_originals = [item["original"] for item in trainX]
        trainX_originals = reshape_original_data(trainX_originals)

        testX_originals = [item["original"] for item in testX]
        testX_originals = reshape_original_data(testX_originals)

        # Calculate Z score array on trainX
        mean = np.nanmean(trainX_originals)
        std_deviation = np.nanstd(trainX_originals)
        z_scores = (testX_originals - mean) / std_deviation

        high = sum(INVERSE_CLASS_MAPPING[label] == "HIGH" for label in testY)
        low = sum(INVERSE_CLASS_MAPPING[label] == "LOW" for label in testY)

        # Results
        high_correct = 0
        low_correct = 0
        size = len(z_scores)

        for item, label in zip(z_scores, testY):
            median = np.nanmedian(item)
            if median > 0:
                if INVERSE_CLASS_MAPPING[label] == "HIGH":
                    high_correct = high_correct + 1
            if median < 0:
                if INVERSE_CLASS_MAPPING[label] == "LOW":
                    low_correct = low_correct + 1

        # print("high_correct", high_correct)
        # print("low_correct", low_correct)
        total_correct = high_correct + low_correct
        # print("total correct", total_correct)
        print(f"----------- {sub_location_name} ------------")
        print("Size", size)
        print("high/low%", f"{high / len(testY)}/{low / len(testY)}")
        print("Accuracy", (100 / size) * total_correct)


if __name__ == "__main__":
    globals()[sys.argv[1]]()
