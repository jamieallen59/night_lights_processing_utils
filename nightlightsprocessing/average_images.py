import numpy
import rasterio
import sys
import numpy as np
from . import helpers
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from matplotlib.dates import DateFormatter, MonthLocator, YearLocator


grid_reliabilities = ["LOW", "HIGH"]


# TODO: copied in 08_run_model
def get_padded_array(parent_array):
    if not parent_array:
        return []

    max_size_1st = max(arr.shape[0] for arr in parent_array)
    max_size_2nd = max(arr.shape[1] for arr in parent_array)

    # Pad arrays with nan values to match the maximum size
    padded_arrays = []

    for arr in parent_array:
        pad_width = (
            (0, max_size_1st - arr.shape[0]),
            (0, max_size_2nd - arr.shape[1]),
        )
        padded_arr = numpy.pad(arr, pad_width, constant_values=numpy.nan)
        padded_arrays.append(padded_arr)

    return padded_arrays


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
    all_array = numpy.array(get_padded_array(all_array))
    low_array = numpy.array(get_padded_array(low_array))
    high_array = numpy.array(get_padded_array(high_array))

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


def myFunc(directory):
    if ".DS_Store" in directory:
        return False
    else:
        return True


TRAINING_DATASET_FOLDERNAMES = [
    "Bahraich-buffer-1-miles",
    "Barabanki-buffer-1-miles",
    "Kanpur-buffer-1-miles",
    "Sitapur-buffer-1-miles",
    "Varanasi-buffer-1-miles",
]

CLASS_MAPPING = {"HIGH": 1, "LOW": 0}
INVERSE_CLASS_MAPPING = {1: "HIGH", 0: "LOW"}


def replace_image_nan_with_zeros(lights_data_combined):
    # Perform mean imputation per image
    updated_images = np.copy(
        lights_data_combined
    )  # Create a copy of the original images to store the updated versions

    # Perform mean imputation per image
    for i in range(updated_images.shape[0]):
        image = updated_images[i]
        image[np.isnan(image)] = 0  # Replace NaN with image mean

    return updated_images


def get_padded_array(max_size_1st, max_size_2nd, parent_array):
    # Pad arrays with nan values to match the maximum size
    padded_arrays = []

    for arr in parent_array:
        pad_width = (
            (0, max_size_1st - arr.shape[0]),
            (0, max_size_2nd - arr.shape[1]),
        )
        padded_arr = np.pad(arr, pad_width, constant_values=np.nan)
        padded_arrays.append(padded_arr)

    return padded_arrays


def reshape_original_data(data):
    new_data = None
    # Reshape data and pad with nan's
    max_size_1st = max(arr.shape[0] for arr in data)
    max_size_2nd = max(arr.shape[1] for arr in data)
    print(f"All data reshaped to {max_size_1st}, {max_size_2nd}")
    # Use same max sizes for training and test to make sure all arrays are the same size.
    new_data = np.array(get_padded_array(max_size_1st, max_size_2nd, data))
    new_data = replace_image_nan_with_zeros(new_data)

    return new_data


def get_all_data():
    training_data_input_folder = "./data/07-cropped-images"

    # Get training dataset
    X_train, y_train, all_items = helpers.get_training_dataset(
        training_data_input_folder, TRAINING_DATASET_FOLDERNAMES
    )

    original = [item["original"] for item in all_items]
    original = reshape_original_data(original)

    # Make sure HIGH is 1 and LOW is 0
    # Create array of 0's and 1's for my dataset
    encoded_Y = [CLASS_MAPPING[label] for label in y_train]
    encoded_Y = np.array(encoded_Y, dtype=int)

    means = [item["mean"] for item in all_items]
    medians = [item["median"] for item in all_items]
    scaled = [item["scaled"] for item in all_items]
    scaled_means = [item["scaled_mean"] for item in all_items]
    scaled_medians = [item["scaled_median"] for item in all_items]
    dates = [item["date"] for item in all_items]
    locations = [item["location"] for item in all_items]
    sub_locations = [item["sub-location"] for item in all_items]

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
    fig.suptitle(f"Overall comparison of mean values")
    plt.show()


# TODO: Could give plots colours for locations?
def plot_radiance_histogram():
    data = get_all_data()
    means = data["means"]
    medians = data["medians"]
    encoded_Y = data["encoded_Y"]
    # -----------------

    values = medians

    high_means = [mean for mean, label in zip(means, encoded_Y) if INVERSE_CLASS_MAPPING[label] == "HIGH"]
    low_means = [mean for mean, label in zip(means, encoded_Y) if INVERSE_CLASS_MAPPING[label] == "LOW"]

    overall_high_avg = np.nanmean(high_means)
    overall_low_avg = np.nanmean(low_means)

    print("overall_high_avg", overall_high_avg)
    print("overall_low_avg", overall_low_avg)

    blue = "#1f77b4"
    red = "#d62728"
    colors = [blue, red]
    cmap = mcolors.ListedColormap(colors)

    x_values = np.arange(len(values))
    # jitter_amount = 0.1
    # jittered_x = np.array(range(len(x_values))) + np.random.uniform(-jitter_amount, jitter_amount, len(x_values))

    # Create the scatter plot
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(x_values, values, c=encoded_Y, cmap=cmap, marker="o", alpha=0.7)
    plt.xlabel("Sample Index")
    plt.ylabel("Radiance")
    plt.title("Full Dataset Mean Values")

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


# RESULTS
def straight_line_analysis():
    data = get_all_data()
    means = data["means"]
    encoded_Y = data["encoded_Y"]
    # -----------------
    values = means

    high_means = [mean for mean, label in zip(values, encoded_Y) if label == 1]
    low_means = [mean for mean, label in zip(values, encoded_Y) if label == 0]

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


if __name__ == "__main__":
    globals()[sys.argv[1]]()
