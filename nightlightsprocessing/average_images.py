import numpy
import rasterio
import sys
from . import helpers

import matplotlib.pyplot as plt

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


def average_images():
    input_folder = "./data/07-cropped-images/Bahraich-buffer-1-miles"

    # Initialize variables to store min and max values
    low_minimum_value, low_maximum_value = float("inf"), float("-inf")
    high_minimum_value, high_maximum_value = float("inf"), float("-inf")

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

    # print("--- OVERALL STATS ---")

    # print("--- Mean ---")
    # print("Spread between mean of min LOW and min HIGH: ", mean_spread_min, f"= {spread_status_min}")
    # print("Spread between mean of max LOW and max HIGH: ", mean_spread_max, f"= {spread_status_max}")

    # print("--- Standard Deviation ---")
    # print("Spread between std of min LOW and min HIGH: ", std_spread_min, f"= {std_spread_status_min}")
    # print("Spread between std of min LOW and min HIGH: ", std_spread_max, f"= {std_spread_status_max}")

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


if __name__ == "__main__":
    globals()[sys.argv[1]]()
