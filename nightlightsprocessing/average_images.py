import numpy
import rasterio
from . import helpers

# import matplotlib.pyplot as plt

# Bahraich
locations = [
    "bahadurpur",
    "bardaha",
    "huzurpur",
    "jagatapur",
    "kanjaya",
    "kurmaura",
    "mihinpurwa",
    "puraini",
]

grid_reliabilities = ["LOW", "HIGH"]


def main():
    input_folder = "./data/07-cropped-images/Bahraich-buffer-1-miles"

    for location in locations:
        # Initialize variables to store min and max values
        min_low, max_low = float("inf"), float("-inf")
        min_high, max_high = float("inf"), float("-inf")

        std_min_low, std_max_low = float("inf"), float("-inf")
        std_min_high, std_max_high = float("inf"), float("-inf")

        low_data_count = 0
        high_data_count = 0

        for reliability in grid_reliabilities:
            filter = f"{location}-bahraich-{reliability}.tif"
            files = helpers.getAllFilesFromFolderWithFilename(input_folder, filter)

            stacked_array = []

            for filename in files:
                filepath = f"{input_folder}/{filename}"

                with rasterio.open(filepath) as src:
                    array = src.read()
                    stacked_array.append(array)

                if reliability == "LOW":
                    low_data_count = low_data_count + 1
                elif reliability == "HIGH":
                    high_data_count = high_data_count + 1

            stacked_array = numpy.array(stacked_array)

            average_array = numpy.nanmean(stacked_array, axis=0)
            std_array = numpy.nanstd(stacked_array, axis=0)

            # Update min and max values based on reliability
            if reliability == "LOW":
                min_low = min(min_low, numpy.nanmin(average_array))
                max_low = max(max_low, numpy.nanmax(average_array))
            elif reliability == "HIGH":
                min_high = min(min_high, numpy.nanmin(average_array))
                max_high = max(max_high, numpy.nanmax(average_array))

            if reliability == "LOW":
                std_min_low = min(std_min_low, numpy.nanmin(std_array))
                std_max_low = max(std_max_low, numpy.nanmax(std_array))
            elif reliability == "HIGH":
                std_min_high = min(std_min_high, numpy.nanmin(std_array))
                std_max_high = max(std_max_high, numpy.nanmax(std_array))

        # Calculate spreads
        mean_spread_min = min_high - min_low
        mean_spread_max = max_high - max_low
        std_spread_min = std_min_high - std_min_low
        std_spread_max = std_max_high - std_max_low

        # Determine spread status
        spread_status_min = "Positive" if mean_spread_min >= 0 else "Negative"
        spread_status_max = "Positive" if mean_spread_max >= 0 else "Negative"
        std_spread_status_min = "Positive" if std_spread_min >= 0 else "Negative"
        std_spread_status_max = "Positive" if std_spread_max >= 0 else "Negative"

        # Print the comparison for the current location
        print(f"Location: {location}")
        print("LOW image count", low_data_count)
        print("HIGH image count", high_data_count)
        print("--- Mean ---")
        print("Lights OFF mean min, max: ", min_low, max_low)
        print("Lights ON mean min, max: ", min_high, max_high)
        print("Min values spread: ", mean_spread_min, f"= {spread_status_min}")
        print("Max values spread: ", mean_spread_max, f"= {spread_status_max}")
        print("--- Standard Deviation ---")
        print("Lights OFF std min, max: ", std_min_low, std_max_low)
        print("Lights ON std min, max: ", std_min_high, std_max_high)
        print("Min values spread: ", std_spread_min, f"= {std_spread_status_min}")
        print("Max values spread: ", std_spread_max, f"= {std_spread_status_max}")
        print()

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
    main()
