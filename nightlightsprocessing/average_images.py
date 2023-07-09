import numpy
from . import helpers
import rasterio
import matplotlib.pyplot as plt


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


def main2():
    input_folder = "./data/07-cropped-images/Bahraich-buffer-1-miles"

    results = []

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

            # print(filter)
            # print("Mean min, max: ")
            # print(numpy.nanmin(average_array), numpy.nanmax(average_array))
            # print("STD min, max: ")
            # print(numpy.nanmin(std_array), numpy.nanmax(std_array))

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


def main():
    input_folder = "./data/07-cropped-images/Bahraich-buffer-1-miles"

    filter = "puraini-bahraich-HIGH.tif"
    files = helpers.getAllFilesFromFolderWithFilename(input_folder, filter)

    stacked_array = []

    for filename in files:
        filepath = f"{input_folder}/{filename}"

        with rasterio.open(filepath) as src:
            # myarray = numpy.array(src)
            myarray = src.read()
            print(myarray)
            stacked_array.append(myarray)
            # min_value = numpy.nanmin(myarray)
            # max_value = numpy.nanmax(myarray)
            # stacked_array.append(numpy.array(myarray))
            # Calculate the sum of non-nan numbers
            sum_non_nan = numpy.nansum(myarray)

            # Calculate the average of non-nan numbers
            count_non_nan = numpy.count_nonzero(~numpy.isnan(myarray))
            average_non_nan = sum_non_nan / count_non_nan

            print("Sum of non-nan numbers:", sum_non_nan)
            print("Average of non-nan numbers:", average_non_nan)

        # print("imarray.shape", imarray.shape)

    # Calculate the average value per coordinate across all arrays
    average_array = numpy.nanmean(stacked_array, axis=0)
    std_array = numpy.nanstd(stacked_array, axis=0)

    # Plotting pixel values on a greyscale image
    plt.imshow(average_array[0], cmap="gray")
    plt.colorbar()
    plt.show()

    # To plot the standard deviation array as a heatmap
    plt.imshow(std_array[0], cmap="hot", vmin=numpy.nanmin(std_array), vmax=numpy.nanmax(std_array))
    plt.colorbar()
    plt.title("Standard Deviation")
    plt.show()

    print("Average Array:")
    print(average_array)
    print("\nStandard Deviation Array:")
    print(std_array)


if __name__ == "__main__":
    main2()


# vikas-nagar--lucknow-[offline]-LOW
# Mean:
# [[[         nan 127.6        126.23636364 136.60909091 143.41666667
#             nan]
#   [137.3        142.11818182 143.25454545 157.84545455 150.70833333
#    132.66666667]
#   [131.4        142.42727273 150.25454545 168.33636364 176.35833333
#    154.2       ]
#   [125.3        140.70909091 144.80909091 168.43636364 170.10833333
#    150.03333333]
#   [         nan 129.04545455 140.92727273 132.74545455 143.65
#             nan]]]
# Standard deviation:
# [[[         nan 107.66812646 109.92754864 118.2489243  110.21836533
#             nan]
#   [100.56275291 113.96275164 121.5460359  127.18444276 122.38776803
#    118.82879374]
#   [ 91.09759003 104.0904335  114.64012754 134.26684606 140.23183458
#    117.08020755]
#   [ 92.6596313  103.11969616 106.72479864 146.16528796 147.69442812
#    116.98978825]
#   [         nan  95.73724803  90.35090271  78.31085335 121.63590821
#             nan]]]

# vikas-nagar--lucknow-[offline]-HIGH
# Mean:
# [[[         nan  82.37692308  77.80769231  79.45714286  77.92857143
#             nan]
#   [ 90.64615385  85.73846154  85.36153846  97.88571429  98.65714286
#     84.00714286]
#   [ 91.49230769  98.67692308  98.36153846 102.62857143 105.07142857
#     86.65714286]
#   [ 84.36153846  94.76923077 102.53846154 104.25       104.1
#     88.2       ]
#   [         nan  88.53076923 102.22307692 104.42857143  86.13076923
#             nan]]]
# Standrad deviation
# [[[        nan 12.99864809 11.55418416 12.4897509  15.05805093
#            nan]
#   [15.00767062 11.41581484 16.14737599 13.98186873 12.93868586
#    16.12156782]
#   [14.42078956 12.49776311 14.60914037 15.40247793 14.04972946
#    16.98729417]
#   [10.25081899 10.92938063 10.60671329 15.35404786 18.564938
#    14.72117037]
#   [        nan  8.71324232 13.35504248 12.75832661 12.08478921
#            nan]]]


# Bahadurpur HIGH
# Average Array:
# [[[        nan 12.1127451   9.26666667  8.38316832  7.35833333
#            nan]
#   [        nan  7.60707071  5.67        4.44653465  3.90909091
#     3.27789474]
#   [ 2.529       2.40612245  2.71752577  2.68888889  2.58947368
#     2.19148936]
#   [ 1.8393617   1.57525773  1.65360825  1.76145833  1.79270833
#     1.75257732]
#   [        nan  1.33578947  1.37446809  1.8606383   1.94787234
#     1.83548387]
#   [        nan         nan         nan  1.80105263         nan
#            nan]]]

# Standard Deviation Array:
# [[[       nan 5.7539189  4.69132143 5.54108095 5.26669304        nan]
#   [       nan 6.4466772  4.45415536 2.12237324 1.74031757 1.44740632]
#   [1.31173892 1.09691742 1.59108407 1.36299499 1.17235469 0.96774135]
#   [0.96262254 0.65880725 0.71638581 0.84881901 0.79519976 0.79094004]
#   [       nan 0.54539523 0.59357126 1.16484115 1.22489502 0.90311827]
#   [       nan        nan        nan 1.14012878        nan        nan]]]

# Bahadurpur LOW
# Average Array:
# [[[        nan 10.76285714  9.22        8.1         7.859375
#            nan]
#   [        nan  7.34285714  5.55        4.40588235  3.82647059
#     3.409375  ]
#   [ 2.259375    2.259375    2.35483871  2.5969697   2.734375
#     2.26363636]
#   [ 1.7483871   1.56451613  1.6375      1.82424242  1.8
#     1.65588235]
#   [        nan  1.35625     1.390625    1.69375     1.690625
#     1.64117647]
#   [        nan         nan         nan  1.678125           nan
#            nan]]]

# Standard Deviation Array:
# [[[       nan 5.19335211 4.62612457 4.78790142 4.77204486        nan]
#   [       nan 5.97479399 2.60470728 2.02498238 1.67723667 1.44013007]
#   [0.95028265 1.1285664  1.11002433 1.09336424 1.15747823 0.85593053]
#   [0.67099876 0.59598029 0.63035208 1.18705909 0.89171269 0.75349819]
#   [       nan 0.5249628  0.57682286 1.27302236 0.93886879 0.756197  ]
#   [       nan        nan        nan 0.92288081        nan        nan]]]
