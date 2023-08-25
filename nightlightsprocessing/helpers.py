import os
import rasterio
import csv
import os
import os.path
import shutil
import sys
from datetime import datetime
from io import StringIO
import matplotlib.pyplot as plt
from . import constants
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import GridSearchCV
from mlxtend.preprocessing import minmax_scaling
import seaborn as sns
import numpy as np

# TODO: find a way to organise better
################################################################################

# Variables
# you will need to replace the following line with the location of a
# python web client library that can make HTTPS requests to an IP address.
USERAGENT = "tis/download.py_1.0--" + sys.version.replace("\n", "").replace("\r", "")

################################################################################


def write_to_csv(data, filename):
    # Write to a .csv file
    with open(filename, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(data)


def filterFilesThatInclude(subString, filenames):
    filtered = []

    for filename in filenames:
        if subString in filename:
            filtered.append(filename)
    return filtered


# TODO:Should be clear that this filters. Name doesn't show that at the mo.
# TODO: should be able to remove filter capability from this now as directories are
# structured differently. Or make 'filename' arg optional.
def getAllFilesFromFolderWithFilename(folder, filename):
    allFiles = os.listdir(folder)

    selectedFiles = filterFilesThatInclude(filename, allFiles)

    # if not selectedFiles:
    #     raise RuntimeError(
    #         f"There are no files in the directory: {folder} with the text: {filename} in the filename \nINFO: All files in {folder}: {allFiles}"
    #     )

    return selectedFiles


# Drops a filtered table's index back to zero
# https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.reset_index.html
def drop_filtered_table_index(filtered_table):
    return filtered_table.reset_index(drop=True)


def get_reliability_dataset_filename(state, location, grid_reliability):
    return f"{constants.VOLTAGE_DATA_FILENAME} - {state} - {location} - filtered unique {grid_reliability}.csv"


# https://ladsweb.modaps.eosdis.nasa.gov/learn/how-to-use-laads-daac-post-processing-tools/
# SDS (subdataset processing)
def getSubDataset(name, dataset):
    for subdataset in dataset:
        if name in subdataset[0]:
            return subdataset[0]


def export_array(array, output_path, metadata):
    # Write numpy array to GeoTiff
    try:
        with rasterio.open(output_path, "w", **metadata) as dst:
            dst.write(array, 1)
    except Exception as error:
        output_message = print(f"ERROR: {error}")
    else:
        output_message = print(f"Exported: {os.path.split(output_path)[-1]}")

    return output_message


def _get_base_filepath(filepath):
    return f"{os.path.basename(filepath)[:-3].lower().replace('.', '-')}"


def get_hd5_to_tif_export_name(filepath):
    return f"{_get_base_filepath(filepath)}.tif"


def get_tif_to_clipped_export_name(filepath, location_name, reliability="OFF"):
    image_country = location_name.replace(" ", "-").lower()

    export_name = f"{_get_base_filepath(filepath)}clipped-{image_country}-{reliability}.tif"
    return export_name


def get_datetime_from_julian_date(julian_date):
    year = int(julian_date[:4])
    day_of_year = int(julian_date[4:])

    # Create a datetime object using the year and day of year
    full_datetime = datetime.strptime(f"{year}-{day_of_year}", "%Y-%j")

    date_only = full_datetime.date()

    return date_only


# tile_descriptor: e.g. h26v06
def _filter_only_tiles(all_files_content, tile_descriptor):
    filtered = []

    for file_content in all_files_content:
        file_name = file_content["name"]

        if tile_descriptor in file_name:
            filtered.append(file_content)
    return filtered


def get_file_details_for_selected_tile(src, token, tile_descriptor):
    try:
        #  Reads a .csv file which represents all the data from the url given
        all_tiles_for_one_day = [
            f for f in csv.DictReader(StringIO(geturl("%s.csv" % src, token)), skipinitialspace=True)
        ]
        # filter for only the tiles needed
        file_details_for_selected_tile = _filter_only_tiles(all_tiles_for_one_day, tile_descriptor)

        # Because there's only one tile image per day, so should only ever be one returned
        first_and_only_item = file_details_for_selected_tile[0]

        return first_and_only_item

    except ImportError as e:
        print("IMPORT ERROR", e)
    except IndexError as e:
        print("INDEX ERROR", e)

    return None


# --- Download helpers ---
# This is the choice of last resort, when other attempts have failed
def getcURL(url, headers=None, out=None):
    import subprocess

    try:
        print("trying cURL", file=sys.stderr)
        # --fail: Specifies that cURL should fail if the HTTP response indicates an error.
        # -sS: Enables silent mode and shows errors if they occur.
        # -L: Instructs cURL to follow redirects if the server responds with a redirect.
        # -b session: Uses the cookie data from the file named session.
        # --get: Specifies that you want to perform an HTTP GET request.
        args = ["curl", "--fail", "-sS", "-L", "-b session", "--get", url]
        for k, v in headers.items():
            args.extend(["-H", ": ".join([k, v])])
        if out is None:
            # python3's subprocess.check_output returns stdout as a byte string
            result = subprocess.run(args, check=True, capture_output=True)
            # result2 = subprocess.check_output(args)
            is_bytes_instance = isinstance(result, bytes)
            print("is_bytes_instance", is_bytes_instance)

            if is_bytes_instance:
                return result.decode("utf-8")
            else:
                return result
        else:
            subprocess.call(args, stdout=out)
    except subprocess.CalledProcessError as e:
        print("curl GET error message: %" + (e.message if hasattr(e, "message") else e.output), file=sys.stderr)
    return None


# read the specified URL and output to a file
def geturl(url, token=None, out=None):
    headers = {"user-agent": USERAGENT}

    if not token is None:
        headers["Authorization"] = "Bearer " + token
    try:
        import ssl
        from urllib.request import urlopen, Request, URLError, HTTPError

        CTX = ssl.SSLContext(ssl.PROTOCOL_TLSv1_2)
        try:
            request = Request(url, headers=headers)

            fh = urlopen(request, context=CTX)
            if out is None:
                return fh.read().decode("utf-8")
            else:
                shutil.copyfileobj(fh, out)
        except HTTPError as e:
            print("TLSv1_2 : HTTP GET error code: %d" % e.code, file=sys.stderr)
            getcURL(url, headers, out)
        except URLError as e:
            print("TLSv1_2 : Failed to make request: %s" % e.reason, file=sys.stderr)
            return getcURL(url, headers, out)

    except AttributeError:
        return getcURL(url, headers, out)


async def sync(src, destination, token, file_details):
    print("Attempting download of:", file_details["name"])
    # currently we use filesize of 0 to indicate directory
    filesize = int(file_details["size"])
    path = os.path.join(destination, file_details["name"])
    url = src + "/" + file_details["name"]
    if filesize == 0:  # size FROM RESPONSE
        try:
            print("creating dir:", path)
            os.mkdir(path)
            sync(src + "/" + file_details["name"], path, token)
        except IOError as e:
            print("mkdir `%s': %s" % (e.filename, e.strerror), file=sys.stderr)
            sys.exit(-1)
    else:
        try:
            if not os.path.exists(path) or os.path.getsize(path) == 0:  # filesize FROM OS
                print("\nDownloading to...", path)
                with open(path, "w+b") as fh:
                    geturl(url, token, fh)
            else:
                print("Skipping, as already downloaded here:", path)
        except IOError as e:
            print("open `%s': %s" % (e.filename, e.strerror), file=sys.stderr)
            sys.exit(-1)


label_index = 0


# Model plotting functions
# @title Define the plotting functions
def plot_the_model(y_test, predictions):
    fpr, tpr, _ = roc_curve(y_test, predictions)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color="darkorange", lw=lw, label="ROC curve (area = %0.2f)" % roc_auc)
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])

    # Set Axis labels
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.legend(loc="lower right")
    plt.show()


def plot_the_loss_curve(epochs, train_loss, val_loss):
    # Plot the loss curve
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, epochs + 1), train_loss, label="Training Loss")
    plt.plot(range(1, epochs + 1), val_loss, label="Validation Loss")

    # Set Axis labels
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss Curve")
    plt.legend()
    plt.grid(True)
    plt.show()


# plot the first 9 images in the planet dataset
from matplotlib import pyplot
from matplotlib.image import imread


def show_multiple_images():
    # define location of dataset
    folder = "train-jpg/"
    # plot first few images
    for i in range(9):
        # define subplot
        pyplot.subplot(330 + 1 + i)
        # define filename
        filename = folder + "train_" + str(i) + ".jpg"
        # load image pixels
        image = imread(filename)
        # plot raw pixel data
        pyplot.imshow(image)
    # show the figure
    pyplot.show()


# Hyper parameter tuning
def hyperparameter_tuning(keras_model, X_train, y_train):
    hyperparameters = {
        "hidden_layer_dim": [80],
        "loss": ["binary_crossentropy"],
        "batch_size": [4, 8, 12, 20],
        "epochs": [10],
        "learning_rate": [0.001],
        "optimizer": ["adam"],
    }

    grid_search = GridSearchCV(keras_model, hyperparameters, cv=5)
    grid_search.fit(X_train, y_train)

    print("Best Hyperparameters:", grid_search.best_params_)


def get_training_dataset(input_folder, training_dataset_foldernames):
    low = []
    high = []

    training_data = []
    training_data_classifications = []

    overall = []

    for directory in training_dataset_foldernames:
        sub_directories_path = f"{input_folder}/{directory}"
        location_sub_directories = os.listdir(sub_directories_path)

        for location_sub_directory in location_sub_directories:
            if not location_sub_directory.startswith("."):
                files_path = f"{input_folder}/{directory}/{location_sub_directory}"

                filepaths = getAllFilesFromFolderWithFilename(files_path, "")

                temp = []

                for filename in filepaths:
                    filepath = f"{files_path}/{filename}"

                    with rasterio.open(filepath) as src:
                        array = src.read()
                        array = array[0]

                        image_mean = np.nanmean(array)
                        image_median = np.nanmedian(array)

                        array[np.isnan(array)] = image_mean  # Replace NaN with image mean
                        # array[np.isnan(array)] = 0  # OR replace Nan with zero?
                        # Handle dates
                        path_from_julian_date_onwards = filepath.split(f"vnp46a2-a", 1)[1]
                        julian_date = path_from_julian_date_onwards.split("-")[0]
                        date = get_datetime_from_julian_date(julian_date)

                        classification = "LOW" if "LOW" in filepath else "HIGH"
                        location = directory.replace("buffer-1-miles", "")

                        item = {
                            "classification": classification,
                            "original": array,
                            "mean": image_mean,
                            "median": image_median,
                            "date": date,
                            "location": location,
                            "sub-location": location_sub_directory,
                        }

                        temp.append(item)

                # # Normalise and scale data per location
                all_temp_items = [item["original"] for item in temp]
                all_temp_items = np.array(all_temp_items)

                # print("all_temp_items", all_temp_items[:3])

                overall_mean = np.nanmean(all_temp_items)
                overall_std_deviation = np.nanstd(all_temp_items)
                # print("Overall Mean:", overall_mean)
                # print("Overall Standard Deviation:", overall_std_deviation)

                # Standard normalisation (- mean / SD)
                # normalised = (all_temp_items - overall_mean) / overall_std_deviation

                # MIN/MAX scaling
                # Add all scaled items and scaled means to temp array
                column_indices = np.arange(all_temp_items.shape[1])
                scaled = minmax_scaling(all_temp_items, columns=column_indices)

                scaled_image_mean = np.nanmean(scaled)
                scaled[np.isnan(scaled)] = scaled_image_mean  # Replace NaN with image mean

                # Add scaled array and means to dataset
                for i in range(len(temp)):
                    scaled_array = scaled[i]
                    scaled_mean = np.nanmean(scaled_array)
                    scaled_median = np.nanmedian(scaled_array)
                    temp[i]["scaled"] = scaled_array
                    temp[i]["scaled_mean"] = scaled_mean
                    temp[i]["scaled_median"] = scaled_median

                # # process arrays with mean and std
                for item in temp:
                    data = item["scaled"]

                    if item["classification"] == "LOW":
                        training_data.append(data)
                        low.append(data)
                        training_data_classifications.append("LOW")
                    else:
                        training_data.append(data)
                        high.append(data)
                        training_data_classifications.append("HIGH")

                    overall.append(item)

                # ------ PLOT MEANS AND SCALED. GOOD FOR THESIS. --------
                # But this is per location
                # fig, ax = plt.subplots(1, 2, figsize=(15, 3))
                # all_means = [item["mean"] for item in temp]
                # sns.histplot(all_means, ax=ax[0], kde=True, legend=False)
                # ax[0].set_title("Mean values (original data)")
                # all_scaled_means = [item["scaled_mean"] for item in temp]
                # sns.histplot(all_scaled_means, ax=ax[1], kde=True, legend=False)
                # ax[1].set_title("Mean values (scaled)")
                # fig.suptitle(f"Comparison of mean values ({directory}/{location_sub_directory})")
                # plt.show()

    print("Training set total size", len(training_data))
    print("HIGH items", training_data_classifications.count("HIGH"))
    print("LOW items", training_data_classifications.count("LOW"))

    return training_data, training_data_classifications, overall
