import os


def filterFilesThatInclude(subString, filenames):
    filtered = []

    for filename in filenames:
        if subString in filename:
            filtered.append(filename)
    return filtered


#  Should be clear that this filters. Name doesn't show that at the mo.
def getAllFilesFromFolderWithFilename(folder, filename):
    owd = os.getcwd()
    print("owd", owd)

    allFiles = os.listdir(f"{owd}{folder}")

    selectedFiles = filterFilesThatInclude(filename, allFiles)

    if not selectedFiles:
        raise RuntimeError(
            f"There are no files in the directory: {folder} with the text: {filename} in the filename \nINFO: All files in {folder}: {allFiles}"
        )

    return selectedFiles


# Drops a filtered table's index back to zero
# https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.reset_index.html
def drop_filtered_table_index(filtered_table):
    return filtered_table.reset_index(drop=True)
