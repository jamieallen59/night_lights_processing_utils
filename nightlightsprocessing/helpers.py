import os


def filterFilesThatInclude(subString, filenames):
    filtered = []

    for filename in filenames:
        if subString in filename:
            filtered.append(filename)
    return filtered


#  Should be clear that this filters. Name doesn't show that at the mo.
def getAllFilesFrom(folder, filterRequirement):
    # Get all files in that folder
    os.chdir(folder)

    allFiles = os.listdir(os.getcwd())

    selectedFiles = filterFilesThatInclude(filterRequirement, allFiles)

    if not selectedFiles:
        raise RuntimeError(
            f"There are no files in the directory: {folder} with the text: {filterRequirement} in the filename \nINFO: All files in {folder}: {allFiles}"
        )

    return selectedFiles
