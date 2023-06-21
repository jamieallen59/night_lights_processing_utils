import os
import rasterio

import datetime as dt
import earthpy.spatial as es
from . import constants
import geopandas as gpd

# https://earthpy.readthedocs.io/en/latest/api/earthpy.spatial.html#earthpy.spatial.crop_image


def extract_date(geotiff_path):
    """Extracts the file date from a preprocessed VNP46A1 GeoTiff.

    Parameters
    ----------
    geotiff_path : str
        Path to the GeoTiff file.

    Returns
    -------
    date : str
        Acquisition date of the preprocessed VNP46A1 GeoTiff.

    Example
    -------
        >>>
        >>>
        >>>
        >>>
    """
    # Get date (convert YYYYJJJ to YYYYMMDD)
    date = dt.datetime.strptime(os.path.basename(geotiff_path)[9:16], "%Y%j").strftime("%Y%m%d")

    return date


def create_clipped_export_name(image_path, location_name):
    """Creates a file name indicating a clipped file.

    Paramaters
    ----------
    image_path : str
        Path to the original (unclipped image).

    Returns
    -------
    export_name : str
        New file name for export, indicating clipping.

    Example
    -------
        >>>
        >>>
        >>>
        >>>
    """
    # Set export name
    image_source = os.path.basename(image_path)[:7]
    image_date = extract_date(image_path)
    image_country = location_name.replace(" ", "-").lower()
    export_name = f"{image_source}-{image_date}-clipped-{image_country}.tif"

    return export_name


def export_array(array, output_path, metadata):
    """Exports a numpy array to a GeoTiff.

    Parameters
    ----------
    array : numpy array
        Numpy array to be exported to GeoTiff.

    output_path : str
        Path to the output file (includeing filename).

    metadata : dict
        Dictionary containing the metadata required
        for export.

    Returns
    -------
    output_message : str
        Message indicating success or failure of export.

    Example
    -------
        >>> # Define export output paths
        >>> radiance_mean_outpath = os.path.join(
        ...     output_directory,
        ...     "radiance-mean.tif")
        # Define export transform
        >>> transform = from_origin(
        ...     lon_min,
        ...     lat_max,
        ...     coord_spacing,
        ...     coord_spacing)
        >>> # Define export metadata
        >>> export_metadata = {
        ...     "driver": "GTiff",
        ...     "dtype": radiance_mean.dtype,
        ...     "nodata": 0,
        ...     "width": radiance_mean.shape[1],
        ...     "height": radiance_mean.shape[0],
        ...     "count": 1,
        ...     "crs": 'epsg:4326',
        ...     "transform": transform
        ... }
        >>> # Export mean radiance
        >>> export_array(
        >>>     array=radiance_mean,
        >>>     output_path=radiance_mean_outpath,
        >>>     metadata=export_metadata)
        Exported: radiance-mean.tif
    """
    # Write numpy array to GeoTiff
    try:
        with rasterio.open(output_path, "w", **metadata) as dst:
            dst.write(array, 1)
    except Exception as error:
        output_message = print(f"ERROR: {error}")
    else:
        output_message = print(f"Exported: {os.path.split(output_path)[-1]}")

    return output_message


def clip_vnp46a2(geotiff_path, clip_boundary, location_name, output_folder):
    """Clips an image to a bounding box and exports the clipped image to
    a GeoTiff file.

    Paramaters
    ----------
    geotiff_path : str
        Path to the GeoTiff image to be clipped.

    clip_boundary : geopandas geodataframe
        Geodataframe for containing the boundary used for clipping.

    location_name : str
        Name of the country the data is being clipped to. The country
        name is used in the name of the exported file. E.g. 'South Korea'.
        Spaces and capital letters are acceptable and handled within the
        function.

    output_folder : str
        Path to the folder where the clipped file will be exported to.

    Returns
    -------
    message : str
        Indication of concatenation completion status (success
        or failure).

    Example
    -------
        >>>
        >>>
        >>>
        >>>
    """
    # Clip VNP46A2 file
    print(f"Started clipping: Clip {os.path.basename(geotiff_path)} " f"to {location_name} boundary")
    try:
        print("Clipping image...")
        # Clip image (return clipped array and new metadata)
        with rasterio.open(geotiff_path) as src:
            cropped_image, cropped_metadata = es.crop_image(raster=src, geoms=clip_boundary)

        print("Setting export name...")
        # Set export name
        export_name = create_clipped_export_name(image_path=geotiff_path, location_name=location_name)

        print("Exporting to GeoTiff...")
        # Export file
        export_array(
            array=cropped_image[0],
            output_path=os.path.join(output_folder, export_name),
            metadata=cropped_metadata,
        )
    except Exception as error:
        message = print(f"Clipping failed: {error}\n")
    else:
        message = print(f"Completed clipping: Clip {os.path.basename(geotiff_path)} " f"to {location_name} boundary\n")

    return message


def main():
    try:
        location_path = f"{os.getcwd()}{constants.LOCATION_INPUT_FOLDER}/lucknow-point.shp"
        lucknowRaw = gpd.read_file(location_path)

        # here we change the crs system to 32634 to allow us to set a buffer in metres
        # https://epsg.io/32634
        lucknow_unit_metres = lucknowRaw.to_crs(crs=32634)
        buffer_distance_miles = 10
        buffer_distance_metres = buffer_distance_miles * 1609.34  # Convert miles to meters
        buffered_lucknow = lucknow_unit_metres.buffer(buffer_distance_metres)

        # Then change it back to allow the crop images
        # https://epsg.io/4326
        lucknow = buffered_lucknow.to_crs(crs=4326)

        geotiff_path = f"{os.getcwd()}{constants.OUTPUT_FOLDER}/vnp46a2-a2014305-h26v06-001-2020214163912.tif"

        clip_boundary = lucknow
        location_name = "Lucknow"
        output_folder = f"{os.getcwd()}{constants.OUTPUT_FOLDER}"
        clip_vnp46a2(geotiff_path, clip_boundary, location_name, output_folder)

    except Exception as error:
        message = print(f"Failed: {error}\n")
    else:
        message = print(f"Completed")

    return message


# TODO: use crop_all in future so done all at once?
# https://earthpy.readthedocs.io/en/latest/api/earthpy.spatial.html#earthpy.spatial.crop_all
if __name__ == "__main__":
    main()

    # input("Press ENTER to exit")
