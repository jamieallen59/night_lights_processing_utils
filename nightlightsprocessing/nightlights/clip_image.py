import os
import rasterio

import datetime as dt
import earthpy.spatial as es
from . import constants
import geopandas as gpd
from matplotlib import pyplot
from shapely.geometry import Point
from shapely.ops import cascaded_union
from earthpy.spatial import crop_image

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
    # TODO: change .shp file to a specific coordinate location
    # click point on google earth to get inital coordinate. Maybe don't even need .shp files?
    # Then use a buffer to get the area: https://geopandas.org/en/stable/getting_started/introduction.html#Geometry-relations
    # See if that gives the same results.
    location_path = f"{os.getcwd()}{constants.LOCATION_INPUT_FOLDER}/lucknow-point.shp"
    lucknowRaw = gpd.read_file(location_path)
    # print("-------------------------------")
    # # lucknow = lucknowRaw.set_crs("epsg:4326")
    lucknow = lucknowRaw.to_crs(crs=4326)

    # lucknow["area"] = lucknow.area
    # lucknow["boundary"] = lucknow.boundary
    # lucknow["centroid"] = lucknow.centroid

    # print("lucknow crs_check", lucknow.crs)
    # print("lucknow area", lucknow.area)
    # print("lucknow boundary", lucknow.boundary)
    # print("lucknow centroid", lucknow.centroid)
    # print("lucknow geometry", lucknow.geometry)
    # lucknow.plot("boundary", legend=True)
    # lucknow.plot("area", legend=True)
    # lucknow.explore("area", legend=False)
    # lucknow.plot()

    geotiff_path = f"{os.getcwd()}{constants.OUTPUT_FOLDER}/vnp46a2-a2015001-h25v06-001-2020219190114.tif"
    clip_boundary = lucknow
    location_name = "Lucknow"
    output_folder = f"{os.getcwd()}{constants.OUTPUT_FOLDER}"

    # # print("geotiff_path", geotiff_path)
    # # print("clip_boundary", clip_boundary)
    # # print("location_name", location_name)
    # # print("output_folder", output_folder)

    # # location_path = f"{os.getcwd()}{constants.OUTPUT_FOLDER}/vnp46a2-a2014305-h25v06-001-2020215000151.tif"

    # buffer_distance_miles = 10
    # lucknow_coords = (26.848668, 80.8599406)
    # point = Point(lucknow_coords)
    # # point_gdf = gpd.GeoDataFrame(geometry=[point], crs="EPSG:4326")
    # # buffered_gdf = point_gdf.buffer(buffer_distance_miles * 1609.34)  # Convert miles to meters
    # # merged_buffer = cascaded_union(buffered_gdf.geometry)
    # existing_tif_gdf = gpd.read_file(geotiff_path)
    # buffered_geometry = point.buffer(buffer_distance_miles * 1609.34)  # Convert miles to meters

    # cropped_data, cropped_meta = es.crop_image(existing_tif_gdf, [buffered_geometry])

    # # cropped_gdf = gpd.overlay(
    # #     existing_tif_gdf, gpd.GeoDataFrame(geometry=[merged_buffer], crs="EPSG:4326"), how="intersection"
    # # )
    # print("cropped_meta", cropped_meta)
    # print("cropped_data", cropped_data)
    # try:
    #     print("TESTing image...")
    #     with rasterio.open(location_path) as src:
    #         print("src", src)
    #         array = src.read(1)
    #         print("SHAPE:", array.shape)

    #         src_profile = src.profile
    #         print("src.profile:", src_profile)

    #         crop_bound = clip_boundary.to_crs(src_profile["crs"])
    #         print("crop_bound:", crop_bound)
    #         clip_extent = [es.extent_to_json(crop_bound)]
    #         print("clip_extent:", clip_extent)

    #         # pyplot.imshow(array, cmap="pink")
    #         # pyplot.show()

    #         # cropped_image, cropped_metadata = es.crop_image(raster=src, geoms=crop_bound)

    #         # print("cropped_image", cropped_image)
    #         # print("cropped_metadata", cropped_metadata)

    # except Exception as error:
    #     message = print(f"Failed: {error}\n")
    # else:
    #     message = print(f"Completed")

    # return message

    clip_vnp46a2(geotiff_path, clip_boundary, location_name, output_folder)


# TODO: use crop_all in future so done all at once?
# https://earthpy.readthedocs.io/en/latest/api/earthpy.spatial.html#earthpy.spatial.crop_all
if __name__ == "__main__":
    main()

    # input("Press ENTER to exit")
