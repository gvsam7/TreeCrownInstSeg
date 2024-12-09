from rasterio.warp import reproject, Resampling
import rasterio

# Open NDVI and RGB files
with rasterio.open("ndvi_image.tif") as ndvi_src, rasterio.open("rgb_image.tif") as rgb_src:
    ndvi = ndvi_src.read(1)
    rgb_meta = rgb_src.meta

    # Create destination array for upscaled NDVI
    ndvi_upscaled = np.empty((rgb_meta["height"], rgb_meta["width"]), dtype=ndvi.dtype)

    # Perform the reprojection and resampling
    reproject(
        source=ndvi,
        destination=ndvi_upscaled,
        src_transform=ndvi_src.transform,
        src_crs=ndvi_src.crs,
        dst_transform=rgb_src.transform,
        dst_crs=rgb_src.crs,
        resampling=Resampling.cubic  # Choose interpolation method
    )

    # Save the upscaled NDVI
    rgb_meta.update({"count": 1})
    with rasterio.open("ndvi_upscaled.tif", "w", **rgb_meta) as dst:
        dst.write(ndvi_upscaled, 1)
