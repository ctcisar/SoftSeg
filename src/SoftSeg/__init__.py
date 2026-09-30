"""SoftSeg -- soft transcript-to-cell assignment for spatial transcriptomics."""

from SoftSeg.SpatialDataHelpers import SpatialDataHelpers

# Reading any store written by spatialdata logs an ome-zarr warning per labels
# element -- hundreds of lines for a per-FOV store, about a layout this package
# uses on purpose. Dropped here, once, so it is gone wherever a store is opened:
# by this package, or by the reader the user calls themselves.
SpatialDataHelpers.quiet_ome_zarr()
