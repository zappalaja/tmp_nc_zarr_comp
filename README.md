Build env for .py

conda create -n arraylake-benchmark python=3.11 -y
conda activate arraylake-benchmark


pip install \
arraylake \
xarray \
zarr \
dask \
netcdf4 \
h5netcdf \
numpy \
pandas \
numcodecs \
fsspec \
s3fs \
jupyter \
ipykernel
