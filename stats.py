#!/usr/bin/env python3

import time
import json
import statistics
from pathlib import Path
from datetime import datetime

import xarray as xr
import zarr
from arraylake import Client

# ============================================================
# USER SETTINGS
# ============================================================

API_KEY = ""

NETCDF_PATH = "/home/zappalaj/TEMP/Benchmark/atmos_4xdaily_avg.1921010100-1930123123.precip.nc"
NETCDF_VARIABLE = "pr"

ARRAYLAKE_REPO = "GFDL/noaa-gfdl-spear-large-ensembles-pds"
ARRAYLAKE_BRANCH = "main"
ARRAYLAKE_GROUP = "historical/6hr"
ARRAYLAKE_MEMBER = "r1i1p1f1"
ARRAYLAKE_VARIABLE = "pr"

START_TIME = "1921-01-01T00:00:00"
END_TIME = "1930-12-31T23:00:00"

TRIALS = 5

OUTPUT_DIR = "/home/zappalaj/TEMP/Benchmark/Output"

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_FILENAME = f"netcdf_vs_arraylake_benchmark_{timestamp}.txt"

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
OUTPUT_PATH = Path(OUTPUT_DIR) / OUTPUT_FILENAME


# ============================================================
# HELPERS
# ============================================================

def time_trial(func):
    start = time.perf_counter()
    result = func()
    end = time.perf_counter()
    return end - start, result


def summarize(times):
    return {
        "runs": times,
        "median": statistics.median(times),
        "mean": statistics.mean(times),
        "min": min(times),
        "max": max(times),
    }


def estimate_mb(da):
    return da.nbytes / 1024 / 1024


def get_coord_name(da, options):
    for name in options:
        if name in da.coords or name in da.dims:
            return name
    raise ValueError(f"Could not find coordinate from options: {options}")


# ============================================================
# OPENERS
# ============================================================

def open_netcdf():
    return xr.open_dataset(NETCDF_PATH)


def open_arraylake_zarr():
    client = Client(token=API_KEY)
    repo = client.get_repo(ARRAYLAKE_REPO)

    session = repo.readonly_session(
        branch=ARRAYLAKE_BRANCH
    )

    ds = xr.open_zarr(
        session.store,
        group=ARRAYLAKE_GROUP,
        zarr_format=3,
        consolidated=False,
    )

    # Select ensemble member 1
    if "member_id" in ds.dims or "member_id" in ds.coords:
        ds = ds.sel(member_id=ARRAYLAKE_MEMBER)

    # Match the selected NetCDF file time range
    ds = ds.sel(
        time=slice(
            START_TIME,
            END_TIME
        )
    )

    return ds


# ============================================================
# BENCHMARKS
# ============================================================

def run_benchmarks(label, opener, variable, trials=5):
    results = {}

    # --------------------------------------------------------
    # 1. Open dataset
    # --------------------------------------------------------
    open_times = []

    for _ in range(trials):
        t, ds_tmp = time_trial(opener)
        open_times.append(t)
        ds_tmp.close()

    results["open_dataset"] = summarize(open_times)

    # Keep one dataset open for remaining tests
    ds = opener()

    if variable not in ds.data_vars:
        raise ValueError(
            f"Variable '{variable}' not found in {label}. "
            f"Available variables: {list(ds.data_vars)}"
        )

    da = ds[variable]

    lat_name = get_coord_name(da, ["lat", "latitude", "y"])
    lon_name = get_coord_name(da, ["lon", "longitude", "x"])

    # --------------------------------------------------------
    # 2. Variable listing
    # --------------------------------------------------------
    times = []

    for _ in range(trials):
        t, _ = time_trial(lambda: list(ds.data_vars))
        times.append(t)

    results["variable_listing"] = summarize(times)

    # --------------------------------------------------------
    # 3. Point time series
    # --------------------------------------------------------
    times = []

    for _ in range(trials):
        def point_time_series():
            mid_lat = da[lat_name].values[len(da[lat_name]) // 2]
            mid_lon = da[lon_name].values[len(da[lon_name]) // 2]

            return da.sel(
                {
                    lat_name: mid_lat,
                    lon_name: mid_lon,
                },
                method="nearest",
            ).load()

        t, out = time_trial(point_time_series)
        times.append(t)

    results["point_time_series"] = summarize(times)
    results["point_time_series"]["loaded_mb"] = estimate_mb(out)
    results["point_time_series"]["shape"] = list(out.shape)

    # --------------------------------------------------------
    # 4. Single map
    # --------------------------------------------------------
    times = []

    for _ in range(trials):
        def single_map():
            return da.isel(time=0).load()

        t, out = time_trial(single_map)
        times.append(t)

    results["single_map"] = summarize(times)
    results["single_map"]["loaded_mb"] = estimate_mb(out)
    results["single_map"]["shape"] = list(out.shape)

    # --------------------------------------------------------
    # 5. Approx 10 MB read
    # --------------------------------------------------------
    times = []

    for _ in range(trials):
        def approx_10mb_read():
            subset = da.isel(
                time=slice(0, 80),
                **{
                    lat_name: slice(0, 180),
                    lon_name: slice(0, 180),
                }
            )

            return subset.load()

        t, out = time_trial(approx_10mb_read)
        times.append(t)

    results["approx_10mb_read"] = summarize(times)
    results["approx_10mb_read"]["loaded_mb"] = estimate_mb(out)
    results["approx_10mb_read"]["shape"] = list(out.shape)

    ds.close()

    return {
        "backend": label,
        "variable": variable,
        "trials": trials,
        "results": results,
    }


# ============================================================
# OUTPUT
# ============================================================

def write_results(all_results, output_path):
    with output_path.open("w") as f:
        f.write("NetCDF vs Arraylake Zarr Benchmark Results\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"NetCDF path: {NETCDF_PATH}\n")
        f.write(f"NetCDF variable: {NETCDF_VARIABLE}\n\n")

        f.write(f"Arraylake repo: {ARRAYLAKE_REPO}\n")
        f.write(f"Arraylake branch: {ARRAYLAKE_BRANCH}\n")
        f.write(f"Arraylake group: {ARRAYLAKE_GROUP}\n")
        f.write(f"Arraylake member: {ARRAYLAKE_MEMBER}\n")
        f.write(f"Arraylake variable: {ARRAYLAKE_VARIABLE}\n\n")

        f.write(f"Time range: {START_TIME} to {END_TIME}\n")
        f.write(f"Trials per test: {TRIALS}\n")
        f.write(f"Output file: {output_path}\n\n")

        for backend_result in all_results:
            f.write("=" * 60 + "\n")
            f.write(f"Backend: {backend_result['backend']}\n")
            f.write("=" * 60 + "\n\n")

            for test_name, stats in backend_result["results"].items():
                f.write(f"Test: {test_name}\n")
                f.write(f"Runs: {stats['runs']}\n")
                f.write(f"Median: {stats['median']:.6f} sec\n")
                f.write(f"Mean: {stats['mean']:.6f} sec\n")
                f.write(f"Min: {stats['min']:.6f} sec\n")
                f.write(f"Max: {stats['max']:.6f} sec\n")

                if "loaded_mb" in stats:
                    f.write(f"Loaded MB: {stats['loaded_mb']:.3f}\n")

                if "shape" in stats:
                    f.write(f"Loaded shape: {stats['shape']}\n")

                f.write("\n")

        f.write("\nRaw JSON\n")
        f.write("=" * 60 + "\n")
        f.write(json.dumps(all_results, indent=2))


# ============================================================
# MAIN
# ============================================================

def main():
    all_results = []

    print("Running NetCDF benchmarks...")
    all_results.append(
        run_benchmarks(
            label="netcdf_hpc",
            opener=open_netcdf,
            variable=NETCDF_VARIABLE,
            trials=TRIALS,
        )
    )

    print("Running Arraylake benchmarks...")
    all_results.append(
        run_benchmarks(
            label="arraylake_zarr",
            opener=open_arraylake_zarr,
            variable=ARRAYLAKE_VARIABLE,
            trials=TRIALS,
        )
    )

    write_results(all_results, OUTPUT_PATH)

    print("Done. Results written to:")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
