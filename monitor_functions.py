#!/usr/bin/env python3

"""
monitor_functions.py — Processing functions for RRFSv2X monitoring.

For one complete ensemble cycle:

1. Read exactly one mpasout* file from each ensemble member.
2. Read latCell, lonCell, and landmask from the MPAS invariant file.
3. For each configured variable:
   - Compute ensemble spread (STD across members) at every MPAS grid cell.
   - Compute spread statistics over LAND cells only:
       * 1st percentile
       * 10th percentile
       * spatial mean
       * 90th percentile
       * 99th percentile
   - Compute minimum and maximum of the original variable values
     across all members and LAND cells only.
   - Plot ensemble spread using ALL MPAS grid cells
     (land + ocean + sea ice).
4. Write statistics to:
       data/YYYYMMDDHH.txt
5. Write figures to:
       figure/YYYYMMDDHH/YYYYMMDDHH_VARIABLE_spread.png
   For tslb layer 1, also write per-cell ensemble extrema over land:
       figure/YYYYMMDDHH/YYYYMMDDHH_tslb_layer1_ens_min.png
       figure/YYYYMMDDHH/YYYYMMDDHH_tslb_layer1_ens_max.png
"""

import datetime as dt
import json
import math
import re
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import xarray as xr


# ----------------------------------------------------------------------
# Map
# ----------------------------------------------------------------------

def create_map_axis():
    """Create a CONUS Lambert Conformal map axis."""

    projection = ccrs.LambertConformal(
        central_longitude=-97.5,
        central_latitude=36.0,
        standard_parallels=(36.0, 36.0),
    )

    fig = plt.figure(
        figsize=(12, 8)
    )

    ax = plt.axes(
        projection=projection
    )

    ax.coastlines(
        resolution="50m",
        linewidth=0.7,
    )

    ax.add_feature(
        cfeature.BORDERS,
        linewidth=0.7,
    )

    ax.add_feature(
        cfeature.STATES,
        linewidth=0.5,
        edgecolor="gray",
    )

    return fig, ax


# ----------------------------------------------------------------------
# Plot spread
# ----------------------------------------------------------------------

def plot_spread(
    lon: np.ndarray,
    lat: np.ndarray,
    spread: np.ndarray,
    cycle_time,
    variable_name: str,
    long_name: str,
    units: str,
    n_land: int,
    spread_spatial_mean: float,
    figure_dir: Path,
) -> Path:
    """
    Plot ensemble spread for all MPAS grid cells.

    The spatial plot includes all cells.

    Statistics shown in the title are calculated over land only.

    The land spatial-mean spread is at 25% of the color scale:
        minimum = 0
        25%     = land spatial mean
        maximum = 4 × land spatial mean

    Values greater than the maximum colorbar value are clipped
    to the red end of the jet colormap.
    """

    cycle_string = (
        cycle_time.strftime(
            "%Y%m%d%H"
        )
    )

    valid_time = (
        cycle_time
        + dt.timedelta(hours=1)
    )

    figure_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure_file = (
        figure_dir
        / f"{cycle_string}_{variable_name}_spread.png"
    )

    # ------------------------------------------------------------------
    # Precision settings
    # ------------------------------------------------------------------

    if variable_name in ("q2", "smois"):
        title_mean_fmt = ".6f"
        tick_fmt = "%.6f"
    else:
        title_mean_fmt = ".3f"
        tick_fmt = "%.3f"

    # ------------------------------------------------------------------
    # Color scale
    #
    # Spatial mean over land is exactly at 25% of the colorbar.
    #
    # Example:
    #
    # mean = 0.000480
    #
    # 0.000000 --- 0.000480 ----------- 0.001920
    #
    # Values >= 0.001920 are plotted using the red end of jet.
    # ------------------------------------------------------------------

    vmin = 0.0

    vcenter = float(
        spread_spatial_mean
    )

    vmax = (
        4.0
        * vcenter
    )

    # Safety check in case spread is exactly zero.
    if vmax <= 0.0:
        vmax = 1.0e-12
        vcenter = 0.25 * vmax

    # ------------------------------------------------------------------
    # Create map
    # ------------------------------------------------------------------

    fig, ax = create_map_axis()

    # ------------------------------------------------------------------
    # Plot all MPAS grid cells
    # ------------------------------------------------------------------

    scatter = ax.scatter(
        lon,
        lat,
        c=spread,
        s=0.6,
        alpha=0.9,
        linewidths=0,
        cmap="jet",
        vmin=vmin,
        vmax=vmax,
        transform=ccrs.PlateCarree(),
        rasterized=True,
    )

    # ------------------------------------------------------------------
    # Colorbar
    # ------------------------------------------------------------------

    colorbar = fig.colorbar(
        scatter,
        ax=ax,
        orientation="horizontal",
        pad=0.05,
        shrink=0.8,
        extend="max",
    )

    colorbar.set_label(
        f"Ensemble spread ({units})"
    )

    # ------------------------------------------------------------------
    # Put the land spatial mean exactly at the 25% tick.
    #
    # Five equally spaced ticks:
    #
    #   0
    #   mean
    #   2.0 * mean
    #   3.0 * mean
    #   4.0 * mean
    # ------------------------------------------------------------------

    tick_values = np.array(
        [
            vmin,
            vcenter,
            2.0 * vcenter,
            3.0 * vcenter,
            vmax,
        ]
    )

    colorbar.set_ticks(
        tick_values
    )

    colorbar.ax.xaxis.set_major_formatter(
        mticker.FormatStrFormatter(
            tick_fmt
        )
    )

    # ------------------------------------------------------------------
    # Three-line title
    # ------------------------------------------------------------------

    ax.set_title(
        f"RRFSv2X {long_name} Ensemble Spread\n"
        f"Initialization: "
        f"{cycle_time:%Y-%m-%d %H UTC}  "
        f"|  Valid: "
        f"{valid_time:%Y-%m-%d %H UTC}\n"
        f"Land grid cells: "
        f"{n_land:,}  "
        f"|  Spatial mean spread over land: "
        f"{spread_spatial_mean:{title_mean_fmt}} "
        f"{units}",
        fontsize=12,
    )

    fig.tight_layout()

    # ------------------------------------------------------------------
    # Save figure
    # ------------------------------------------------------------------

    fig.savefig(
        figure_file,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)

    return figure_file



# ----------------------------------------------------------------------
# Plot layer-1 soil-temperature ensemble minima and maxima
# ----------------------------------------------------------------------

def plot_tslb_ensemble_extreme(
    lon: np.ndarray,
    lat: np.ndarray,
    values: np.ndarray,
    land_mask: np.ndarray,
    ensemble: np.ndarray,
    cycle_time: dt.datetime,
    extreme: str,
    figure_dir: Path,
) -> Path:
    """Plot per-cell minimum/maximum across members for tslb layer 1.

    Only land cells are displayed because soil temperature is a land field.
    The annotated global extreme is likewise restricted to land cells.
    The MPAS cell index in the title is 1-based (Python array index + 1).
    """
    if extreme not in ("min", "max"):
        raise ValueError("extreme must be 'min' or 'max'")

    if not (lon.shape == lat.shape == values.shape == land_mask.shape):
        raise ValueError("Longitude, latitude, values, and land mask must match")

    land_indices = np.flatnonzero(land_mask & np.isfinite(values))
    if land_indices.size == 0:
        raise ValueError("No finite soil temperature values over land")

    land_values = values[land_indices]
    local_index = (np.argmin(land_values) if extreme == "min"
                   else np.argmax(land_values))
    cell_index = int(land_indices[local_index])
    extreme_value = float(values[cell_index])
    extreme_lon = float(lon[cell_index])
    extreme_lat = float(lat[cell_index])

    # Identify the member contributing the extreme at this MPAS cell.
    # Member rows are ordered mem001, mem002, ..., mem030.
    if ensemble.ndim != 2 or ensemble.shape[1] != values.size:
        raise ValueError("Expected ensemble shape (nMembers, nCells)")
    cell_member_values = ensemble[:, cell_index]
    if not np.any(np.isfinite(cell_member_values)):
        raise ValueError("No finite member values at extreme MPAS cell")
    member_index = (np.nanargmin(cell_member_values) if extreme == "min"
                    else np.nanargmax(cell_member_values))
    member_name = f"mem{member_index + 1:03d}"

    cycle_string = cycle_time.strftime("%Y%m%d%H")
    valid_time = cycle_time + dt.timedelta(hours=1)
    figure_dir.mkdir(parents=True, exist_ok=True)
    figure_file = figure_dir / f"{cycle_string}_tslb_layer1_ens_{extreme}.png"

    fig, ax = create_map_axis()
    # Use a 10 K color range anchored to the actual land extreme.
    # Values outside this range are clipped to the end of the colormap.
    if extreme == "min":
        vmin, vmax = extreme_value, extreme_value + 10.0
        colorbar_extend = "max"
    else:
        vmin, vmax = extreme_value - 10.0, extreme_value
        colorbar_extend = "min"

    scatter = ax.scatter(
        lon[land_indices], lat[land_indices], c=land_values,
        s=0.6, alpha=0.9, linewidths=0,
        cmap="jet", vmin=vmin, vmax=vmax,
        transform=ccrs.PlateCarree(), rasterized=True,
    )
    # Mark the extreme location with a large hollow circle (no fill).
    ax.plot(
        extreme_lon, extreme_lat, marker="o", markersize=20,
        markerfacecolor="none", markeredgecolor="black",
        markeredgewidth=2.0, linestyle="None",
        transform=ccrs.PlateCarree(), zorder=10,
    )
    colorbar = fig.colorbar(
        scatter, ax=ax, orientation="horizontal",
        pad=0.05, shrink=0.8, extend=colorbar_extend,
    )
    colorbar.set_label("Soil temperature (K)")
    colorbar.ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))

    label = "Minimum" if extreme == "min" else "Maximum"
    ax.set_title(
        f"RRFSv2X Soil temperature (layer 1) Ensemble {label}\n"
        f"Initialization: {cycle_time:%Y-%m-%d %H UTC}  "
        f"|  Valid: {valid_time:%Y-%m-%d %H UTC}\n"
        f"Over land, {label.lower()}: {extreme_value:.3f} K "
        f"at lon {extreme_lon:.3f}, lat {extreme_lat:.3f} "
        f"| MPAS cell index (1-based): {cell_index + 1:,} "
        f"| Member: {member_name}",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(figure_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return figure_file


# ----------------------------------------------------------------------
# Process one cycle
# ----------------------------------------------------------------------

def process_cycle(
    cycle_time,
    config: dict,
    output_dir: Path,
) -> Path:
    """Process one complete RRFS ensemble cycle."""

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    monitor_config = (
        config["common"]["rrfs_monitor"]
    )

    rrfs_com = Path(
        monitor_config["rrfs_com"]
    )

    invariant_file = Path(
        monitor_config["invariant_file"]
    )

    num_members = int(
        monitor_config.get(
            "num_members",
            30,
        )
    )

    file_pattern = (
        monitor_config.get(
            "forecast_file_pattern",
            "mpasout*",
        )
    )

    spread_ddof = int(
        monitor_config.get(
            "spread_ddof",
            1,
        )
    )

    variables = (
        monitor_config[
            "monitor_variables"
        ]
    )

    # ------------------------------------------------------------------
    # Cycle information
    # ------------------------------------------------------------------

    pdy = (
        cycle_time.strftime(
            "%Y%m%d"
        )
    )

    cyc = (
        cycle_time.strftime(
            "%H"
        )
    )

    cycle_string = (
        cycle_time.strftime(
            "%Y%m%d%H"
        )
    )

    # ------------------------------------------------------------------
    # Output paths
    # ------------------------------------------------------------------

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = (
        output_dir
        / f"{cycle_string}.txt"
    )

    figure_dir = (
        output_dir.parent
        / "figure"
        / cycle_string
    )

    # ------------------------------------------------------------------
    # Find exactly one MPAS file from each member
    # ------------------------------------------------------------------

    member_files = []

    for member in range(
        1,
        num_members + 1,
    ):

        member_name = (
            f"mem{member:03d}"
        )

        member_dir = (
            rrfs_com
            / f"rrfs.{pdy}"
            / cyc
            / "fcst"
            / "enkf"
            / member_name
        )

        mpas_files = sorted(
            path
            for path in member_dir.glob(
                file_pattern
            )
            if path.is_file()
        )

        if len(mpas_files) != 1:

            raise RuntimeError(
                f"{member_name}: expected "
                f"exactly 1 {file_pattern} file, "
                f"found {len(mpas_files)}"
            )

        member_files.append(
            mpas_files[0]
        )

    # ------------------------------------------------------------------
    # Read invariant/static grid
    # ------------------------------------------------------------------

    if not invariant_file.is_file():

        raise FileNotFoundError(
            f"Invariant file not found: "
            f"{invariant_file}"
        )

    with xr.open_dataset(
        invariant_file
    ) as static_ds:

        required_variables = [
            "latCell",
            "lonCell",
            "landmask",
        ]

        for name in required_variables:

            if name not in static_ds:

                raise KeyError(
                    f"'{name}' not found in "
                    f"{invariant_file}"
                )

        # --------------------------------------------------------------
        # Coordinates are stored in radians.
        # --------------------------------------------------------------

        lat = np.degrees(
            static_ds[
                "latCell"
            ].values
        )

        lon = np.degrees(
            static_ds[
                "lonCell"
            ].values
        )

        land_mask = (
            static_ds[
                "landmask"
            ].values
            == 1
        )

    # ------------------------------------------------------------------
    # Normalize longitude to -180 ... 180
    # ------------------------------------------------------------------

    lon = np.where(
        lon > 180.0,
        lon - 360.0,
        lon,
    )

    # ------------------------------------------------------------------
    # Grid counts
    # ------------------------------------------------------------------

    n_total = int(
        land_mask.size
    )

    n_land = int(
        np.count_nonzero(
            land_mask
        )
    )

    n_ocean = (
        n_total
        - n_land
    )

    # ------------------------------------------------------------------
    # Output header
    # ------------------------------------------------------------------

    lines = [
        "=" * 80,
        "RRFSv2X Ensemble Variable Monitor",
        "=" * 80,
        f"Cycle: {cycle_string}",
        f"Members: {num_members}",
        f"Files processed: {len(member_files)}",
        f"Spread STD ddof: {spread_ddof}",
        f"Total grid cells: {n_total}",
        f"Land grid cells: {n_land}",
        f"Ocean grid cells: {n_ocean}",
        "",
    ]

    # ------------------------------------------------------------------
    # Process each configured variable
    # ------------------------------------------------------------------

    for variable_config in variables:

        variable_name = (
            variable_config[
                "variable"
            ]
        )

        display_name = (
            variable_config[
                "name"
            ]
        )

        long_name = (
            variable_config.get(
                "long_name",
                display_name,
            )
        )

        units = (
            variable_config.get(
                "units",
                "",
            )
        )

        # Optional soil-layer selection (1-based, so layer 1 is index 0).
        layer = variable_config.get("layer")
        if layer is not None:
            layer = int(layer)
            if layer < 1:
                raise ValueError(f"{display_name}: layer must be >= 1")

        # --------------------------------------------------------------
        # Full-domain fields from all members
        #
        # Shape after stacking:
        #
        #   (num_members, nCells)
        # --------------------------------------------------------------

        member_fields = []

        overall_minimum = None
        overall_maximum = None

        # --------------------------------------------------------------
        # Read all members
        # --------------------------------------------------------------

        for member_index, mpas_file in enumerate(
            member_files,
            start=1,
        ):

            member_name = (
                f"mem{member_index:03d}"
            )

            with xr.open_dataset(
                mpas_file
            ) as ds:

                if variable_name not in ds:

                    raise KeyError(
                        f"Variable "
                        f"'{variable_name}' "
                        f"not found in "
                        f"{mpas_file}"
                    )

                field = ds[variable_name]

                if layer is not None:
                    # Remove singleton dimensions such as Time, but retain
                    # the soil dimension until after selecting the layer.
                    field = field.squeeze(drop=True)
                    other_dims = [d for d in field.dims if d != "nCells"]
                    if len(other_dims) != 1 or "nCells" not in field.dims:
                        raise ValueError(
                            f"{variable_name}: expected nCells and one soil "
                            f"dimension; got {field.dims}"
                        )
                    soil_dim = other_dims[0]
                    if layer > field.sizes[soil_dim]:
                        raise ValueError(
                            f"{variable_name}: requested layer {layer}, "
                            f"but only {field.sizes[soil_dim]} available"
                        )
                    field = field.isel({soil_dim: layer - 1})

                data = field.squeeze().values

                # ------------------------------------------------------
                # Verify shape
                # ------------------------------------------------------

                if data.ndim != 1:

                    raise ValueError(
                        f"{variable_name} in "
                        f"{member_name} has "
                        f"unexpected shape "
                        f"{data.shape}"
                    )

                if data.size != n_total:

                    raise ValueError(
                        f"{variable_name} in "
                        f"{member_name} has "
                        f"{data.size} cells, "
                        f"but invariant file has "
                        f"{n_total} cells"
                    )

                # ------------------------------------------------------
                # Land-only values for min/max statistics
                # ------------------------------------------------------

                land_values = (
                    data[
                        land_mask
                    ]
                )

                member_min = float(
                    np.nanmin(
                        land_values
                    )
                )

                member_max = float(
                    np.nanmax(
                        land_values
                    )
                )

                if (
                    overall_minimum is None
                    or member_min
                    < overall_minimum
                ):

                    overall_minimum = (
                        member_min
                    )

                if (
                    overall_maximum is None
                    or member_max
                    > overall_maximum
                ):

                    overall_maximum = (
                        member_max
                    )

                # ------------------------------------------------------
                # Save full field for spread calculation
                # ------------------------------------------------------

                member_fields.append(
                    data
                )

        # --------------------------------------------------------------
        # Stack all ensemble members
        #
        # Shape:
        #
        #   (num_members, nCells)
        # --------------------------------------------------------------

        ensemble = np.stack(
            member_fields,
            axis=0,
        )

        # --------------------------------------------------------------
        # Ensemble spread over full domain
        # --------------------------------------------------------------

        ensemble_spread = (
            np.nanstd(
                ensemble,
                axis=0,
                ddof=spread_ddof,
            )
        )

        # --------------------------------------------------------------
        # Land-only spread for statistics
        # --------------------------------------------------------------

        ensemble_spread_land = (
            ensemble_spread[
                land_mask
            ]
        )

        # --------------------------------------------------------------
        # Spatial statistics over land
        # --------------------------------------------------------------

        spread_percentile_01 = float(
            np.nanpercentile(
                ensemble_spread_land,
                1,
            )
        )

        spread_percentile_10 = float(
            np.nanpercentile(
                ensemble_spread_land,
                10,
            )
        )

        spread_spatial_mean = float(
            np.nanmean(
                ensemble_spread_land
            )
        )

        spread_percentile_90 = float(
            np.nanpercentile(
                ensemble_spread_land,
                90,
            )
        )

        spread_percentile_99 = float(
            np.nanpercentile(
                ensemble_spread_land,
                99,
            )
        )

        # --------------------------------------------------------------
        # Plot all cells
        # --------------------------------------------------------------

        figure_file = plot_spread(
            lon=lon,
            lat=lat,
            spread=ensemble_spread,
            cycle_time=cycle_time,
            variable_name=variable_name if layer is None else f"{variable_name}_layer{layer}",
            long_name=long_name,
            units=units,
            n_land=n_land,
            spread_spatial_mean=spread_spatial_mean,
            figure_dir=figure_dir,
        )

        # Additional maps for first-layer soil temperature only.
        # Per-cell extrema are taken across all ensemble members; map and
        # reported global extreme are restricted to land cells.
        if variable_name == "tslb" and layer == 1:
            tslb_ens_min = np.nanmin(ensemble, axis=0)
            tslb_ens_max = np.nanmax(ensemble, axis=0)
            plot_tslb_ensemble_extreme(
                lon=lon, lat=lat, values=tslb_ens_min,
                land_mask=land_mask, ensemble=ensemble, cycle_time=cycle_time,
                extreme="min", figure_dir=figure_dir,
            )
            plot_tslb_ensemble_extreme(
                lon=lon, lat=lat, values=tslb_ens_max,
                land_mask=land_mask, ensemble=ensemble, cycle_time=cycle_time,
                extreme="max", figure_dir=figure_dir,
            )
            del tslb_ens_min, tslb_ens_max

        # --------------------------------------------------------------
        # Output precision
        # --------------------------------------------------------------

        if variable_name == "q2":

            spread_fmt = ".6f"
            extrema_fmt = ".6f"

        else:

            spread_fmt = ".6f"
            extrema_fmt = ".6f"

        # --------------------------------------------------------------
        # Output
        # --------------------------------------------------------------

        lines.append(
            display_name
        )

        lines.append(
            f"  Variable: "
            f"{variable_name}"
        )

        lines.append(
            f"  Long name: "
            f"{long_name}"
        )

        lines.append(
            f"  Units: "
            f"{units}"
        )

        lines.append(
            f"  Spread percentile 1: "
            f"{spread_percentile_01:{spread_fmt}}"
        )

        lines.append(
            f"  Spread percentile 10: "
            f"{spread_percentile_10:{spread_fmt}}"
        )

        lines.append(
            f"  Spread spatial mean: "
            f"{spread_spatial_mean:{spread_fmt}}"
        )

        lines.append(
            f"  Spread percentile 90: "
            f"{spread_percentile_90:{spread_fmt}}"
        )

        lines.append(
            f"  Spread percentile 99: "
            f"{spread_percentile_99:{spread_fmt}}"
        )

        lines.append(
            f"  Minimum: "
            f"{overall_minimum:{extrema_fmt}}"
        )

        lines.append(
            f"  Maximum: "
            f"{overall_maximum:{extrema_fmt}}"
        )


        lines.append("")

        # --------------------------------------------------------------
        # Release large arrays
        # --------------------------------------------------------------

        del member_fields
        del ensemble
        del ensemble_spread
        del ensemble_spread_land

    # ------------------------------------------------------------------
    # Write output
    # ------------------------------------------------------------------

    lines.append(
        "=" * 80
    )

    output_file.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    return output_file

# ----------------------------------------------------------------------
# Dashboard JSON: read only the existing processed text reports.
# ----------------------------------------------------------------------

_REPORT_FIELDS = {
    "Spread percentile 1": "spread_percentile_1",
    "Spread percentile 10": "spread_percentile_10",
    "Spread spatial mean": "spread_spatial_mean",
    "Spread percentile 90": "spread_percentile_90",
    "Spread percentile 99": "spread_percentile_99",
    "Minimum": "minimum",
    "Maximum": "maximum",
}
_HEADER_FIELDS = {
    "Members": "members",
    "Files processed": "files_processed",
    "Spread STD ddof": "spread_ddof",
    "Total grid cells": "total_grid_cells",
    "Land grid cells": "land_grid_cells",
    "Ocean grid cells": "ocean_grid_cells",
}


def parse_cycle_report(report_file: Path, monitor_variables: list[dict]) -> dict:
    """Parse a completed data/YYYYMMDDHH.txt report (no MPAS I/O)."""
    text = report_file.read_text(encoding="utf-8")
    result = {"source": f"data/{report_file.name}", "variables": {}}
    expected = {entry["name"]: entry for entry in monitor_variables}
    current = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or set(line) == {"="}:
            continue
        if line in expected:
            current = line
            result["variables"][current] = {}
            continue
        if ":" not in line:
            continue
        key, value = (part.strip() for part in line.split(":", 1))
        if current is None:
            if key == "Cycle":
                result["cycle"] = value
            elif key in _HEADER_FIELDS:
                result[_HEADER_FIELDS[key]] = int(value)
        else:
            variable = result["variables"][current]
            if key == "Variable":
                variable["variable"] = value
            elif key == "Long name":
                variable["long_name"] = value
            elif key == "Units":
                variable["units"] = value
            elif key in _REPORT_FIELDS:
                number = float(value)
                if not math.isfinite(number):
                    raise ValueError(f"Non-finite {key} in {report_file}")
                variable[_REPORT_FIELDS[key]] = number

    if result.get("cycle") != report_file.stem:
        raise ValueError(f"Invalid cycle header in {report_file}")
    for field in _HEADER_FIELDS.values():
        if field not in result:
            raise ValueError(f"Missing {field} in {report_file}")
    if result["files_processed"] != result["members"]:
        raise ValueError(f"Incomplete member processing in {report_file}")
    for name, expected_config in expected.items():
        variable = result["variables"].get(name)
        if variable is None:
            raise ValueError(f"Missing variable {name} in {report_file}")
        if variable.get("variable") != expected_config["variable"]:
            raise ValueError(f"Unexpected variable for {name} in {report_file}")
        for required in ("long_name", "units", *_REPORT_FIELDS.values()):
            if required not in variable:
                raise ValueError(f"Missing {name}.{required} in {report_file}")
    return result


def write_dashboard_json(
    now: dt.datetime,
    config: dict,
    output_dir: Path,
    json_file: Path,
) -> Path:
    """Build dashboard JSON for the latest lookback_json hourly cycles.

    Missing or malformed reports are explicitly represented as unavailable.
    The MPAS ensemble output is never reread.
    """
    monitor_config = config["common"]["rrfs_monitor"]
    lookback_json = int(monitor_config.get("lookback_json", 4))
    if lookback_json < 1:
        raise ValueError("lookback_json must be >= 1")
    threshold = monitor_config.get("soil_temperature_alert", {})
    low = float(threshold.get("min_k", 253.0))
    high = float(threshold.get("max_k", 323.0))
    if not math.isfinite(low) or not math.isfinite(high) or low >= high:
        raise ValueError("Invalid soil_temperature_alert thresholds")
    current_cycle = now.replace(minute=0, second=0, microsecond=0)
    cycles = []
    alerts = []
    variables = monitor_config["monitor_variables"]
    soil_names = [v["name"] for v in variables if v["variable"] == "tslb" and int(v.get("layer", 1)) == 1]

    for offset in range(lookback_json):
        cycle_time = current_cycle - dt.timedelta(hours=offset)
        cycle = cycle_time.strftime("%Y%m%d%H")
        report_file = output_dir / f"{cycle}.txt"
        if not report_file.is_file():
            cycles.append({"cycle": cycle, "status": "unavailable", "reason": "report_missing", "variables": {}})
            continue
        try:
            item = parse_cycle_report(report_file, variables)
        except (OSError, ValueError) as exc:
            cycles.append({"cycle": cycle, "status": "unavailable", "reason": "report_invalid", "detail": str(exc), "variables": {}})
            continue
        item["status"] = "available"
        item["alerts"] = []
        for name in soil_names:
            soil = item["variables"][name]
            for stat, comparison, limit in (("minimum", "below", low), ("maximum", "above", high)):
                value = soil[stat]
                if (comparison == "below" and value < limit) or (comparison == "above" and value > limit):
                    alert = {
                        "cycle": cycle,
                        "variable": name,
                        "field": stat,
                        "value": value,
                        "threshold": limit,
                        "units": "K",
                        "message": f"{name} {stat} {value:.6f} K {comparison} {limit:g} K",
                    }
                    item["alerts"].append(alert)
                    alerts.append(alert)
        cycles.append(item)

    payload = {
        "title": "RRFSv2X Ensemble Variable Monitor",
        "generated_at_utc": now.astimezone(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
        "lookback_json": lookback_json,
        "soil_temperature_alert": {"min_k": low, "max_k": high, "units": "K"},
        "alerts": alerts,
        "cycles": cycles,
    }
    json_file.parent.mkdir(parents=True, exist_ok=True)
    # Atomic replacement prevents the dashboard from seeing half-written JSON.
    temp_file = json_file.with_name(json_file.name + ".tmp")
    try:
        temp_file.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        temp_file.replace(json_file)
    finally:
        temp_file.unlink(missing_ok=True)
    return json_file
