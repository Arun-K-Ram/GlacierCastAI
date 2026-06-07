"""
Download and plot ERA5 climate data for Aletsch glacier region.
Generates climate trend figures for the paper.
"""

import cdsapi
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

BBOX    = [7.8, 46.3, 8.2, 46.7]
OUT_DIR = Path("data/raw/climate/aletsch")
FIG_DIR = Path("paper/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

client = cdsapi.Client()

#  Download ERA5 monthly means 2000-2023 
out_file = OUT_DIR / "era5_2000_2023.nc"

if not out_file.exists():
    print("Downloading ERA5 data...")
    client.retrieve(
        "reanalysis-era5-single-levels-monthly-means",
        {
            "product_type": "monthly_averaged_reanalysis",
            "variable": [
                "2m_temperature",
                "total_precipitation",
                "snowfall",
            ],
            "year":  [str(y) for y in range(2000, 2024)],
            "month": ["06", "07", "08", "09"],
            "time":  "00:00",
            "area":  [46.7, 7.8, 46.3, 8.2],
            "data_format": "netcdf",
            "download_format": "unarchived",
        },
        str(out_file),
    )
    print(f"Saved: {out_file}")
else:
    print(f"Already exists: {out_file}")

#  Load and process 
print("Loading ERA5 data...")

nc_files = [f for f in OUT_DIR.glob("*.nc") if "stream" in f.name]
print(f"Found NetCDF files: {[f.name for f in nc_files]}")

ds = xr.open_mfdataset(nc_files, engine="netcdf4", combine="by_coords")
print(ds)

# Spatial mean over the glacier region
t2m    = ds["t2m"].mean(dim=["latitude", "longitude"]) - 273.15
precip = ds["tp"].mean(dim=["latitude", "longitude"]) * 1000
snow   = ds["sf"].mean(dim=["latitude", "longitude"]) * 1000

# Annual summer mean (JJA+Sep)
years      = sorted(set(t2m.valid_time.dt.year.values))
temp_ann   = [float(t2m.sel(valid_time=t2m.valid_time.dt.year == y).mean()) for y in years]
precip_ann = [float(precip.sel(valid_time=precip.valid_time.dt.year == y).mean()) for y in years]
snow_ann   = [float(snow.sel(valid_time=snow.valid_time.dt.year == y).mean()) for y in years]

#  Plot 1: Summer temperature trend 
fig, ax = plt.subplots(figsize=(14, 5))

ax.plot(years, temp_ann, "o-", color="#E53935", linewidth=2.5,
        markersize=7, markerfacecolor="white", markeredgewidth=2)
ax.fill_between(years, temp_ann, min(temp_ann), alpha=0.1, color="#E53935")

# Trend line
z    = np.polyfit(years, temp_ann, 1)
p    = np.poly1d(z)
ax.plot(years, p(years), "--", color="#B71C1C", linewidth=1.5, alpha=0.7,
        label=f"Trend: {z[0]:+.3f}°C/yr")

ax.set_xlabel("Year", fontsize=13)
ax.set_ylabel("Mean Summer Temperature (°C)", fontsize=13)
ax.set_title("Aletsch Region - Summer Temperature Trend (ERA5)\nJune–September Mean",
             fontsize=14, fontweight="bold")
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(FIG_DIR / "aletsch_temperature_trend.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: aletsch_temperature_trend.png")

#  Plot 2: Snowfall trend 
fig, ax = plt.subplots(figsize=(14, 5))

ax.bar(years, snow_ann, color="#1565C0", alpha=0.7, label="Summer snowfall")

z2 = np.polyfit(years, snow_ann, 1)
p2 = np.poly1d(z2)
ax.plot(years, p2(years), "--", color="#0D47A1", linewidth=2,
        label=f"Trend: {z2[0]:+.4f} mm/yr")

ax.set_xlabel("Year", fontsize=13)
ax.set_ylabel("Mean Summer Snowfall (mm)", fontsize=13)
ax.set_title("Aletsch Region - Summer Snowfall Trend (ERA5)\nJune–September Mean",
             fontsize=14, fontweight="bold")
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig(FIG_DIR / "aletsch_snowfall_trend.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: aletsch_snowfall_trend.png")

#  Plot 3: Temperature vs glacier area 
# Glacier areas from plot_retreat.py - hardcode what we got
glacier_years = [2002,2003,2004,2005,2008,2009,2010,2011,2015,2016,2017,2022,2023]
glacier_areas = [309.7,130.7,86.9,289.9,251.8,73.7,187.1,72.4,139.3,92.4,143.4,104.4,6.3]

# Match years
common_years  = [y for y in glacier_years if y in years]
common_temps  = [temp_ann[years.index(y)] for y in common_years]
common_areas  = [glacier_areas[glacier_years.index(y)] for y in common_years]

fig, ax1 = plt.subplots(figsize=(14, 6))
ax2 = ax1.twinx()

ax1.plot(common_years, common_temps, "o-", color="#E53935", linewidth=2.5,
         markersize=7, markerfacecolor="white", markeredgewidth=2, label="Temperature (°C)")
ax2.plot(common_years, common_areas, "s--", color="#2196F3", linewidth=2.5,
         markersize=7, markerfacecolor="white", markeredgewidth=2, label="Glacier Area (km²)")

ax1.set_xlabel("Year", fontsize=13)
ax1.set_ylabel("Summer Temperature (°C)", color="#E53935", fontsize=13)
ax2.set_ylabel("Glacier Area (km²)", color="#2196F3", fontsize=13)
ax1.set_title("Aletsch Glacier - Temperature vs Area\n(ERA5 + Landsat NDSI)",
              fontsize=14, fontweight="bold")

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=12)
ax1.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(FIG_DIR / "aletsch_temp_vs_area.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: aletsch_temp_vs_area.png")