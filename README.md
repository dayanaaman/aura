# 🌍 Allergy Risk Map (CAMS Pollen + PM2.5)

**Allergy Risk Map** is a Python-based project that downloads real atmospheric data from **CAMS (Copernicus Atmosphere Monitoring Service)**, calculates allergy risk levels for selected European cities, and generates both a **static map (PNG)** and an **interactive map (HTML)** using real forecast data.

---

## 🚀 Features

- ✅ Automatically downloads **CAMS Europe Air Quality Forecasts** (pollen + PM2.5)
- ✅ Calculates a combined **allergy risk index**
- ✅ Classifies risk as:
  - 🟢 **Safe (0–3)**
  - 🟠 **Moderate (4–6)**
  - 🔴 **High (7–10)**
- ✅ Generates:
  - `europe_allergy_cities.png` — static visualization with city markers
  - `allergy_risk_map_europe.html` — interactive Leaflet map
- ✅ Works fully offline after first data download
- ✅ Built with `xarray`, `matplotlib`, `cartopy`, `leaflet.js`, and `cdsapi`

---

## 🧠 Data Source

The data comes from **Copernicus Atmosphere Monitoring Service (CAMS)**:
> [https://atmosphere.copernicus.eu/](https://atmosphere.copernicus.eu/)

The script uses:
- CAMS dataset: `cams-europe-air-quality-forecasts`
- Variables: alder, birch, grass, mugwort, olive, ragweed pollen + PM2.5

---

## 🧩 Project Structure

