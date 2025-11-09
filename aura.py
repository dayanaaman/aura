# full_pipeline_cams_cities.py
import os
import cdsapi
import zipfile
import shutil
import traceback
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.lines import Line2D
from matplotlib import colors, cm

# -----------------------
# Настройки запроса CAMS (не меняем, если уже скачано)
# -----------------------
dataset = "cams-europe-air-quality-forecasts"
request = {
    "variable": ["alder_pollen","birch_pollen","grass_pollen","mugwort_pollen","olive_pollen","ragweed_pollen","pm2p5"],
    "model": ["ensemble"],
    "level": ["0"],
    "date": "2025-11-05/2025-11-08",
    "type": ["forecast"],
    "time": ["00:00"],
    "leadtime_hour": ["0"],
    "data_format": "netcdf_zip",
    "area": [72, -25, 34, 45]
}

zip_file = "cams_data.zip"
extract_dir = "cams_data"
nc_name = "ENS_FORECAST.nc"

# Безопасная локальная папка для работы (не OneDrive)
safe_dir = r"C:\temp\cams_test"
os.makedirs(safe_dir, exist_ok=True)
safe_path = os.path.join(safe_dir, nc_name)

# -----------------------
# 1) Скачать архив если нужно
# -----------------------
if not os.path.exists(zip_file) or os.path.getsize(zip_file) == 0:
    print("Скачиваем данные CAMS в", os.path.abspath(zip_file))
    client = cdsapi.Client()  # предполагается ~/.cdsapirc или env-config
    client.retrieve(dataset, request, zip_file)
else:
    print("Архив уже есть:", os.path.abspath(zip_file), "size:", os.path.getsize(zip_file))

# -----------------------
# 2) Проверка содержимого архива и распаковка
# -----------------------
if not os.path.exists(zip_file):
    raise FileNotFoundError("Архив не найден: " + zip_file)

with zipfile.ZipFile(zip_file, 'r') as zf:
    members = zf.namelist()
    print("Архив содержит:", members[:50])
    nc_members = [m for m in members if m.endswith('.nc')]
    if not nc_members:
        raise FileNotFoundError("В архиве нет .nc файлов.")
    # Распакуем в extract_dir
    os.makedirs(extract_dir, exist_ok=True)
    zf.extractall(extract_dir)
    print("Распаковано в", os.path.abspath(extract_dir))

# Найдём файл в распакованной папке
found = []
for root, dirs, files in os.walk(extract_dir):
    for f in files:
        if f.endswith('.nc'):
            found.append(os.path.join(root, f))
if not found:
    raise FileNotFoundError("Не найден .nc после распаковки")
nc_path = found[0]
print("Найден .nc:", nc_path)

# -----------------------
# 3) Копируем в безопасную папку (обход OneDrive)
# -----------------------
try:
    print("Копирую .nc в безопасную папку:", safe_path)
    shutil.copy2(nc_path, safe_path)
    print("Копирование OK, size:", os.path.getsize(safe_path))
except Exception as e:
    print("Ошибка копирования, пробуем извлечь напрямую в safe_dir:", e)
    try:
        with zipfile.ZipFile(zip_file, 'r') as zf:
            member = nc_members[0]
            dst = os.path.join(safe_dir, os.path.basename(member))
            with zf.open(member) as src, open(dst, 'wb') as out:
                shutil.copyfileobj(src, out)
        safe_path = dst
        print("Извлёк напрямую:", safe_path, "size:", os.path.getsize(safe_path))
    except Exception as e2:
        print("Не удалось извлечь в safe_dir:", e2)
        raise

# -----------------------
# 4) Открываем dataset из safe_path
# -----------------------
print("Открываю dataset из:", safe_path)
ds = xr.open_dataset(safe_path, engine='netcdf4')
print("Открыт. Переменные:", list(ds.variables))

# -----------------------
# 5) Определение реальных имен пыльцы и сбор данных
# -----------------------
# Возможные реальные имена, поддерживаем вариант apg_conc/etc. и альтернативы
possible_pollen_sets = [
    # apg_conc etc. (видно в твоём файле)
    ['apg_conc', 'bpg_conc', 'gpg_conc', 'mpg_conc', 'opg_conc', 'rwpg_conc'],
    # альтернативные имена, если бы они были
    ['alder_pollen','birch_pollen','grass_pollen','mugwort_pollen','olive_pollen','ragweed_pollen']
]

pollen_vars = None
for cand in possible_pollen_sets:
    if any(v in ds.variables for v in cand):
        pollen_vars = [v for v in cand if v in ds.variables]
        break
# если не найдено, попробуем угадать по подстроке 'pollen' или '_conc'
if pollen_vars is None:
    pollen_vars = [v for v in ds.variables if ('pollen' in v or v.endswith('_conc'))][:6]

if not pollen_vars:
    raise RuntimeError("Не удалось обнаружить переменные пыльцы в dataset. Проверь список ds.variables")

print("Используем pollen vars:", pollen_vars)

# Селектор (time=0, level=0 если есть)
sel = {}
if 'time' in ds.dims or 'time' in ds.coords:
    sel['time'] = 0
if 'level' in ds.dims or 'level' in ds.coords:
    sel['level'] = 0
print("Selector:", sel)

# Шаблон для нулей
if pollen_vars:
    template = ds[pollen_vars[0]].isel(**sel)
else:
    lat_n = next((n for n in ('latitude','lat') if n in ds.coords or n in ds.variables), None)
    lon_n = next((n for n in ('longitude','lon') if n in ds.coords or n in ds.variables), None)
    if lat_n and lon_n:
        latv = ds[lat_n].values
        lonv = ds[lon_n].values
        template = xr.DataArray(np.zeros((len(latv),len(lonv))),
                                coords={lat_n: latv, lon_n: lonv},
                                dims=(lat_n, lon_n))
    else:
        raise RuntimeError("Не удалось создать шаблон для массивов")

# Суммируем пыльцу, отсутствующие переменные подставляем нулями
total_pollen = None
for v in pollen_vars:
    if v in ds.variables:
        arr = ds[v].isel(**sel)
    else:
        arr = xr.zeros_like(template)
    total_pollen = arr if total_pollen is None else (total_pollen + arr)

# PM2.5 обработка
if 'pm2p5' in ds.variables:
    pm2p5 = ds['pm2p5'].isel(**sel)
    maxv = float(pm2p5.max()) if float(pm2p5.max()) != 0 else 0.0
    pm2p5_norm = (pm2p5 / maxv) if maxv > 0 else xr.zeros_like(pm2p5)
else:
    pm2p5_norm = xr.zeros_like(total_pollen)

allergy_index = total_pollen + pm2p5_norm * 10.0
print("Allergy index stats:", float(allergy_index.min()), float(allergy_index.max()))

# -----------------------
# 6) Города и классификация
# -----------------------
cities = {
    "Milan": (45.4642, 9.19),
    "Paris": (48.8566, 2.3522),
    "Venice": (45.4408, 12.3155)
}

def classify_score(v):
    if np.isnan(v):
        return "unknown"
    if v <= 3:
        return "safe"
    if v <= 6:
        return "moderate"
    return "high"

colors_map = {"safe":"green", "moderate":"orange", "high":"red"}
marker_sizes = {"safe": 60, "moderate": 90, "high": 130}

# координаты
lat_name = next((n for n in ('latitude','lat','y') if n in ds.coords or n in ds.variables), None)
lon_name = next((n for n in ('longitude','lon','x') if n in ds.coords or n in ds.variables), None)
if lat_name is None or lon_name is None:
    raise KeyError("Не найдены координаты lat/lon в dataset")
lats = ds[lat_name].values
lons = ds[lon_name].values

def nearest_grid_value(array2d, lats, lons, target_lat, target_lon):
    lat_arr = np.asarray(lats)
    lon_arr = np.asarray(lons)
    if lat_arr.ndim == 2:
        lat_vals = lat_arr[:,0]
    else:
        lat_vals = lat_arr
    if lon_arr.ndim == 2:
        lon_vals = lon_arr[0,:]
    else:
        lon_vals = lon_arr
    i = np.abs(lat_vals - target_lat).argmin()
    j = np.abs(lon_vals - target_lon).argmin()
    return float(array2d.values[i, j]), i, j

city_results = {}
for name, (plat, plon) in cities.items():
    try:
        val, i, j = nearest_grid_value(allergy_index, lats, lons, plat, plon)
        cat = classify_score(val)
        city_results[name] = {"value": val, "category": cat, "lat_idx": i, "lon_idx": j, "lat": plat, "lon": plon}
    except Exception as e:
        city_results[name] = {"value": np.nan, "category": "unknown", "error": str(e)}

print("City results:")
for k,v in city_results.items():
    print(f" - {k}: {v}")

# -----------------------
# 7) Визуализация: фон + точки городов
# -----------------------
out_png = "europe_allergy_cities.png"
plt.figure(figsize=(11,9))
ax = plt.axes(projection=ccrs.PlateCarree())

# Ограничим вид на Европу
ax.set_extent([-12, 30, 35, 60], crs=ccrs.PlateCarree())

# Попытка фоновой карты allergy_index (полупрозрачная)
try:
    # используем pcolormesh для контроля
    data = allergy_index.values
    # ensure 2D orientation (lat, lon)
    if data.ndim == 2:
        Lon, Lat = np.meshgrid(lons, lats)
        mesh = ax.pcolormesh(Lon, Lat, data, transform=ccrs.PlateCarree(), cmap='OrRd', alpha=0.5)
        cb = plt.colorbar(mesh, ax=ax, orientation='vertical', fraction=0.04, pad=0.02)
        cb.set_label('Allergy Index')
    else:
        print("Unexpected dims for allergy_index:", data.shape)
except Exception as e:
    print("Не удалось отобразить фон allergy_index:", e)
    traceback.print_exc()

# Рисуем маркеры городов
for name, info in city_results.items():
    cat = info["category"]
    val = info["value"]
    plat = info["lat"]
    plon = info["lon"]
    if cat == "unknown":
        continue
    col = colors_map[cat]
    size = marker_sizes[cat]
    ax.scatter(plon, plat, s=size, c=col, edgecolor='k', transform=ccrs.PlateCarree(), zorder=6)
    ax.text(plon + 0.3, plat + 0.1, f"{name}\n{cat} ({val:.1f})", transform=ccrs.PlateCarree(),
            fontsize=9, weight='bold', zorder=7,
            bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2'))

# Оформление
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS, linestyle=':')
ax.add_feature(cfeature.LAND, zorder=0, edgecolor='black')

legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='safe (0-3)', markerfacecolor=colors_map['safe'], markersize=10, markeredgecolor='k'),
    Line2D([0], [0], marker='o', color='w', label='moderate (4-6)', markerfacecolor=colors_map['moderate'], markersize=12, markeredgecolor='k'),
    Line2D([0], [0], marker='o', color='w', label='high (7-10)', markerfacecolor=colors_map['high'], markersize=14, markeredgecolor='k'),
]
ax.legend(handles=legend_elements, loc='lower left')

ax.set_title("Allergy Risk — cities (safe/moderate/high)\nCities: Milan, Paris, Venice")
plt.savefig(out_png, dpi=200, bbox_inches='tight')
print("Saved:", out_png)
plt.show()

ax.set_title("Allergy Risk — cities (safe/moderate/high)\nCities: Milan, Paris, Venice")
plt.savefig(out_png, dpi=200, bbox_inches='tight')
print("Saved:", out_png)
plt.show()

# -----------------------
# 8) Генерация интерактивного HTML с Leaflet
# -----------------------
out_html = "allergy_risk_map_europe.html"

# Подготовим данные городов в JS-формате
cities_list = []
for name, info in city_results.items():
    lat = float(info.get("lat", np.nan))
    lon = float(info.get("lon", np.nan))
    val = float(info.get("value", np.nan)) if info.get("value") is not None else None
    cat = info.get("category", "unknown")
    cities_list.append({"name": name, "lat": lat, "lon": lon, "value": val, "category": cat})

# Цвета для категорий
import json
color_map = {"safe": "green", "moderate": "orange", "high": "red", "unknown": "gray"}
cities_json = json.dumps(cities_list)

html_template = f"""
<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8"/>
  <title>Allergy Risk — Europe (cities)</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0">

  <!-- Leaflet CSS/JS -->
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" crossorigin=""/>
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js" crossorigin=""></script>

  <style>
    body {{ margin:0; font-family: Arial, Helvetica, sans-serif; }}
    #map {{ height: 70vh; width: 100%; }}
    .legend {{ background:white; padding:8px; border-radius:6px; box-shadow:0 0 8px rgba(0,0,0,0.2); }}
    .city-label {{ font-weight:bold; }}
    .info-box {{ padding:10px; max-width:900px; margin:10px auto; }}
    img.backup {{ max-width:100%; height:auto; display:block; margin:10px auto; border:1px solid #ddd; }}
  </style>
</head>
<body>
  <div class="info-box">
    <h2>Allergy Risk — cities (safe / moderate / high)</h2>
    <p>Источник: CAMS. Отображены города: Milan, Paris, Venice. Маркеры окрашены по уровню риска.</p>
  </div>

  <div id="map"></div>

  <div class="info-box">
    <h3>Сводка по городам</h3>
    <ul id="city-list"></ul>
    <p>Статическая карта (бэкап):</p>
    <img class="backup" src="europe_allergy_cities.png" alt="Allergy map (static)"/>
  </div>

  <script>
    const cities = {cities_json};
    const map = L.map('map').setView([47.0, 8.0], 5);
    L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
      maxZoom: 19,
      attribution: '&copy; OpenStreetMap contributors'
    }}).addTo(map);
    const colorMap = {json.dumps(color_map)};
    cities.forEach(c => {{
      if (!isFinite(c.lat) || !isFinite(c.lon)) return;
      const cat = c.category || 'unknown';
      const col = colorMap[cat] || 'gray';
      const radius = (cat === 'high') ? 12 : (cat === 'moderate') ? 9 : (cat === 'safe') ? 7 : 6;
      const circle = L.circleMarker([c.lat, c.lon], {{
        color: '#000', weight: 1, fillColor: col, fillOpacity: 0.9, radius: radius
      }}).addTo(map);
      const popupHtml = `<div class="city-label">${{c.name}}</div>
                         <div>Score: ${{(c.value !== null && !isNaN(c.value)) ? c.value.toFixed(1) : 'N/A'}}</div>
                         <div>Category: ${{c.category}}</div>`;
      circle.bindPopup(popupHtml);
      const li = document.createElement('li');
      li.innerHTML = `<strong>${{c.name}}</strong>: ${{c.category}} (${{(c.value !== null && !isNaN(c.value)) ? c.value.toFixed(1) : 'N/A'}})`;
      document.getElementById('city-list').appendChild(li);
    }});
    const legend = L.control({{position: 'bottomright'}});
    legend.onAdd = function(map) {{
      const div = L.DomUtil.create('div', 'legend');
      div.innerHTML += '<div style="margin-bottom:6px"><strong>Risk levels</strong></div>';
      for (const [k,v] of Object.entries(colorMap)) {{
        div.innerHTML += `<div style="display:flex;align-items:center;margin:4px 0;">
                            <span style="display:inline-block;width:18px;height:18px;background:${{v}};border:1px solid #000;margin-right:8px;"></span>
                            <span style="text-transform:capitalize;">${{k}}</span>
                          </div>`;
      }}
      return div;
    }};
    legend.addTo(map);
  </script>
</body>
</html>
"""

with open(out_html, "w", encoding="utf-8") as fh:
    fh.write(html_template)

print("HTML saved:", out_html)
print("Открой в браузере: file:///" + os.path.abspath(out_html))