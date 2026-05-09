# Hat Yai Flood Route App

This is a starter web prototype for your flood-aware routing project.

## Folder structure

- app.py
- data/
  - roads_safe.geojson
  - roads_emg.geojson
  - roads_last.geojson
- templates/
  - index.html
- static/
  - style.css

## What to do next

1. Export your QGIS layers as:
   - roads_safe.geojson
   - roads_emg.geojson
   - roads_last.geojson
2. Put them inside the `data` folder.
3. Install Flask:
   pip install flask
4. Run:
   python app.py
5. Open:
   http://127.0.0.1:5000
