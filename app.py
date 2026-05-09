from flask import Flask, render_template, jsonify, request
import json
import os
import math
import geopandas as gpd
import networkx as nx
from shapely.geometry import LineString, MultiLineString
from dotenv import load_dotenv
from openai import OpenAI

app = Flask(__name__)

load_dotenv(override=True)

api_key = os.getenv("OPENAI_API_KEY")
print("LOADED KEY:", repr(api_key[:15]) if api_key else "None")

client = OpenAI(api_key=api_key)

DATA_FOLDER = "data"

NETWORK_FILES = {
    "safe": "roads_safe.geojson",
    "emg": "roads_emg.geojson",
    "last": "roads_last.geojson"
}

MODE_LABELS = {
    "safe": "🚗 Normal Car",
    "emg": "🚑 Rescue Vehicle",
    "last": "🚛 High-Clearance Truck"
}

RISK_ORDER = {
    "none": 0,
    "low": 1,
    "medium": 2,
    "high": 3,
    "boat": 4
}

RISK_LABELS = {
    "none": "Dry / no flood",
    "low": "Low flood",
    "medium": "Medium flood",
    "high": "High flood",
    "boat": "Boat-level flooding"
}

graphs = {}


def load_geojson(filename):
    path = os.path.join(DATA_FOLDER, filename)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def point_key(x, y, precision=6):
    return (round(float(x), precision), round(float(y), precision))


def normalize_flood_class(value):
    if value is None:
        return "none"
    v = str(value).strip().lower()
    return v if v in RISK_ORDER else "none"


def haversine_meters(a, b):
    lon1, lat1 = map(math.radians, [a[0], a[1]])
    lon2, lat2 = map(math.radians, [b[0], b[1]])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    h = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    )
    earth_radius_m = 6371000
    return 2 * earth_radius_m * math.asin(math.sqrt(h))


def add_segment_to_graph(G, a, b, seg_len, flood_class):
    if seg_len <= 0:
        return

    flood_class = normalize_flood_class(flood_class)

    if G.has_edge(a, b):
        existing = G[a][b]
        existing["weight"] = min(existing.get("weight", seg_len), seg_len)

        existing_class = normalize_flood_class(existing.get("flood3", "none"))
        if RISK_ORDER[flood_class] > RISK_ORDER[existing_class]:
            existing["flood3"] = flood_class
    else:
        G.add_edge(a, b, weight=seg_len, flood3=flood_class)


def add_linestring_to_graph(G, line, flood_class):
    coords = list(line.coords)
    for i in range(len(coords) - 1):
        a = point_key(coords[i][0], coords[i][1])
        b = point_key(coords[i + 1][0], coords[i + 1][1])
        seg_len = haversine_meters(a, b)
        add_segment_to_graph(G, a, b, seg_len, flood_class)


def build_graph(mode):
    filename = NETWORK_FILES[mode]
    path = os.path.join(DATA_FOLDER, filename)

    gdf = gpd.read_file(path)

    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=4326, allow_override=True)
    else:
        gdf = gdf.to_crs(epsg=4326)

    G = nx.Graph()

    for _, row in gdf.iterrows():
        geom = row.geometry
        if geom is None:
            continue

        flood_class = normalize_flood_class(row.get("flood3", "none"))

        if isinstance(geom, LineString):
            add_linestring_to_graph(G, geom, flood_class)

        elif isinstance(geom, MultiLineString):
            for part in geom.geoms:
                add_linestring_to_graph(G, part, flood_class)

    graphs[mode] = G


def ensure_graph_loaded(mode):
    if mode not in graphs:
        build_graph(mode)


def nearest_node(G, lon, lat):
    target = (lon, lat)
    best_node = None
    best_dist = float("inf")

    for node in G.nodes:
        d = haversine_meters(node, target)
        if d < best_dist:
            best_dist = d
            best_node = node

    return best_node


def summarize_path(G, path_nodes, mode):
    exposure_m = {
        "none": 0.0,
        "low": 0.0,
        "medium": 0.0,
        "high": 0.0,
        "boat": 0.0
    }

    max_risk = "none"
    total_m = 0.0

    for u, v in zip(path_nodes[:-1], path_nodes[1:]):
        edge_data = G.get_edge_data(u, v) or {}
        seg_len = float(edge_data.get("weight", 0.0))
        flood_class = normalize_flood_class(edge_data.get("flood3", "none"))

        exposure_m[flood_class] += seg_len
        total_m += seg_len

        if RISK_ORDER[flood_class] > RISK_ORDER[max_risk]:
            max_risk = flood_class

    exposure_km = {
        key: round(val / 1000, 2)
        for key, val in exposure_m.items()
    }

    return {
        "mode": mode,
        "mode_label": MODE_LABELS[mode],
        "length_m": round(total_m, 1),
        "length_km": round(total_m / 1000, 2),
        "max_risk": max_risk,
        "max_risk_label": RISK_LABELS[max_risk],
        "exposure_km": exposure_km,
        "status": "Route found"
    }


def solve_route_for_mode(mode, origin, destination):
    ensure_graph_loaded(mode)
    G = graphs[mode]

    start_node = nearest_node(G, origin["lng"], origin["lat"])
    end_node = nearest_node(G, destination["lng"], destination["lat"])

    if start_node is None or end_node is None:
        return {
            "available": False,
            "mode": mode,
            "mode_label": MODE_LABELS[mode],
            "error": "Could not find nearby road nodes."
        }

    try:
        path_nodes = nx.shortest_path(G, source=start_node, target=end_node, weight="weight")
    except nx.NetworkXNoPath:
        return {
            "available": False,
            "mode": mode,
            "mode_label": MODE_LABELS[mode],
            "error": "No route"
        }

    summary = summarize_path(G, path_nodes, mode)
    summary["available"] = True
    summary["coordinates"] = [[node[0], node[1]] for node in path_nodes]
    return summary


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/flood")
def flood():
    filename = "flood_display.geojson"
    path = os.path.join(DATA_FOLDER, filename)

    if not os.path.exists(path):
        return jsonify({"error": "Missing file: flood_display.geojson"}), 404

    return jsonify(load_geojson(filename))


@app.route("/boat")
def boat():
    filename = "boat_display.geojson"
    path = os.path.join(DATA_FOLDER, filename)

    if not os.path.exists(path):
        return jsonify({"error": "Missing file: boat_display.geojson"}), 404

    return jsonify(load_geojson(filename))


@app.route("/route", methods=["POST"])
def route():
    data = request.get_json()

    mode = data.get("mode")
    origin = data.get("origin")
    destination = data.get("destination")

    if mode not in NETWORK_FILES:
        return jsonify({"error": "Invalid mode"}), 400

    if not origin or not destination:
        return jsonify({"error": "Origin and destination are required."}), 400

    result = solve_route_for_mode(mode, origin, destination)

    if not result["available"]:
        return jsonify({"error": result["error"]}), 404

    return jsonify({
        "type": "Feature",
        "geometry": {
            "type": "LineString",
            "coordinates": result["coordinates"]
        },
        "properties": {
            "mode": result["mode"],
            "mode_label": result["mode_label"],
            "length_m": result["length_m"],
            "length_km": result["length_km"],
            "max_risk": result["max_risk"],
            "max_risk_label": result["max_risk_label"],
            "exposure_km": result["exposure_km"],
            "status": result["status"]
        }
    })


@app.route("/assistant", methods=["POST"])
def assistant():
    data = request.get_json()
    message = (data.get("message") or "").strip()

    if not message:
        return jsonify({"reply": "Please type your question."})

    system_prompt = """
You are Flood Help Assistant for a flood-aware routing website in Hat Yai, Thailand.

Help users with:
- flood safety
- evacuation advice
- emergency guidance
- route mode explanation
- boat access meaning
- emergency contact numbers

Use simple, calm, practical language.
If the user writes in Thai, reply in Thai.
If the situation sounds dangerous, advise the user to contact emergency services immediately.

Emergency contacts in Thailand:
- General Emergency: 191
- Medical Emergency: 1669
- Fire and Rescue: 199
- Disaster Hotline: 1784
"""

    try:
        response = client.responses.create(
            model="gpt-5-mini",
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": message}
            ]
        )

        return jsonify({"reply": response.output_text})

    except Exception as e:
        print("OPENAI ERROR:", e)
        return jsonify({"reply": f"Assistant error: {str(e)}"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)