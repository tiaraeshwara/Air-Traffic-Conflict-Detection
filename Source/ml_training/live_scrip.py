import pandas as pd
import numpy as np
import requests
import joblib

from datetime import datetime
from itertools import combinations
from geographiclib.geodesic import Geodesic


#---Configs-----
USERNAME = "ajan eshwara"
PASSWORD = "Eshwara@1995"
BBOX = (47.0, 56.0, 5.0, 15.0)  # min_lat, max_lat, min_lon, max_lon
MODEL_PATH = "../../Model/random_forest_si_model.joblib"

# --Load trained model---
clf = joblib.load(MODEL_PATH)

#------Get live ADS-B data from OpenSky-----
def fetch_opensky_live(bbox):
    url = "https://opensky-network.org/api/states/all"
    params = {"bbox": ",".join(map(str, BBOX))}
    r = requests.get(url, auth=(USERNAME, PASSWORD), params=params)
    r.raise_for_status()
    data = r.json()

    columns = [
        "icao24", "callsign", "origin_country", "time_position",
        "last_contact", "longitude", "latitude", "baro_altitude",
        "on_ground", "velocity", "true_track", "vertical_rate",
        "sensors", "geo_altitude", "squawk", "spi", "position_source"
    ]
    if data["states"] is None:
        return pd.DataFrame(columns=columns)

    df = pd.DataFrame(data["states"], columns=columns)

    # Keep only airborne aircraft with complete position data
    df = df.dropna(subset=["longitude", "latitude", "geo_altitude", "velocity", "true_track","vertical_rate"])
    return df

# ==== 3. Create aircraft pairs + same features used in training ====
def generate_features_from_live(df):
    rows = []
    for (_, ac1), (_, ac2) in combinations(df.iterrows(), 2):
        # Horizontal separation (NM)
        g = Geodesic.WGS84.Inverse(ac1.latitude, ac1.longitude,
                                   ac2.latitude, ac2.longitude)
        horiz_sep_m = g['s12']
        horiz_sep_nm = horiz_sep_m / 1852
        # Vertical separation (ft)
        vert_sep_ft = abs(ac1.geo_altitude - ac2.geo_altitude) * 3.28084
        #heading differance
        heading_track = abs(ac1.true_track - ac2.true_track)
        # Relative velocity (m/s diff)
        delta_v = abs(ac1.velocity - ac2.velocity)
        # Vertical rate difference
        verrate_diff = abs(ac1.vertical_rate - ac2.vertical_rate)
        #bearing A to B
        bearing = g['azi1']
        #track_course_diff
        track_diff = abs(ac1.track_course - bearing)

        # Match your training feature names/order exactly!
        rows.append({
            "horizontal_sep_m":horiz_sep_m,
            "horizontal_sep_NM": horiz_sep_nm,
            "vertical_sep_ft": vert_sep_ft,
            "heading_diff_deg":heading_track,
            "velocity_diff": delta_v,
            "vertical_rate_diff":verrate_diff,
            "bearing_A_to_B":bearing,
            "track_course_diff": track_diff
            # add any other features you used in training
        })
    return pd.DataFrame(rows)

# ==== 4. Run prediction ====
live_df = fetch_opensky_live(BBOX)
if live_df.empty:
    print("No live aircraft in this area at the moment.")
else:
    feat_df = generate_features_from_live(live_df)

    if not feat_df.empty:
        preds = clf.predict(feat_df)
        probs = clf.predict_proba(feat_df)[:, 1]
        feat_df["SI_prediction"] = preds
        feat_df["SI_probability"] = probs

        print(feat_df[feat_df["SI_prediction"] == 1]
              .sort_values("SI_probability", ascending=False))
    else:
        print("No pairs generated from live data.")
