import pandas as pd
import numpy as np
import os
from utilities import *

files_loc = "C:/Users/its/Documents/racing/data_pkl"
all_files = os.listdir(files_loc)

#################################################################################
# EXECUTE THIS SECTION ONLY THE FIRST TIME: IT CREATES AN EMPTY DATA FRAME

#df = pd.DataFrame(columns = ['subject', 'level', 'track', 'session', 'block', 'lap', 'turn', 'quantity', 'value', 'detect', 'midline_value'])
#df.to_pickle('/select/your/directory/df_allpoints.pkl')
#################################################################################

fs = 240 # sampling rate
g = 9.81 # gravity acc
normlen = 10001 # desired number of data points for normalization
speed_threshold = 20 # in kph
quantities = ["x","y","z","speed_kph","engine_rpm","steer","acc_vert","curvature","curv1","curv2","gas","gas1","brake"]

edges_df = pd.read_pickle("C:/Users/its/Documents/racing/edges_ideal_pkl/df_all_edges.pkl") # load data frame with tracks'  edges
edges_bh = edges_df[edges_df.track == 'brandshatch'].reset_index(drop = True) # specify edges of the tacks one is interested in
edges_ok = edges_df[edges_df.track == 'okayama'].reset_index(drop = True)
df_turns = pd.read_pickle("df_turns.pkl") # load data frame with turn information
dfallsubj = pd.read_pickle("df_allpoints.pkl") # upload df where the corner metrics will be saved

id = 959 # participant's ID
overwrite = False # if for any reason it is necessary to re-compute data of one subject only while leaving all the rest unchanged
tracks = {}

for track_name, edges in {"okayama": edges_ok, "brandshatch": edges_bh}.items():
    inner = edges.loc[edges.edge == 'inner', ["x", "z"]].to_numpy(copy=True)
    outer = edges.loc[edges.edge == 'outer', ["x", "z"]].to_numpy(copy=True)
    reverse_z = track_name == "brandshatch"
    reverse_x = track_name == "okayama"

    tracks[track_name] = { #creating a support structure to speed up the code
        "track": TrackLimits(inner, outer),
        "inner": inner,
        "outer": outer,
        "turns": df_turns[df_turns.track == track_name],
        "reverse_x": reverse_x,
        "reverse_z": reverse_z,
        "curv_threshold": 0.25 if track_name == "okayama" else 0.5,
        "frames_threshold": 40 if track_name == "okayama" else 25
    }

files = [f for f in all_files if int(f.split("_")[1]) == id]
dfs = []
for file in files:

    parts = file.split("_")
    level = parts[2]
    track_name = "brandshatch" if parts[3] == "easy" else "okayama"

    session = int(parts[5])
    block = int(parts[7])
    cfg = tracks[track_name]
    data = pd.read_pickle(os.path.join(files_loc, file))
    data = rename_cols_and_filter(data, fs)

    if cfg["reverse_z"]:
        data.z *= -1
        nturns = 5

    if cfg["reverse_x"]:
        data.x *= -1
        nturns = 7

    data = data[data.pit != 1].reset_index(drop=True) # exclude laps with pit stops
    df = perc_completion_norm(data, normlen, quantities, speed_threshold, cfg["track"], fs) # exclude laps based on criteria listed in the paper and mormalized retained laps to 10000 data points
    df_real = entry_apex_clip_exit(df, cfg, nturns, fs) # computes corner metrics
    df_real = df_real.sort_values(["lap", "turn", "quantity"]).reset_index(drop = True)
    points = df[["x", "z"]].iloc[df_real.value.to_numpy()]
    _, idx = cfg['track'].tree.query(points) # compute closest mid line value point to the corner metrics computed above ( all laps have different distance: midline value is a common reference that never changes)
    df_real['midline_value'] = idx/100

    df_real.insert(0, "subject", id)
    df_real.insert(1, "level", level)
    df_real.insert(2, "track", track_name)
    df_real.insert(3, "session", session)
    df_real.insert(4, "block", block)
    dfs.append(df_real)  # append data frame just created

if dfs:

    dfsubj = pd.concat(dfs, ignore_index=True)

    if overwrite:
        dfallsubj = dfallsubj[dfallsubj.subject != id]
        dfallsubj = pd.concat([dfallsubj, dfsubj], ignore_index=True)
    else:
        dfallsubj = pd.concat([dfallsubj, dfsubj], ignore_index=True)

    dfallsubj.to_pickle("df_allpoints.pkl") # saves to pkl


