import pandas as pd
import numpy as np
import os
from utilities import *


files_loc = "E:/backups matteo/06_2026_ufficio/Documents/projects/LSU/racing/data_pkl"
all_files = os.listdir(files_loc)

fs = 240
g = 9.81
normlen = 10001
speed_threshold = 20
quantities = ["x","y","z","speed_kph","engine_rpm","steer","acc_lat","curvature","curv1","curv2","gas","gas1","brake"]

edges_ok = pd.read_pickle("edges_okayama.pkl")
edges_bh = pd.read_pickle("edges_brandshatch.pkl")
df_turns = pd.read_pickle("df_turns.pkl")
dfallsubj = pd.read_pickle("df_allpoints.pkl")

id = 959 #227 beginner
overwrite = False
tracks = {}

for track_name, edges in {"okayama": edges_ok, "brandshatch": edges_bh}.items():
    inner = edges.loc[edges.lapcount == 3, ["x", "z"]].to_numpy(copy=True)
    outer = edges.loc[edges.lapcount == 1, ["x", "z"]].to_numpy(copy=True)
    reverse_z = track_name == "brandshatch"
    reverse_x = track_name == "okayama"

    if reverse_z:
        inner[:, 1] *= -1
        outer[:, 1] *= -1


    if reverse_x:
        inner[:, 0] *= -1
        outer[:, 0] *= -1


    tracks[track_name] = {
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

    if track_name != 'okayama':
        continue

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


    data = data[data.pit != 1].reset_index(drop=True)
    df = perc_completion_norm(data, normlen, quantities, speed_threshold, cfg["track"], fs)
    df_real = entry_apex_clip_exit(df, cfg, nturns, fs)
    df_real = df_real.sort_values(["lap", "turn", "quantity"]).reset_index(drop = True)
    approved = input(f"{file}: data quality satisfactory? y/n: ")

    if approved != "y":
        break

    df_real.insert(0, "subject", id)
    df_real.insert(1, "level", level)
    df_real.insert(2, "track", track_name)
    df_real.insert(3, "session", session)
    df_real.insert(4, "block", block)

    dfs.append(df_real)

if dfs:

    dfsubj = pd.concat(dfs, ignore_index=True)

    if overwrite:
        dfallsubj = dfallsubj[dfallsubj.subject != id]
        dfallsubj = pd.concat([dfallsubj, dfsubj], ignore_index=True)
    else:
        dfallsubj = pd.concat([dfallsubj, dfsubj], ignore_index=True)

    dfallsubj.to_pickle("df_allpoints.pkl")
######################################################
####### plots


df_real = df_real.sort_values(by=['lap', 'turn', 'quantity']).reset_index(drop = True)
df_ideal = df_ideal.sort_values(by=['lap', 'turn', 'quantity']).reset_index(drop = True)
dfdelta = pd.DataFrame(columns = ['lap', 'turn', 'quantity', 'perc'])

for lap in df_real.lap.unique():
    dflap = df_real[df_real.lap == lap].reset_index(drop = True)
    dfcopy = dflap.copy()
    dfcopy['perc'] = dflap['perc'] - df_ideal['perc']
    dfdelta = pd.concat([dfdelta, dfcopy], axis = 0)

plt.figure()
sns.boxplot(data = dfdelta, x = 'turn', y = 'perc', hue = 'quantity')
plt.show()






for lap in df.lap.unique():
    plt.figure()
    plt.plot(df['curvature'][df.lap ==lap])
    #plt.figure()
    #plt.plot(df['x'][df.lap ==lap], df['z'][df.lap ==lap])
plt.show()


for lap in df.lap.unique():

    fig, ax1 = plt.subplots()
    ax1.plot(df['curvature'][df.lap == lap])
    ax1.plot(df['curv1'][df.lap == lap])
    ax2 = ax1.twinx()
    ax2.plot(df['curv2'][df.lap == lap], color = 'green')
plt.show()

for lap in df.lap.unique():
    plt.figure()
    plt.plot(df.x[df.lap == lap], df.z[df.lap == lap])
    plt.plot(track.inner[:, 0], track.inner[:, 1])
    plt.plot(track.outer[:, 0], track.outer[:, 1])
plt.show()