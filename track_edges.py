import pandas as pd
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns
matplotlib.use('TkAgg')
from utilities import *
from scipy.signal import butter, filtfilt

fs = 240
#####################################################################
# get track edges
#df = pd.DataFrame(columns = ['track', 'edge', 'perc_space', 'x', 'y', 'z'])
#df.to_pickle('edges_ideal_pkl/df_all_edges.pkl')
dfall = pd.read_pickle('edges_ideal_pkl/df_all_edges.pkl')

overwrite = 0
outer_edge = 1
inner_edge = 3
normlen = 10001
track_name = 'okayama'
reverse_z = reverse_x = 0

if track_name == 'brandshatch':
    reverse_z = 1
else:
    reverse_x = 1

quantities = ['x', 'y', 'z']


data = pd.read_csv("files/track_edges_ideal_traj/" + track_name + "_edges.csv")
data = rename_cols_and_filter(data, fs)
data = data[data.lapcount.isin([1,3])]
data.to_pickle('edges_brandshatch.pkl')

data = pd.read_pickle('edges_' + track_name + '.pkl')
df = perc_completion_norm_edges(data, normlen, inner_edge, outer_edge, quantities)
df['track'] = track_name
df = df[['track', 'edge', 'perc_space', 'x', 'y', 'z']]


if reverse_z == 1:
    df['z'] *= -1

if reverse_x == 1:
    df['x'] *= -1


matplotlib.use('TkAgg')
plt.plot(df[df.edge == 'inner'].x, df[df.edge == 'inner'].z)
plt.plot(df[df.edge == 'outer'].x, df[df.edge == 'outer'].z)

plt.plot(df[(df.edge == 'inner') & (df.perc_space == 0)].x, df[(df.edge == 'inner') & (df.perc_space == 0)].z, 'o', color = 'red')
plt.plot(df[(df.edge == 'outer') & (df.perc_space == 0)].x, df[(df.edge == 'outer') & (df.perc_space == 0)].z, 'o', color = 'red')
plt.show()

if overwrite == 0:
    dfall = pd.concat([dfall, df], axis=0, ignore_index=True)
else:
    dfall.loc[dfall.track == track_name, :] = df.values # if I forget to reverse z or anything


dfall.to_pickle("edges_ideal_pkl/df_all_edges.pkl")

#####################################################################
# get ideal lap
#dfid = pd.DataFrame(columns = ['track', 'perc_time', 'perc_space', 'x', 'y', 'z', 'curvature']) # solo 1 volta
#dfid.to_pickle('edges_ideal_pkl/df_all_ideal.pkl')
"""
dfid = pd.read_pickle('edges_ideal_pkl/df_all_ideal.pkl')
matplotlib.use('TkAgg')
fs = 240
lap_traj = 1
reverse_z = 1
reverse_x = 0
overwrite = 0
#track_name = 'brandshatch'
normlen = 10001
track_name = 'brandshatch'

data = pd.read_csv("data_csv/" + track_name + '_ideal.csv')
data = rename_cols_and_filter(data, fs)
data["lap_global"] = ((data["disp_perc"][1:].reset_index(drop = True).diff() < 0)
                      & (data["disp_perc"] >= 0.99)).shift(1, fill_value=False).cumsum() + 1

quantities = ['x', 'y', 'z', 'curvature']
data_curv = data[data.lapcount== lap_traj].reset_index(drop = True)


if reverse_z == 1:
    data_curv['z'] *= -1

if reverse_x == 1:
    data_curv['x'] *= -1

data_curv['curvature'] = compute_curvature(data_curv, fs)

df = perc_completion_norm(data_curv, normlen, quantities , -1)
df = df.drop(columns=["lap"])
df['track'] = track_name
df = df[['track', 'perc_time', 'perc_space', 'x', 'y', 'z', 'curvature']]

dfedges = pd.read_pickle("edges_ideal_pkl/df_all_edges.pkl")
dfedges = dfedges[dfedges.track == track_name]

matplotlib.use('TkAgg')
plt.plot(dfedges[dfedges.edge == 'inner'].x, dfedges[dfedges.edge == 'inner'].z)
plt.plot(dfedges[dfedges.edge == 'outer'].x, dfedges[dfedges.edge == 'outer'].z)
plt.plot(dfedges[(dfedges.edge == 'inner') & (dfedges.perc_space == 0)].x,
         dfedges[(dfedges.edge == 'inner') & (dfedges.perc_space == 0)].z, 'o', color = 'red')
plt.plot(dfedges[(dfedges.edge == 'outer') & (dfedges.perc_space == 0)].x,
         dfedges[(dfedges.edge == 'outer') & (dfedges.perc_space == 0)].z, 'o', color = 'red')

plt.plot(df.x, df.z)
plt.plot(df[df.perc_space == 0].x, df[df.perc_space == 0].z, 'o', color = 'red')
plt.show()

plt.figure()
plt.plot(df[df.track == track_name].perc_space, df[df.track == track_name].curvature)
plt.show()


if overwrite == 0:
    dfid = pd.concat([dfid, df], axis=0, ignore_index=True)
else:
    dfid.loc[dfid.track == track_name, :] = df.values # if I forget to reverse z or anything

dfid.to_pickle('edges_ideal_pkl/df_all_ideal.pkl')

print('done')
"""
#####################################################################
#compute apex entry exit clip i ideal lap
"""
edges = pd.read_pickle('edges_ideal_pkl/df_all_edges.pkl')
df_id = pd.read_pickle('edges_ideal_pkl/df_all_ideal.pkl')
df_turns = pd.read_pickle('df_turns.pkl')
speed_threshold = 20
frames_threshold = 25
df_id_tot = pd.DataFrame(columns= ['lap', 'turn', 'quantity', 'perc'])


track_names = ['brandshatch', 'okayama']
for track_name in track_names:

    curv_threshold = 0.45 if track_name == 'okayama' else 0.5
    df_id['lap'] = 1 # forgot to add lap to df_id
    df_track = df_id[df_id.track == track_name].reset_index(drop = True)
    df_ideal = entry_apex_clip_exit(df_track, edges[edges.track == track_name], df_turns[df_turns.track == track_name],
                           frames_threshold, curv_threshold)
    df_ideal = df_ideal.sort_values(by=['lap', 'turn', 'quantity'])
    df_id_tot = pd.concat([df_id_tot, df_ideal], ignore_index=True)

df_id_tot.to_pickle('df_allpoints_ideal.pkl')
"""
#####################################################################
"""
# create (and grow) df with locations of turns in each track
df = pd.DataFrame(columns = ['track', 'turn', 'start', 'end', 'pre', 'post'])
#df.to_pickle('df_turns.pkl')
#df = pd.read_pickle('df_turns.pkl')

# Brandshatch
track_name = 'brandshatch'
turns_range = [[7.5,20], [25, 36], [42,49], [57,68], [65,85]]
prec = ['s', 's', 's', 's', 't']
foll = ['s', 's', 's', 't', 's']
counter = 0
for i in np.arange(len(turns_range)):
    counter += 1
    row = [track_name, counter, turns_range[i][0], turns_range[i][1], prec[i], foll[i]]
    df.loc[len(df)] = row


# Okayama
track_name = 'okayama'
turns_range = [[6,20], [21,31],[30, 38], [44,53], [52,63], [64,74], [74,84]]
prec = ['s', 's', 't', 's', 't', 's', 's']
foll = ['s', 't', 's', 't', 's', 's', 's']
counter = 0
for i in np.arange(len(turns_range)):
    counter += 1
    row = [track_name, counter, turns_range[i][0], turns_range[i][1], prec[i], foll[i]]
    df.loc[len(df)] = row

df['start'] = df.start.astype(float)
df['end'] = df.end.astype(float)

df.to_pickle('df_turns.pkl')
"""

track_name = 'okayama'
edges = pd.read_pickle('edges_ideal_pkl/df_all_edges.pkl')
df_id = pd.read_pickle('edges_ideal_pkl/df_all_ideal.pkl')


plt.figure()
for edge in ['outer', 'inner']:

    plt.plot(edges.x[(edges.track == track_name) & (edges.edge == edge)],
             edges.z[(edges.track == track_name) & (edges.edge == edge)])

plt.plot(df_id.x[df_id.track == track_name], df_id.z[df_id.track == track_name])
plt.xlabel('X coordinate', fontsize = 35)
plt.ylabel('Z coordinate', fontsize = 35)
plt.title('Okayama Piper', fontsize = 45)

track = edges[edges.track == track_name]

for edge in ['outer', 'inner']:
    data = track[track.edge == edge]
    data_100 = data[data.perc_space % 1 == 0]
    plt.plot(data_100.x, data_100.z, 'o', markersize=3, color='grey', label='1% track length' if edge == 'inner' else None)
    data_10 = data[data.perc_space % 10 == 0]
    plt.plot(data_10.x, data_10.z, 'o', markersize=10, color='black', label='10% track length' if edge == 'inner' else None)
    data_0 = data[data.perc_space == 0]
    plt.plot(data_0.x, data_0.z, 'o', markersize=10, color='red', label = 'Lap start / end' if edge == 'inner' else None)



plt.tick_params(axis='both', labelsize=25)
plt.legend(fontsize=30)
plt.show()


for lap in df.lap.unique():
    plt.figure()
    plt.plot(df['curvature'][df.lap ==lap])
    #plt.figure()
    #plt.plot(df['x'][df.lap ==lap], df['z'][df.lap ==lap])
plt.show()
