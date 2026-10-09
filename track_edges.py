import pandas as pd
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from utilities import *


#################################################################################
#################################################################################
# EXECUTE THIS SECTION ONLY THE FIRST TIME: IT CREATES AN EMPTY DATA FRAME

df = pd.DataFrame(columns = ['track', 'edge', 'perc_completion', 'x', 'y', 'z'])
df.to_pickle('/select/your/directory/df_edges.pkl')
#################################################################################
##################################################################################

#EXECUTE THIS SECTION TO OBTAIN THE TRACK EDGES AFTER CREATING AN EMPTY DATA FRAME

fs = 240 # SAMPLING FREQUENCY FROM THE SIMULATOR
dfall = pd.read_pickle('/select/your/directory/df_edges.pkl') # UPLOAD DATA FRAME
overwrite = 0 # 1 IF ANY ERRORS ARE PRESENT IN THE EDGES PREVIOUSLY OBTAINED: IT OVERWRITES ONLY THAT GIVEN TRACK
outer_edge = 1  # LAP WHERE THE CAR WAS DRIVEN ALONG THE OUTER EDGE (THIS SHOULD BE KNOWN FROM DATA COLLECTION)
inner_edge = 3 # SAME AS ABOVE
normlen = 10001 # NUMBER OF DATAPOINTS TO WHICH DATA SHOULD BE NORMALIZED
track_name = 'okayama'
reverse_z = reverse_x = 0 # 1 IF ANY AXIS SHOULD BE REVERSED: SOMETIMES TELEMETRY DATA RETURN THE CIRCUIT UPSIDE-DOWN OR FLIPPED LEFT/RIGHT

if track_name == 'brandshatch': # FOR EXAMPLE, IN OUR DATA BRANDS HATCH INDY WAS  FLIPPED WRT THE Z COORDINATE
    reverse_z = 1
else:
    reverse_x = 1

quantities = ['x', 'y', 'z']

data = pd.read_csv("/select/your/directory//" + track_name + "_edges.csv") # UPLOAD .CSV DATA CONTAINING TELEMETRY FILES
data = rename_cols_and_filter(data, fs) # CUSTOM FUNCTION TO ONLY SELECT THE NECESSARY COLUMNS
data = data[data.lapcount.isin([1,3])] # ONLY RETAIN LAPS DRIVEN ON THE EDGES
data.to_pickle('edges_brandshatch.pkl') # SAVE THEM TO PKL (LIGHTER THAN .CSV) SHOULD ANY FUTURE ANALYSIS BE NECESSARY
df = perc_completion_norm_edges(data, normlen, inner_edge, outer_edge, quantities) # CUSTOM FUNCTION TO NORMALIZE ALL RETAINED COLUMNS TO 10001 DATA POINTS
df['track'] = track_name # ADDING TO THE DATA FRAME A COLUMN WITH TRACK NAME
df = df[['track', 'edge', 'perc_space', 'x', 'y', 'z']] # SORTING DATA FRAME COLUMNS

if reverse_z == 1: # IF ANY AXIS SHOULD BE FLIPPED, FLIP THEM!
    df['z'] *= -1

if reverse_x == 1:
    df['x'] *= -1


# PLOTTING EDGES
matplotlib.use('TkAgg')
plt.plot(df[df.edge == 'inner'].x, df[df.edge == 'inner'].z)
plt.plot(df[df.edge == 'outer'].x, df[df.edge == 'outer'].z)

plt.plot(df[(df.edge == 'inner') & (df.perc_space == 0)].x, df[(df.edge == 'inner') & (df.perc_space == 0)].z, 'o', color = 'red')
plt.plot(df[(df.edge == 'outer') & (df.perc_space == 0)].x, df[(df.edge == 'outer') & (df.perc_space == 0)].z, 'o', color = 'red')
plt.show()


if overwrite == 0: # IF NECESSARY OVERWRITE DATA FROM A GIVEN CIRCUIT, IF ANY ERRORS WERE MADE PRVIOUSLY
    dfall = pd.concat([dfall, df], axis=0, ignore_index=True)
else:
    dfall.loc[dfall.track == track_name, :] = df.values


dfall.to_pickle("/select/your/directory/df_all_edges.pkl") #SAVE DF WITH TRACK EDGES















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
