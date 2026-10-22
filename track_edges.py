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
    reverse_x = 1 # OKAYAMA IS FLIPPED WRT THE Z COORDINATE INSTEAD

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

#################################################################################
#################################################################################
# EXECUTE THESE 2 LINES ONLY THE FIRST TIME TO CREATE A NEW DATA FRAME
# create (and grow) df with locations of turns in each track
df = pd.DataFrame(columns = ['track', 'turn', 'start', 'end', 'pre', 'post'])
df.to_pickle('df_turns.pkl')
#################################################################################
#################################################################################
df = pd.read_pickle('df_turns.pkl')

# Brandshatch
track_name = 'brandshatch'
turns_range = [[7.5,20], [25, 36], [42,49], [57,68], [65,85]]
prec = ['s', 's', 's', 's', 't']
foll = ['s', 's', 's', 't', 's']
counter = 0

for i in np.arange(len(turns_range)):  # CREATES DATAFRAME FROM LISTS ABOVE
    counter += 1
    row = [track_name, counter, turns_range[i][0], turns_range[i][1], prec[i], foll[i]]
    df.loc[len(df)] = row

# Okayama
track_name = 'okayama'
turns_range = [[6,20], [21,31],[30, 38], [44,53], [52,63], [64,74], [74,84]] #   EACH COUPLE OF VALUES IS THE START AND END OF A GIVEN CORNER
prec = ['s', 's', 't', 's', 't', 's', 's'] # SPECIFY IF CORNER IS PRECEDED BY A STRAIGHT OR BY ANOTHER TURN
foll = ['s', 't', 's', 't', 's', 's', 's'] # SAME, BUT FOLLOWED INSTEAD OF PRECEDED
counter = 0

for i in np.arange(len(turns_range)): # CREATES DATA FRAME FROM LISTS ABOVE
    counter += 1
    row = [track_name, counter, turns_range[i][0], turns_range[i][1], prec[i], foll[i]]
    df.loc[len(df)] = row

df['start'] = df.start.astype(float)
df['end'] = df.end.astype(float)

df.to_pickle('df_turns.pkl') # SAVES DATA FRAME






