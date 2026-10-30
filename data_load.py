import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os

files_loc = "select/your/files/directory/data_csv" # SET CSV FILES LOCATION
os.chdir("select/your/working/directory") # SET CURRENT WORKING LOCATION, IF NEEDED
all_files = os.listdir(files_loc)

cols_to_keep = ["time",  "lapCount", "lapTime", "carPositionNormalized", "carCoordinatesX", "carCoordinatesY", "carCoordinatesZ",  
                "speedKmh", "engineRPM", "gear", "gas", "brake", "steer", "isAbsInAction", "isTcInAction", "isInPit",
                "accGVertical", "accGHorizontal", "accGFrontal"] # TELEMETRY DATA HAE APPROX. 90 COLS. HERE WE SELECT THOSE TO BE RETAINED

id = 000 # SUBJECT'S id
for file in all_files: # CONVERTS TO THE  EASIER-TO-READ PKL FORMAT

    parts = file.split("_")
    if int(parts[1]) != id:
        continue

    data = pd.read_csv(files_loc + '/' + file)
    parts = file.split("_")
    data = data[cols_to_keep]
    data.insert(0, 'subject', int(parts[1]))
    data.insert(1, 'level', parts[2])
    data.insert(2, 'track', 'brandshatch' if parts[3] == 'easy' else 'okayama')
    data.insert(3, 'session', int(parts[5]))
    data.insert(4, 'block', int(parts[7]))
    data.to_pickle('data_pkl/' + file[:-4] + '.pkl')

    print('done')