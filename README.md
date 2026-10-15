# Scientific Algorithm for Corner Detection

## Description
This software identifies characteristic points along a racing simulator trajectory.

## Requirements
- Microsoft Excel
- Python 3.10 or newer

## We provide:
- Data sample from a driving session in .csv format
- Data (.csv) containing 2 laps driven slowly with the tip of the car on the inner and outer edges of Brands Hatch Indy to obtain the track edges used for subsequent analysis 
- The scripts to convert the data to .pkl (lightweight and easy-to-read in Python).

####################################################################################################################################################################################

# Procedure
-Firstly, in the working directory, two subfolder should be created: one containing the .csv files to be processed and one where the .pkl files will be saved upon creation. The scripts should be located in the main working directory instead.

## Creating file with track edges
- To run this script the 'track_edges.csv' file is necessary.
  
- When opening the 'track_edges.py' script, it is possible to create an empty data frame which will be filled (and grown, if one needs to analyze multiple circuits) with the coordinates of the track edges. In the script, it is possible specify several parameters including how many data points the edges should be normalized to, sampling rate of the collected data, which driven lap in the .csv file corresponds to the inner or outer edge, the circuit name and whether any coordinates should be reversed (sometimes the circuit may be visualized upside-down or flipped left-to-right). 
- After prompting the file location, one may simply run the script. The output will be a data frame containing the track inner and outer edges normalized to the desired number of data points


## Converting telemetry data from .csv to .pkl

- The first script to be used is 'data_load.py', where the location of the .csv file must be entered as a string to 'file_loc'; the participant ID number must also be entered and the script may now be run. The output will be a .pkl file with the same name as the .csv file processed saved to the previously created subfolder. The reason why only these files are converted to .pkl, whilst the 'track_edge.csv's file is not, is that the latter is just one file per circuit (containing relatively few data), and it must be run just once. Conversely, for the driving data it is plausible to have multiple files per participant, for which the .csv format may be inconvenient.

## Obtaining corner points from telemetry data (entry, maximum curvature, clipping and exit)
- When opening the 'main.py' script, it is possible to create an empty data frame which will be filled (and grown, if one needs to analyze multiple circuits) with the corner points to be computed. The data frame
  can be saved and re-loaded for future subjects.
  The ID number of the subject must be specified, as well as the file location (i.e. the folder containing the .pkl file produced in the previous step). Other parameters (such as speed threshold amd sampling rate to
  name a few) can also be prompted.
  The algorithm will the print what laps were discarded when any of the criteria listed in the article were met. A picture of the car trajectory in each lap will be produced as well.
  Afterwards, the algorithm will compute the relevant points and metrics for each retained lap, and will produce a picture (always for each lap) where car trajectory and relevant points are plotted for each corner. 
  The algorithm will then ask to prompt 'y' or 'n' based on whether the quality of the data is satisfactory and those should be retained (or discarded). Once the answer is entered, the algorithm will end
  or continue to the next file(s) if a given subject has more than one. 
  At the end, the algorithm will save the data frame with the corner metrics for each subject, circuit, lap and corner.
