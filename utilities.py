import pandas as pd
import numpy as np
np.set_printoptions(legacy = '1.25')
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.collections import LineCollection
from scipy.signal import butter, filtfilt, find_peaks
from scipy.spatial import cKDTree
from numpy.lib.stride_tricks import  sliding_window_view
def rename_cols_and_filter(df, fs):
    new_df = df.rename(columns = {
        "session": "session",
        "block": "block",
        "time": "time_abs",
        "lapCount": "lapcount",
        "lapTime": "laptime_ms",
        "carPositionNormalized": "disp_perc",
        "carCoordinatesX": "x",
        "carCoordinatesY": "y",
        "carCoordinatesZ": "z",
        "speedKmh": "speed_kph",
        "engineRPM": "engine_rpm",
        "gear": "gear",
        "gas": "gas",
        "brake": "brake",
        "steer": "steer",
        "isAbsInAction": "abs",
        "isTcInAction": "tx",
        "isInPit": "pit",
        "lap_within_block": "lap_block",
        "lap_global": "lap_global",
        "accGVertical": "acc_vert",
        "accGHorizontal": "acc_lat",
        "accGFrontal": "acc_front"})

    # filtering accelerations
    b, a = butter(2, 5 / (0.5 * fs), btype='low', analog=False)
    for col in new_df.columns[new_df.columns.str.contains('acc')]:
        new_df[col] = filtfilt(b, a, new_df[col].values)

    #creating col with time for each lap
    df['timelap_s'] = 0
    for lap in new_df.lapcount.unique():
        new_df.loc[new_df.lapcount == lap, 'timelap_s'] = (
                new_df['time_abs'][new_df.lapcount == lap] - new_df['time_abs'][new_df.lapcount == lap].iloc[0])



    return  new_df

def compute_curvature(data, fs):
    curv_list, curv1_list, curv2_list = [], [], []
    b, a = butter(4, 7 / (0.5 * fs), btype='low', analog=False)

    p = data.timelap_s.values #data.perc.values
    x = data.x.values  # get x
    z = data.z.values  # get z

    x_filt = filtfilt(b, a, x)
    z_filt = filtfilt(b, a, z)

    dx_dp = np.gradient(x_filt, p)
    dz_dp = np.gradient(z_filt, p)
    d2x_dp2 = np.gradient(dx_dp, p)
    d2z_dp2 = np.gradient(dz_dp, p)
    curvature = (dx_dp * d2z_dp2 - dz_dp * d2x_dp2) / (dx_dp ** 2 + dz_dp ** 2) ** 1.5  # larger values = sharper turn

    b, a = butter(4, 2 / (0.5 * fs), btype='low', analog=False) # 0.5 Hz for slow driving ideal lap
    curv_smooth = filtfilt(b, a, curvature)
    curv1 = np.gradient(curv_smooth, p)
    curv1_smooth = filtfilt(b, a, curv1)
    curv2 = np.gradient(curv1_smooth, p)
    curv2_smooth = filtfilt(b, a, curv2)

    curv_list.extend(curv_smooth)
    curv1_list.extend(curv1_smooth)
    curv2_list.extend(curv2_smooth)
    return curv_list, curv1_list, curv2_list
"""
def find_exit(data, curv_threshold, frames_threshold, apex_abs, clip_abs, range_end):

    window_start = apex_abs if apex_abs > clip_abs else clip_abs
    window_end = int(data.index[0] + (range_end * 100))
    curv_max = data.curvature.loc[apex_abs]
    lapafter = data.curvature.loc[window_start : window_end] # take an entire lap after detecting window_start
    lapdelta = np.sign(abs(lapafter) - abs(curv_threshold * curv_max)) #1st sample is apex by def
    signchange = np.insert(np.where(np.diff(lapdelta) != 0)[0], 0,0) # odd index I pass from above to below

    if len(signchange) % 2 == 0:
        signchange = np.append(signchange, len(lapafter))

    minus = np.array([signchange[i+1] - signchange[i] for i in range(1, len(signchange)-1, 2)])


    if any(x > frames_threshold for x in minus):
        exit_abs = window_start + signchange[np.where(minus >= frames_threshold)[0][0] * 2 + 1] + frames_threshold
    else: #assuming signchange has an odd number of elemets
        exit_abs = window_end

    return exit_abs
"""

def find_exit(data, cfg, apex_lap, clip_lap, entry_lap, turn, fs):
    post = cfg['turns'].loc[cfg['turns'].turn == turn, 'post'].iloc[0]

    match post:
        case 's':

            window_start = data.loc[max(apex_lap[turn -1], clip_lap[turn - 1]), 'perc_space']
            window_end = cfg['turns'].loc[cfg['turns'].turn == turn, 'end'].iloc[0]
            mask_window = ((data.perc_space >= window_start) & (data.perc_space <= window_end))
            dataturn = data.loc[mask_window]
            post = cfg['turns'].loc[cfg['turns'].turn == turn, 'post'].iloc[0]
            idx = dataturn.index
            # brake = dataturn['brake'].to_numpy()
            gas = dataturn['gas'].to_numpy()
            curv2 = dataturn['curv2'].to_numpy()

            direction = 'R' if data.loc[apex_lap[turn - 1], 'curvature'] <0 else 'L'
            if direction == 'L': # I want peak in the opposite direction of curvature --> unwind! positive peaks if unwind from right turns and vv
                curv2 *= -1

            # compute first peak in curv 2 --> unwind
            """
            peaks, _ = find_peaks(curv2)
            peak_idx = idx[peaks]
            unwind_idx = peak_idx[0]
            """
            # compute where I have full gas
            mask = gas > 0.75 # how much
            window = int(1 * fs) # how long
            if len(mask) < window: window = len(mask) - 1
            windows = sliding_window_view(mask, window)
            gas_where = np.where(windows.sum(axis = 1) == window)[0]
            if len(gas_where) > 0:
                gas_idx = idx[gas_where[0]]  #if len(gas_where) > 0 else unwind_idx + 1
                #detect = 'unwind' if unwind_idx < gas_idx else 'gas'
                detect = 'gas'
                #return min(unwind_idx, gas_idx), detect
                return gas_idx, detect

            window_end = cfg['turns'].loc[cfg['turns'].turn == turn, 'end'].iloc[0]
            mask_window = ((data.perc_space >= window_start) & (data.perc_space <= window_end))
            dataturn = data.loc[mask_window]

            curv_max = data.loc[apex_lap[turn - 1], 'curvature']
            threshold = abs(cfg['curv_threshold'] * curv_max)
            lappre = dataturn['curvature'].to_numpy()
            mask = np.abs(lappre) > threshold

            diff = np.diff(mask.astype(int))
            starts = np.where(diff == 1)[0] + 1
            ends = np.where(diff == -1)[0]

            if mask[0]:
                starts = np.r_[0, starts]
            if mask[-1]:
                ends = np.r_[ends, len(mask) - 1]

            for s, e in zip(starts, ends):
                if (e - s + 1) >= cfg['frames_threshold']:
                    if s + cfg['frames_threshold'] - 1 < len(mask):
                        thresh_idx = idx[int(s - 1)]
                        detect = 'curv_thresh'
                        return thresh_idx, detect

            detect = 'curv_thresh'
            return idx[-1], detect


        case 't':
            window_start = data.loc[max(apex_lap[turn - 1], clip_lap[turn - 1]), 'perc_space']
            window_end = data.loc[entry_lap[turn] + 1, 'perc_space'] # assume that last turn of a lap is followed by a straight
            mask_window = ((data.perc_space >= window_start) & (data.perc_space <= window_end))
            dataturn = data.loc[mask_window]
            idx = dataturn.index
            # brake = dataturn['brake'].to_numpy()
            gas = dataturn['gas'].to_numpy()
            curv2 = dataturn['curv2'].to_numpy()
            curv = dataturn['curvature'].to_numpy()
            zero_crossing = np.where(np.diff(np.signbit(curv)))[0][-1]

            direction = 'R' if data.loc[apex_lap[turn - 1], 'curvature'] < 0 else 'L'
            if direction == 'L':  # I want peak in the opposite direction of curvature --> unwind!
                curv2 *= -1

            # compute first peak in curv 2 --> unwind
            peaks, _ = find_peaks(curv2)
            if len(peaks) > 0:
                peak_idx = idx[peaks]
                unwind_idx = peak_idx[np.argmax(peak_idx < idx[zero_crossing])]
                detect = 'unwind_0_cross'

                return unwind_idx, detect

            window_end = cfg['turns'].loc[cfg['turns'].turn == turn, 'end'].iloc[0]
            mask_window = ((data.perc_space >= window_start) & (data.perc_space <= window_end))
            dataturn = data.loc[mask_window]

            curv_max = data.loc[apex_lap[turn - 1], 'curvature']
            threshold = abs(cfg['curv_threshold'] * curv_max)
            lappre = dataturn['curvature'].to_numpy()
            mask = np.abs(lappre) > threshold

            diff = np.diff(mask.astype(int))
            starts = np.where(diff == 1)[0] + 1
            ends = np.where(diff == -1)[0]

            if mask[0]:
                starts = np.r_[0, starts]
            if mask[-1]:
                ends = np.r_[ends, len(mask) - 1]

            for s, e in zip(starts, ends):
                if (e - s + 1) >= cfg['frames_threshold']:
                    if s + cfg['frames_threshold'] - 1 < len(mask):
                        thresh_idx = idx[int(s - 1)]
                        detect = 'curv_thresh'
                        return thresh_idx, detect

            detect = 'curv_thresh'
            return idx[-1], detect



def find_entry(data, cfg, apex_lap, clip_lap, turn):

    pre = cfg['turns'].loc[cfg['turns'].turn == turn, 'pre'].iloc[0]
    match pre:
        case 's':
            window_start = cfg['turns'].loc[cfg['turns'].turn == turn, 'start'].iloc[0]
            window_end = data.loc[min(apex_lap[turn - 1], clip_lap[turn - 1]), 'perc_space']
            mask_window = ((data.perc_space >= window_start) & (data.perc_space <= window_end))
            dataturn = data.loc[mask_window]

            if dataturn.empty:
                return window_start

            idx = dataturn.index
            brake = dataturn['brake'].to_numpy()
            gas = dataturn['gas'].to_numpy()
            curv2 = dataturn['curv2'].to_numpy()

            if np.any(brake != 0): # brake present

                brake_start = idx[np.argmax(brake > 0)]
                peaks, _ = find_peaks(curv2)
                peak_idx = idx[peaks]
                valid = peak_idx[peak_idx > brake_start]
                detect = 'brake_peak'
                return valid[0] if len(valid) else brake_start, detect

            if np.any(gas == 0): # gas == 0

                throttle_off = idx[np.where(gas == 0)[0][0]]
                peaks, _ = find_peaks(curv2)
                peak_idx = idx[peaks]
                valid = peak_idx[peak_idx > throttle_off]
                detect = 'gas_0_peak'
                return valid[0] if len(valid) else idx[0], detect

            curv_max = data.loc[apex_lap[turn - 1], 'curvature']
            threshold = abs(cfg['curv_threshold'] * curv_max)
            lappre = dataturn['curvature'].to_numpy()
            mask = np.abs(lappre) > threshold

            diff = np.diff(mask.astype(int))
            starts = np.where(diff == 1)[0] + 1
            ends = np.where(diff == -1)[0]

            if mask[0]:
                starts = np.r_[0, starts]
            if mask[-1]:
                ends = np.r_[ends, len(mask) - 1]

            for s, e in zip(starts[::-1], ends[::-1]):
                if (e - s + 1) >= cfg['frames_threshold']:
                    if s + cfg['frames_threshold']- 1 < len(mask):
                        detect = 'threshold'
                        thresh_idx = idx[int(s - 1)]
                        return thresh_idx, detect

            detect = 'threshold'
            return idx[0], detect

        case 't':

            window_start = cfg['turns'].loc[cfg['turns'].turn == turn, 'start'].iloc[0] - 3 # start scanning 3% before turn range
            window_end = data.loc[apex_lap[turn - 1], 'perc_space']
            mask_window = ((data.perc_space >= window_start) & (data.perc_space <= window_end))
            dataturn = data.loc[mask_window]

            idx = dataturn.index
            curv = dataturn['curvature'].to_numpy()
            curv2 = dataturn['curv2'].to_numpy()
            zero_crossing = np.where(np.diff(np.signbit(curv)))[0][0]
            peaks, _ = find_peaks(curv2)
            valid = peaks[peaks > zero_crossing]
            valid = idx[valid]
            detect = 'zero_crossing_peak'
            """
            plt.figure()
            plt.plot(data.curvature)
            plt.plot(dataturn.curvature)
            plt.show()
            """

            if len(valid): return valid[0], detect
            window_start = data.loc[idx[zero_crossing], 'perc_space']  # start scanning at zero_crossing
            mask_window = ((data.perc_space >= window_start) & (data.perc_space <= window_end))
            dataturn = data.loc[mask_window]
            idx = dataturn.index

            curv_max = data.loc[apex_lap[turn - 1], 'curvature']
            threshold = abs(cfg['curv_threshold'] * curv_max)
            lappre = dataturn['curvature'].to_numpy()
            mask = np.abs(lappre) > threshold

            diff = np.diff(mask.astype(int))
            starts = np.where(diff == 1)[0] + 1
            ends = np.where(diff == -1)[0]

            if mask[0]:
                starts = np.r_[0, starts]
            if mask[-1]:
                ends = np.r_[ends, len(mask) - 1]

            for s, e in zip(starts[::-1], ends[::-1]):
                if (e - s + 1) >= cfg['frames_threshold']:
                    if s + cfg['frames_threshold'] - 1 < len(mask):
                        detect = 'threshold'
                        return idx[int(window_start + s - 1)], detect

            detect = 'threshold'
            return idx[0], detect


def perc_completion_norm(data,normlen,quantities,speed_threshold,track,fs):
    percpoints=np.linspace(0,1,normlen)
    lists={q:[] for q in quantities}
    lists["perc_time"],lists["perc_space"],lists["lap"]=[],[],[]
    b,a=butter(4,2/(0.5*fs),btype="low",analog=False)

    for lap,datalap in data.groupby("lapcount",sort=False):

        x=datalap["x"].to_numpy()
        z=datalap["z"].to_numpy()
        disp=datalap["disp_perc"].to_numpy()
        speed=datalap["speed_kph"].to_numpy()
        timelap=datalap["timelap_s"].to_numpy()

        off_track,_=track.query(np.column_stack((x,z)))

        d=np.diff(np.r_[0,off_track.astype(np.int8),0])
        starts=np.where(d==1)[0]
        ends=np.where(d==-1)[0]
        max_off=0 if len(ends)==0 else np.max(ends-starts)

        reasons=[]

        if np.sum(np.diff(disp)<0)>1: reasons.append("Had spin around")
        if disp.max()<0.99: reasons.append("Did not finish")
        if disp[0]>0.01: reasons.append("Started from box")
        if speed.min()<speed_threshold: reasons.append("Speed threshold violated")
        if max_off> 2*fs: reasons.append("Went off track longer than 1s")

        plt.figure()
        plt.plot(x,z)
        plt.plot(track.inner[:,0],track.inner[:,1])
        plt.plot(track.outer[:,0],track.outer[:,1])
        plt.show()

        if reasons:
            print(f"Lap {lap} was discarded due to: {'; '.join(reasons)}")
            continue

        datalap["curvature"],datalap["curv1"],datalap["curv2"]=compute_curvature(datalap,fs)
        datalap["gas1"]=np.gradient(datalap["gas"].to_numpy(),timelap)

        pts=np.column_stack((x,z))
        dists=np.linalg.norm(np.diff(pts,axis=0),axis=1)

        cum_dist=np.r_[0,np.cumsum(dists)]
        cum_dist/=cum_dist[-1]

        inv_time=timelap/timelap[-1]

        for q in quantities:
            ts=datalap[q].to_numpy()
            if q=="acc_lat":
                ts=filtfilt(b,a,ts)
            lists[q].extend(np.interp(percpoints,cum_dist,ts))

        lists["perc_time"].extend(np.interp(percpoints,cum_dist,inv_time)*100)
        lists["lap"].extend([lap]*normlen)
        lists["perc_space"].extend(percpoints*100)

    return pd.DataFrame(lists)
def plot_exploratory(df, quantities):

    x = df[quantities[0]].values
    z = df[quantities[1]].values
    c = df[quantities[2]].values

    points = np.array([x, z]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    norm = Normalize(vmin=c.min(), vmax=c.max())
    lc = LineCollection(segments, cmap="RdYlGn", norm=norm)
    lc.set_array(c)
    lc.set_linewidth(2.5)
    fig, ax = plt.subplots()
    ax.add_collection(lc)
    ax.autoscale()

    plt.colorbar(lc, ax=ax, label=quantities[-1])
    plt.show()

def entry_apex_clip_exit(data, cfg, nturns, fs):

    (lap_list, turn_list, apex_list, clip_list, exit_list, entry_list,
     detect_apex, detect_clip, detect_exit, detect_entry)  = [[] for _ in range(10)]


    for lap in data.lap.unique():
        datalap = data[data.lap == lap]
        counter = 0
        for turn in cfg['turns'].turn.unique():
            counter += 1
            start = float(cfg['turns'][cfg['turns'].turn == turn]['start'].values)
            end = float(cfg['turns'][cfg['turns'].turn == turn]['end'].values)

            ### finding if left or right turn
            curv_turn = datalap[(datalap['perc_space'] >= start) & (datalap['perc_space'] < end)]
            values, counts = np.unique(np.sign(curv_turn.curvature), return_counts=True)
            direction = 'L' if values[np.argmax(counts)] == 1 else 'R' # right turn = positive curv and v.v.

            ### finding apex;   assuming to drive clockwise, right turns will clip the inner edge, while left turns the outer one

            if direction == 'L':
                apex = np.argmax(datalap.curvature[(datalap.perc_space > start) & (datalap.perc_space < end)])
                edge_clip = cfg['track'].outer_mapped
            else:
                apex = np.argmin(datalap.curvature[(datalap.perc_space > start) & (datalap.perc_space < end)])
                edge_clip = cfg['track'].inner

            apex_abs = datalap.index[0] + np.where(datalap.perc_space > start)[0][0] + apex
            apex_list.append(apex_abs)
            detect_apex.extend(['-'])
            edge_turn = edge_clip[int(start * 100) : int(end * 100), :]

            #computing clipping via nearest neighbour
            tree = cKDTree(edge_turn)
            dist, _ = tree.query(curv_turn[['x', 'z']].to_numpy(), k=1)
            clip_abs = np.argmin(dist) + curv_turn.index[0]
            clip_list.extend([clip_abs])
            detect_clip.extend(['-'])

        counter = 0
        for turn in cfg['turns'].turn.unique():
            counter += 1
            apex_lap = apex_list[-nturns : ]
            clip_lap = clip_list[-nturns : ]

            ### finding entry
            entry_turn, detect = find_entry(datalap, cfg, apex_lap, clip_lap, turn)
            entry_list.append(entry_turn)
            detect_entry.extend([detect])

        counter = 0
        for turn in cfg['turns'].turn.unique():
            counter += 1
            apex_lap = apex_list[-nturns:]
            clip_lap = clip_list[-nturns:]
            entry_lap = entry_list[-nturns:]

            exit_turn, detect = find_exit(datalap, cfg, apex_lap, clip_lap, entry_lap, turn, fs)
            exit_list.append(exit_turn)
            detect_exit.extend([detect])









        matplotlib.use('TkAgg')
        plt.figure()
        plt.title('Lap: ' + str(lap))
        plt.plot(datalap.x, datalap.z, color = 'red')
        clip_lap = [x for x in clip_list if datalap.index[0] <= x <= datalap.index[-1]]
        exit_lap = [x for x in exit_list if datalap.index[0] <= x <= datalap.index[-1]]
        entry_lap = [x for x in entry_list if datalap.index[0] <= x <= datalap.index[-1]]
        apex_lap = [x for x in apex_list if datalap.index[0] <= x <= datalap.index[-1]]

        for apex, clip, exit, entry in zip(apex_lap, clip_lap, exit_lap, entry_lap):
            plt.plot(cfg['track'].inner[:,0], cfg['track'].inner[:,1], color = 'black')
            plt.plot(cfg['track'].outer_mapped[:,0], cfg['track'].outer_mapped[:,1], color = 'black')
            plt.plot(datalap.x[clip], datalap.z[clip], marker='o', color='k', markersize=8, label = 'clip')
            plt.plot(datalap.x[exit], datalap.z[exit], marker='o', color='r', markersize=8, label = 'exit')
            plt.plot(datalap.x[entry], datalap.z[entry], marker='o', color='g', markersize=8, label = 'entry')
            plt.plot(datalap.x[apex], datalap.z[apex], marker='o', color='orange', markersize=8, label = 'apex')

        plt.show()

        lap_list.extend([lap] * nturns)
        turn_list.extend(np.arange(nturns) + 1)

    dfall = pd.DataFrame({'lap': lap_list, 'turn': turn_list, 'entry': entry_list, 'apex': apex_list,
                          'clip': clip_list, 'exit': exit_list, 'd_entry': detect_entry,
                          'd_exit': detect_exit, 'd_apex': detect_apex, 'd_clip': detect_clip})

    df_long = pd.concat([pd.DataFrame({'lap': dfall.lap, 'turn': dfall.turn, 'quantity': q,
                                       'value': dfall[q], 'detect': dfall[f'd_{q}']})
                         for q in ['entry', 'apex', 'clip', 'exit']], ignore_index=True)

    return df_long




def perc_completion_norm_edges(data, normlen, inner_edge, outer_edge, quantities):

    percpoints = np.linspace(0, 1, normlen)
    lists = {q: [] for q in quantities}
    lists['perc_space'], lists['edge'] = [], []

    for lap in [inner_edge, outer_edge]:
        datalap = data[data.lapcount == lap].reset_index(drop=True)

        deltax_array = datalap['x'].iloc[1:].values - datalap['x'].iloc[:-1].values
        deltaz_array = datalap['z'].iloc[1:].values  - datalap['z'].iloc[:-1].values
        dists = np.sqrt(deltax_array ** 2 + deltaz_array ** 2)
        cum_dist = np.insert(np.cumsum(dists), 0 , 0)
        cum_dist /= cum_dist[-1]

        for q in quantities:
            timeserie = datalap[q].values
            ts_new = np.interp(percpoints, cum_dist, timeserie) # interpolate wrt to SPACE completion percentage
            lists[q].extend(ts_new)

        edge = 'inner' if lap == inner_edge else 'outer'
        lists['perc_space'].extend(percpoints * 100)  # passing to percentage 0-100 rather than keeping 0-1
        lists['edge'].extend([edge] * len(percpoints))


    return pd.DataFrame(lists)

def track_rotation(to_rotate, ref):
    deltatot = []
    thetas = np.linspace(0, 2 * np.pi, 3600)

    for theta in thetas:  # rotate by 0.1 degrees

        r = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        to_rotate_rot = (r @ to_rotate.T).T

        deltatot.extend([np.sum(np.linalg.norm(to_rotate_rot - ref, axis=1) ** 2)])

    theta_min = thetas[deltatot.index(min(deltatot))]
    rmin = np.array([[np.cos(theta_min), -np.sin(theta_min)], [np.sin(theta_min), np.cos(theta_min)]])
    to_rotate_rot = (rmin @ to_rotate.T).T

    return to_rotate_rot

import numpy as np

from scipy.interpolate import CubicSpline
from scipy.spatial import cKDTree


class TrackLimits:
    def __init__(self, inner, outer, n_samples=10000):
        self.n_samples=n_samples
        self.inner=self._resample(inner)
        self.outer=self._resample(outer)
        self._build()

    def _resample(self, pts):
        d=np.linalg.norm(np.diff(pts,axis=0,append=pts[:1]),axis=1)
        s=np.cumsum(d);s=np.insert(s,0,0.0);s=s[:-1]/s[-1]
        sn=np.linspace(0,1,self.n_samples,endpoint=False)
        x=np.interp(sn,s,pts[:,0]);y=np.interp(sn,s,pts[:,1])
        return np.column_stack([x,y])

    def _build(self):
        tree_o=cKDTree(self.outer)
        dist,idx=tree_o.query(self.inner)
        self.outer_mapped=self.outer[idx]
        self.center=0.5*(self.inner+self.outer_mapped)
        self.half_width=0.5*np.linalg.norm(self.inner-self.outer_mapped,axis=1)
        t=np.gradient(self.center,axis=0)
        t=t/np.linalg.norm(t,axis=1,keepdims=True)
        self.normal=np.column_stack([-t[:,1],t[:,0]])
        self.tree=cKDTree(self.center)

    def query(self,car_pos):
        car_pos=np.atleast_2d(car_pos)
        _,idx=self.tree.query(car_pos)
        vec=car_pos-self.center[idx]
        lateral=np.sum(vec*self.normal[idx],axis=1)
        off=np.abs(lateral)>self.half_width[idx]
        return off,idx
