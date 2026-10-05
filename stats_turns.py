import pandas as pd
import numpy as np
import os
import seaborn as sns
from utilities import *

data = pd.read_pickle('df_allpoints.pkl')
data.insert(9, 'valnorm', np.round((data['value'] % 10000) / 100, 2))
data['quantity'] = data['quantity'].replace('apex', 'max curv')
data["quantity"] = data["quantity"].str.capitalize()
data["quantity"]  = data["quantity"] .replace("Clip", "Clipping")
data["quantity"]  = data["quantity"] .replace("Max curv", "Max Curv")
"""
data['zscore'] = (data['midline_value'] -
                  data.groupby(['subject', 'level', 'track',  'turn',
                                'quantity'])['midline_value'].transform('mean')) / data.groupby([
    'subject', 'level', 'track',  'turn', 'quantity'])['midline_value'].transform('std')
"""

datastats = data.groupby(['subject', 'level', 'track',  'turn', 'quantity'])['midline_value'].agg(mean="mean",
                                                                             std="std").reset_index(drop = False)

########################################### INDIVIDUAL LEVEL ###########################################
id = 227
track = 'brandshatch'
turns_range = [[7.5,20], [25, 36], [42,49], [57,68], [65,85]] if track == 'brandshatch' else \
    [[6,20], [21,31],[30, 38], [44,53], [52,63], [64,74], [74,84]]

turns_range = np.array(turns_range)
trackname = 'Brands Hatch Indy' if track == 'brandshatch' else 'Okayama Piper'
point = 'Clipping'
datasubj = data[(data.subject == id) & (data.track == track)].reset_index(drop = True)
datasubj['quantity'] = datasubj['quantity'].astype('category')
datasubj['turn'] = datasubj['turn'].astype('int')
dataplot = datasubj[(datasubj.quantity == point)]

dataplot['valnorm_range'] = ((dataplot['midline_value'] - turns_range[dataplot['turn'].to_numpy() - 1, 0]) /
                     (turns_range[dataplot['turn'].to_numpy() - 1, 1] - turns_range[dataplot['turn'].to_numpy()  - 1, 0])) * 100

dataplot = dataplot[dataplot['turn'].isin([2,4])].reset_index(drop = True)

matplotlib.use('TkAgg')

from matplotlib.lines import Line2D
fig, axes = plt.subplots(2, 1, figsize=(16, 14), sharex=True, sharey=True)
for ax, turn in zip(axes, [2, 4]):
    for session in [1, 2]:
        lstyle = '-' if session == 1 else '--'
        sns.lineplot(dataplot[dataplot.session == session][dataplot[dataplot.session == session].turn == turn], x='lap', y='valnorm_range', linestyle=lstyle, legend=False, linewidth=2.5, ax=ax, color='black')

    ax.set_title('Turn ' + str(turn), fontsize=30, pad=15)
    ax.set_ylim([30, 90])
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.tick_params(axis='both', labelsize=25)

handles = [Line2D([0], [0], color='black', linestyle='-', label='Session 1'), Line2D([0], [0], color='black', linestyle='--', label='Session 2')]
axes[0].legend(handles=handles, title='Session', fontsize=20, title_fontsize=25,
               ncol=1, loc='upper left', bbox_to_anchor=(0.02, 0.95), frameon=True)

fig.supxlabel('Lap #', fontsize=30)
fig.supylabel('Corner range completion [%]', fontsize=30)
axes[1].set_xticks(np.arange(int(axes[1].get_xlim()[0]), int(axes[1].get_xlim()[1]) + 1, 2))
fig.suptitle(point + ' Point across Laps in ' + trackname, fontsize=40, y=0.995)
plt.tight_layout(rect=[0.06, 0.05, 1, 0.93])
plt.show()



########################################### GROUP LEVEL ###########################################
track = 'okayama'
datastats2 = datastats[datastats.track == track]
# Normalize all quantities relative to entry for each subject, track and turn
entry_values = datastats2[datastats2['quantity'] == 'Entry'].set_index(['subject', 'track', 'turn'])['mean']
datastats2['mean'] = datastats2['mean'] - datastats2.set_index(['subject', 'track', 'turn']).index.map(entry_values)
matplotlib.use('TkAgg')
for level in datastats2.level.unique():
    lev = 'Experienced' if level == 'exp' else 'Novices'
    datanov2 = datastats2[datastats2.level == level]
    circ = 'Okayama Piper' if track == 'okayama' else 'Brands Hatch Indy'

    fig, ax = plt.subplots(figsize=(8, 5))

    quantities = datanov2['quantity'].unique()
    quantities = [q for q in quantities if q != 'entry']
    turns = sorted(datanov2['turn'].unique())
    offsets = np.linspace(-0.1, 0.1, len(quantities))
    colors = sns.color_palette('tab10', len(quantities))

    for quantity, offset, color in zip(quantities, offsets, colors):
        for i, turn in enumerate(turns):
            values = datanov2[(datanov2['quantity'] == quantity) & (datanov2['turn'] == turn)]['mean'].dropna()
            x = i + offset

            ax.scatter([x] * len(values), values, facecolors='none', edgecolors=color, s=120, linewidths=1.7, alpha=1)

            mean = values.mean()
            ci = 1.96 * values.sem()

            ax.errorbar(x, mean, yerr=ci, fmt='o', markersize=22, markerfacecolor=color, markeredgecolor=color, color=color, linewidth=2.5, capsize=4)

    ax.set_xlabel('Corner #', fontsize=30)
    ax.set_ylabel("Location [Lap Completion %]", fontsize=30)
    ax.set_title('Location of Points for ' + lev + ' in ' + circ, fontsize=40)
    ax.set_ylim([-1, 8])
    ax.set_xticks(range(len(turns)))
    ax.set_xticklabels(turns, fontsize=25)
    ax.tick_params(axis='y', labelsize=25)

    handles = [plt.Line2D([0], [0], marker='o', color=color, markerfacecolor=color, markeredgecolor=color, markersize=14, linestyle='None') for color in colors]
    ax.legend(handles, quantities, fontsize=25)

    plt.tight_layout()
    plt.show()

print("done")

