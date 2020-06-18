'''
Author: Dragan Rangelov (d.rangelov@uq.edu.au)
File Created: 2020-02-05
-----
Last Modified: 2020-02-05
Modified By: Dragan Rangelov (d.rangelov@uq.edu.au)
-----
Licence: Creative Commons Attribution 4.0
Copyright 2019-2020 Dragan Rangelov, The University of Queensland
'''
#===============================================================================
# %% importing libraries
#===============================================================================
import matplotlib as mpl
mpl.use('qt5agg')
mpl.interactive(True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sbn
sbn.set()
from scipy import stats
import h5py
from os.path import dirname
from pathlib import Path
import sys
sys.path.append(dirname(__file__))
import mmodel_reversals as mm
#===============================================================================
# %% setting paths
#===============================================================================
ROOTPATH = Path(__file__).parent
(ROOTPATH / 'Export').mkdir(parents=True, exist_ok=True)
#===============================================================================
# %% defining complex regression function
#===============================================================================
def complexGLM(pred, crit):
    '''
    Compute regression weights for predicting the criterion variable using predictor arrays
    In -> pred = predictor array, crit = criterion vector
    Out -> coefs = regression coefficients/weights
    '''
    pred = np.array(pred)
    crit = np.array(crit)
    if len(crit.shape) < 2:
        crit = crit.reshape(-1, 1)
    if pred.dtype is not np.dtype('complex'):
        pred = np.exp(pred * 1j)
    if crit.dtype is not np.dtype('complex'):
        crit = np.exp(crit * 1j)
    a, b = [crit.shape[0], pred.shape[0]]
    if crit.shape[0] != pred.shape[0]:
        raise ValueError('The two arrays are of incompatible shape, {} and {}'.format(a, b))
    coefs = np.asmatrix(np.asmatrix(pred).H * np.asmatrix(pred)).I * (np.asmatrix(pred).H * np.asmatrix(crit))
    return coefs
#===============================================================================
# %% simulating data
#===============================================================================
np.random.seed(0)
# simulation parameters
trlN = 1000
runN = 10000
simK = np.sort([.1, 2.5, 1., 5., 10.])
# %% simulating six INDEPENDENTLY sampled motions
presDirs_ind = np.angle(
    np.exp(
        np.random.uniform(
            0, 2 * np.pi, 
            size = [runN, trlN, 6]
        ) * 1j
    )
)

percDirs_ind = np.concatenate([
    np.angle(
        np.exp(
            np.array(
                [
                    np.random.vonmises(
                        presDirs_ind, K
                    )
                    for K in simK
                ]
            ) * 1j
        )
    ),
    # no noise condition, K = inf
    presDirs_ind[None]
])
# saving data for independently sampled directions
with h5py.File(ROOTPATH / 'Export' / 'simData.hdf', 'a') as f:
    f.create_dataset(
        name = 'presDirs_ind', 
        data = presDirs_ind, 
        compression = 9
    )
with h5py.File(ROOTPATH / 'Export' / 'simData.hdf', 'a') as f:
    f.create_dataset(
        name = 'percDirs_ind', 
        data = percDirs_ind, 
        compression = 9
    )
presDirs_ind = None
percDirs_ind = None

# %% simulating six DEPENDENTLY sampled motions
frstTar, frstFoil = np.random.choice(
    np.arange(0, 360), 
    size = [2, runN, trlN]
)
frstDis, scndTar = (
    frstTar[None] 
    # random direction (CW/CCW)
    + np.random.choice(
        [-1, 1],
        size = [2, runN, trlN]
    ) 
    # random angular offset
    * np.random.choice(
        np.arange(30, 151),
        size = [2, runN, trlN]
    )
)
scndDis, scndFoil = (
    np.stack(
        [scndTar, frstFoil]
    )
    # random direction (CW/CCW)
    + np.random.choice(
        [-1, 1],
        size = [2, runN, trlN]
    ) 
    # random angular offset
    * np.random.choice(
        np.arange(30, 151),
        size = [2, runN, trlN]
    )
)
presDirs_dep = np.angle(
    np.exp(
        np.deg2rad(np.stack(
            [frstTar, scndTar, frstDis, scndDis, frstFoil, scndFoil],
            axis = -1
        )) * 1j
    )
)

percDirs_dep = np.concatenate([
    np.angle(
        np.exp(
            np.array(
                [
                    np.random.vonmises(
                        presDirs_dep, K
                    )
                    for K in simK
                ]
            ) * 1j
        )
    ),
    # no noise condition, K = inf
    presDirs_dep[None]
])

# saving data for dependently sampled directions
with h5py.File(ROOTPATH / 'Export' / 'simData.hdf', 'a') as f:
    f.create_dataset(
        name = 'presDirs_dep', 
        data = presDirs_dep, 
        compression = 9
    )
with h5py.File(ROOTPATH / 'Export' / 'simData.hdf', 'a') as f:
    f.create_dataset(
        name = 'percDirs_dep', 
        data = percDirs_dep, 
        compression = 9
    )
presDirs_dep = None
percDirs_dep = None

# %% simulating weighting coeficients
simCoefAbs = np.random.uniform(size = [runN, 6])
# the angles of weigthing coeficients
simCoefAng = np.random.uniform(
    0, 2 * np.pi,
    size = [runN, 6]
)
with h5py.File(ROOTPATH / 'Export' / 'simData.hdf', 'a') as f:
    f.create_dataset(
        name = 'coefsAbs', 
        data = simCoefAbs, 
        compression = 9
    )
    f.create_dataset(
        name = 'coefsAng', 
        data = simCoefAng, 
        compression = 9
    )
simCoefAbs = None
simCoefAng = None
#===============================================================================
# %% estimating weighting coefficients under different noise levels
# and for different simulation conditions
#===============================================================================
for cond in ['ind', 'dep', 'dep_ss']:
    # there are three conditions:
    # ind: independently sampled motion
    # dep: dependently sampled motion
    # dep: dependently sampled motion, 100 trials per run
    print('Analysing {} simulation condition'.format(cond.upper()))
    ssize = None
    cond_raw = cond
    if 'ss' in cond.split('_'):
        cond, ssize = cond.split('_')
    with h5py.File(ROOTPATH / 'Export' / 'simData.hdf', 'r') as f:
        presDirs = f['presDirs_{}'.format(cond)][:]
        percDirs = f['percDirs_{}'.format(cond)][:]
        coefsAbs = f['coefsAbs'][:]
        coefsAng = f['coefsAng'][:]
    if ssize:
        presDirs = presDirs[:, :100]
        percDirs = percDirs[:, :, :100]

    # running complex-values OLS for different simulated weight angles
    for idx_simAngle, simAngle in enumerate(['null', 'real']):
        # two analyses are run
        # null: the angles of the simulated complex-valued regression weights are zero
        # real: the angles are are randomly sampled 
        simCoefs = (
            np.exp(
                [0, 1][idx_simAngle] * coefsAng * 1j
            ) * coefsAbs
        )   
        # %% simulating response on the basis of perceived directions and simulated
        respDirs = np.array([
            np.angle(
                np.sum(
                    simCoefs[:, None] 
                    * np.exp(simKappa * 1j), 
                    -1))
            for simKappa in percDirs
        ])
        # weighting coefficients
        coefs = np.array(
            [
                [
                    complexGLM(presDirs[idxRun], run)
                    for idxRun, run in enumerate(simKappa)
                ]
                for  simKappa in respDirs
            ]    
        ).squeeze()
        print('Finished complex OLS')
        # %% goodness of fit
        predDirs = np.array([
            np.angle(
                np.sum(
                    simKappa[:, None, :] 
                    * np.exp(presDirs * 1j), -1
                )
            )
            for simKappa in coefs
        ])
        GoF = np.array([
            np.angle(
                np.exp(respDirs[simKappa] * 1j)
                / np.exp(predDirs[simKappa] * 1j)
            )
            for simKappa in range(coefs.shape[0])
        ])
        # saving data
        with h5py.File(ROOTPATH / 'Export' / 'simCoefs.hdf', 'a') as f:
            f.create_dataset(
                name = 'coefsAbsHat_{}_{}'.format(cond_raw,simAngle), 
                data = np.abs(coefs), 
                compression = 9
            )
            f.create_dataset(
                name = 'coefsAngHat_{}_{}'.format(cond_raw,simAngle), 
                data = np.angle(coefs), 
                compression = 9
            )
            f.create_dataset(
                name = 'GoF_{}_{}'.format(cond_raw,simAngle), 
                data = GoF, 
                compression = 9
            )
#===============================================================================
# %% Plotting
#===============================================================================
# two different plottings can be performed
# first, the results for simulated complex-valued weights using real angles
# second, the results for simulated weights using zero angles
# here, only the real values are plotted.
# N.B., the results for zero angles yields similart goodness-of-fit
# N.B., the ability of the complex-valued OLS to recover the angles (not plotted)
# is similar to its ability to recover the lengths, i.e., the decision weights .
conds = [
    'GoF_ind_real',
    'GoF_dep_real',
    'GoF_dep_ss_real'
]
with h5py.File(ROOTPATH / 'Export' / 'simCoefs.hdf', 'r') as f:
        GoF = dict([(cond, f[cond][:]) for cond in conds])
# %% Plotting SF1
sbn.set_style('ticks')
SSIZE = 8
MSIZE = 10
LSIZE = 12
params = {'lines.linewidth' : 1.5,
          'grid.linewidth' : 1,
          'xtick.labelsize' : MSIZE,
          'ytick.labelsize' : MSIZE,
          'xtick.major.width' : 1,
          'ytick.major.width' : 1,
          'xtick.major.size' : 5,
          'ytick.major.size' : 5,
          'xtick.direction' : 'inout',
          'ytick.direction' :'inout',
          'axes.linewidth': 1,
          'axes.labelsize' : MSIZE,
          'axes.titlesize' : MSIZE,
          'figure.titlesize' : LSIZE,
          'font.size' : MSIZE,
          'savefig.dpi': 300,
          'font.sans-serif' : ['Calibri'],
          'legend.fontsize' : MSIZE,
          'hatch.linewidth' : .2}
sbn.mpl.rcParams.update(params)
cols = sbn.husl_palette(6, h = .15, s = .75, l = .5)
simK = np.sort([.1, 2.5, 1., 5., 10.])
simNoise = np.random.vonmises(0, simK[:, None], [5, 100000])
fig = plt.figure(figsize = (8,2.8))
ax = fig.add_subplot(1, 4, 1)
for idx_noise, noise in enumerate(simNoise):
    sbn.kdeplot(
        noise,
        color = cols[idx_noise],
        alpha = .8,
        lw = 2,
        label = simK[idx_noise],
        ax = ax
    )
ax.axvline(0, color = cols[-1], alpha = .8, lw = 2, label = 'No noise')
for idx_cond, cond in enumerate(conds):
    ax = fig.add_subplot(1,4,2 + idx_cond)
    for idxK, err in enumerate(GoF[cond]):
        sbn.kdeplot(
            err.flatten(),
            color = cols[idxK],
            alpha = .8,
            lw = 2,
            label = '{}$\degree$'.format(
                np.rad2deg(mm.cstd(err.flatten())).astype('int')
            ),
            ax = ax
        )
for idx_ax, ax in enumerate(fig.axes):
    title = '$\kappa$'
    xlab = 'Perceptual noise'
    if idx_ax:
        title = '$\sigma$'
        xlab = 'Prediction error'
    ax.legend(
        title = title, 
        frameon = False,
        handlelength = 1,
        handletextpad = .5,
        markerfirst = False
    )
    ax.set_ylim(-0.05, 7)
    ax.set_xlim(-np.pi*1.1, np.pi*1.1)
    ax.set_xticks([-np.pi, 0, np.pi])
    ax.set_xticklabels(['-$\pi$', '0', '$\pi$'])
    ax.set_yticks([])
    ax.set_xlabel(xlab)
    ax.set_ylabel('Probability density')
    sbn.despine(ax = ax)
    ax.spines['bottom'].set_bounds(-np.pi, np.pi)
    ax.spines['left'].set_visible(False)
    if idx_ax:
        ax.yaxis.set_visible(False)
plt.tight_layout(rect = (0, 0, 1, 1))
fig.savefig(
    str(ROOTPATH / 'Export'/ 'GoodnessOfFit_All.png'), 
    dpi = 600
)
plt.close(fig)
# %% plotting SF2 panels
with h5py.File(ROOTPATH / 'Export' / 'simData.hdf', 'r') as f:
    coefsAbs = f['coefsAbs'][:]
conds = [
    'ind_real',
    'dep_real',
    'dep_ss_real'
]
cols = sbn.husl_palette(6, h = .15, s = .75, l = .5)
for idx_cond, cond in enumerate(conds):
    fig = plt.figure(figsize = (4,2.8))
    with h5py.File(ROOTPATH / 'Export' / 'simCoefs.hdf', 'r') as f:
        coefsAbsHat = f['_'.join(['coefsAbsHat', cond])][:]
    for idxK, weights in enumerate(coefsAbsHat):
        ax = fig.add_subplot(2, 3, idxK + 1)
        scatter = ax.plot(
            coefsAbs.flatten(), 
            weights.flatten(), 
            '.',
            mec = (.9,.9,.9),
            mfc = 'none',
            zorder = -10
        )
        line = ax.plot(
            np.array([0, 1]), np.array([0, 1]), 
            'k--',
            lw = 1,
            zorder = 0
        )
        bins = pd.qcut(coefsAbs.flatten(), 4).codes
        dataset = [weights.flatten()[bins == bin] for bin in np.unique(bins)]
        vlnplt = ax.violinplot(
            dataset, 
            positions = [.125, .375, .625, .875],
            showextrema = False,
            showmedians = True,
            widths = .15,
        ) 
        for i in vlnplt['bodies']:
            i.set_alpha(.8)
            i.set_facecolor(cols[idxK])
            i.set_lw(0)
        vlnplt['cmedians'].set_edgecolor('white')
        vlnplt['cmedians'].set_lw(.5)
        ax.text(
            .05, .95,
            (
                ['$\kappa$ = {}'.format(k) for k in simK] 
                + ['No noise']
            )[idxK],
            transform = ax.transAxes,
            va = 'top'
        )
        ax.set_xlabel('Simulated weights')
        ax.set_ylabel('Estimated weights')
    for idx_ax, ax in enumerate(fig.axes):
        ax.tick_params('both', direction = 'out')
        ax.set_xlim(-.1, 1.1)
        ax.set_ylim(-.1, 1.1)
        ax.spines['bottom'].set_bounds(0,1)
        ax.spines['left'].set_bounds(0,1)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xticks(np.linspace(0, 1, 3))
        ax.set_yticks(np.linspace(0, 1, 3))
        if idx_ax not in [0, 3]:
            ax.yaxis.set_visible(False)
            ax.spines['left'].set_visible(False)
        if idx_ax not in [3, 4, 5]:
            ax.xaxis.set_visible(False)
            ax.spines['bottom'].set_visible(False)
    plt.tight_layout(rect = (0, 0, 1, .975))
    label = [
        'Independently sampled motion, 10$^3$ trials, 10$^4$ runs',
        'Dependently sampled motion, 10$^3$ trials, 10$^4$ runs',
        'Dependently sampled motion, 10$^2$ trials, 10$^4$ runs'
    ][idx_cond]
    fig.text(
        .5, 1, 
        label,
        ha = 'center',
        va = 'top'
    )
    fig.savefig(
        str(
            ROOTPATH / 
            'Export' / 
            'WeightRecovery_{}.png'
        ).format([
            'A', 'B', 'C'
        ][idx_cond]),
        dpi = 600
    )
    plt.close(fig)
# %% SF2 panel D
from mpl_toolkits.axes_grid1 import ImageGrid
cols = sbn.husl_palette(6, h = .15, s = .75, l = .5)
fig = plt.figure(figsize = (4,2.8))
grid = ImageGrid(
    fig, 111, nrows_ncols = (2, 3), 
    share_all = True, cbar_mode= 'single', aspect= True
)
for idxK, weights in enumerate(coefsAbsHat):
    ax = grid[idxK]
    heatmap, xedges, yedges = np.histogram2d(
        np.array(list(map(
                stats.rankdata,
                coefsAbs
            ))).flatten(), 
        np.array(list(map(
            stats.rankdata,
            weights
        ))).flatten(),
        bins = np.linspace(.5, 6.5, 7)
    )
    heatmap /= heatmap.sum()
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    im = ax.imshow(
        heatmap, 
        extent = extent, origin = 'lower', 
        vmin = 0, vmax = .15,
        cmap = 'viridis'
    )
    ax.text(
        .05, .95,
        (
            ['$\kappa$ = {}'.format(k) for k in simK] 
            + ['No noise']
        )[idxK],
        transform = ax.transAxes,
        va = 'top',
        color = 'white'
    )
grid.cbar_axes[0].colorbar(im)
grid.cbar_axes[0].set_ylim(0, .14)
grid.cbar_axes[0].set_yticks([.0, .05, .10, .15])
grid.cbar_axes[0].set_yticklabels(['0','5','10', '15'])
grid.cbar_axes[0].tick_params(direction = 'inout', length = 5)
grid[0].tick_params('both', direction = 'out', length = 5)
for idx_ax, ax in enumerate(grid):
    ax.tick_params('both', direction = 'inout', length = 5)
    ax.set_yticks(np.linspace(1,6,6))
    ax.set_xticks(np.linspace(1,6,6))
    if idx_ax not in [0, 3]:
        ax.yaxis.set_visible(False)
    if idx_ax < 3:
        ax.xaxis.set_visible(False)
plt.tight_layout(rect = (.01, .01, .94, .99))
fig.text(
    .5, .99, 
    'Dependently sampled motion, 10$^2$ trials, 10$^4$ runs',
    ha = 'center',
    va = 'top'
)
fig.text(
    .01, .5,
    'Estimated weight rank',
    ha = 'left',
    va = 'center',
    rotation = 90
)
fig.text(
    .5, .01,
    'Simulated weight rank',
    ha = 'center',
    va = 'bottom',
)
fig.text(
    .99, .5,
    'Frequency [%]',
    ha = 'right',
    va = 'center',
    rotation = -90
)
fig.savefig(
    str(
        ROOTPATH /
        'Export' /
        'WeightRecovery_D.png'
    ), 
    dpi = 600
)
plt.close(fig)
# %%
