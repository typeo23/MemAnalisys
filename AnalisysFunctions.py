import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import scipy.stats
import os


def load_and_filter(filname, thikness):
    df = pd.read_csv(filname, delimiter=r'\s+')
    df['q2'] = df['10*q2_uniq_ny']
    cutoff_q = (2 * np.pi) / (2 * thikness)
    df_filtered = df[(df['10*q2_uniq_ny'] <= cutoff_q)]
    return df_filtered


def load_and_plot(filedir, filename, mods, kbt):
    df = pd.read_csv(filedir + filename, delimiter='\s+')
    df['q2'] = df['10*q2_uniq_ny']
    y = n_par(df.q2, kbt / mods['k_c'], mods['c_par'])
    plt.scatter(df['10*q2_uniq_ny'], df['10*q2_uniq_ny'] ** 2 * df['umparq2_uniq'])
    plt.plot(df.q2, y)
    plt.savefig('./figsMartini/' + filename + '_par.png')
    plt.gcf().clear()
    y = n_per(df.q2, mods['k_t'] / kbt, mods['k_tw'] / kbt, mods['c_per'])
    plt.scatter(df['10*q2_uniq_ny'], df['umperq2_uniq'])
    plt.plot(df.q2, y)
    plt.savefig('./figsMartini/' + filename + '_per.png')
    plt.gcf().clear()


def n_par(q, k_c, C):
    return k_c + q ** 2 * C


def n_per(q, k_t, k_tw, C):
    return 1 / (k_t + q ** 2 * k_tw) + C


def h_q(q, k_c, k_t, C):
    return k_c + q ** 2 * k_t + q ** 4 * C


def fit_spectra(df, KbT):
    df_filtered = df[df['10*q2_uniq_ny'] > 0]
    popt_par, pcov_par = curve_fit(n_par, df_filtered.q2, df_filtered.q2**2 * df_filtered.umparq2_uniq)
    popt_per, pcov_per = curve_fit(n_per, df.q2, df.umperq2_uniq, maxfev=1000000, bounds=(0.005, np.inf))
    popt_h, pcov_h = curve_fit(h_q, df_filtered.q2, df_filtered.q2 **4 * df_filtered.hq2_uniq)

    k_c = KbT / popt_par[0]
    c_par = popt_par[1]

    k_t = KbT * popt_per[0]
    k_tw = KbT * popt_per[1]
    c_per = popt_per[2]

    k_c_h = KbT/ popt_h[0]
    k_t_h = KbT / popt_h[1]
    c_h = popt_h[2]

    return k_c, c_par, k_t, k_tw, c_per, k_c_h, k_t_h, popt_h


def get_se(df_list, kbT):
    n = len(df_list)
    k_c = []
    c_par = []
    k_t = []
    k_tw = []
    c_per = []
    k_c_h = []
    k_t_h = []
    c_h = []

    for df in df_list:
        mods = fit_spectra(df, kbT)
        k_c.append(mods[0])
        c_par.append(mods[1])

        k_t.append(mods[2])
        k_tw.append(mods[3])
        c_per.append(mods[4])

        k_c_h.append(mods[5])
        k_t_h.append(mods[6])
        c_h.append(mods[7])

    k_c_se = scipy.stats.sem(k_c)
    c_par_se = scipy.stats.sem(c_par)
    k_t_se = scipy.stats.sem(k_t)
    k_tw_se = scipy.stats.sem(k_tw)
    c_per_se = scipy.stats.sem(c_per)
    k_c_h_se = scipy.stats.sem(k_c_h)
    k_t_h_se = scipy.stats.sem(k_t_h)
    c_h_se = scipy.stats.sem(c_h)

    return k_c_se, c_par_se, k_t_se, k_tw_se, c_per_se, k_c_h_se, k_t_h_se, c_h_se


def mod_from_dir(filedir, dirname):
    thickness = 2
    kb = 1.38e-23
    t = 318
    kbt = kb * t

    dfs = []

    files = os.listdir(filedir)
    for filename in files:
        if filename.startswith('q' + dirname + 'a'):
            break
    file_write = filename

    df = load_and_filter(filedir + filename, thickness)
    mods = fit_spectra(df, kbt)

    files = os.listdir(filedir)
    for filename in files:
        if filename.startswith('SE25a'):
            break
    filedir_se = filedir + filename + '/'
    files = os.listdir(filedir_se)
    for filename in files:
        if filename.split('.')[1] == 'txt':
            dfs.append(load_and_filter(filedir_se + filename, thickness))

    se = get_se(dfs, kbt)

    mods = {'k_c': mods[0], 'k_c_se': se[0], 'c_par': mods[1], 'c_par__se': se[1],
            'k_t': mods[2], 'k_t_se': se[2], 'k_tw': mods[3], 'k_tw_se': se[3], 'c_per': mods[4], 'c_per__se': se[4],
            'k_c_h': mods[5], 'k_c_h_se': se[5], 'k_t_h': mods[6], 'k_t_h_se': se[6], 'c_h': mods[7], 'c_h_se': se[7]}
    load_and_plot(filedir, file_write, mods, kbt)
    return mods
