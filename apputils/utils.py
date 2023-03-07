import plotly.graph_objects as go
from plotly.subplots import make_subplots

import os
import re
import pandas as pd
import numpy as np
from numba import jit
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error

#@jit(nopython=True)
def get_averted_mat(Y, Y0, Yaux=None, method="ratio"):
    averted_mat = np.zeros((Y.shape[0], int(Y.shape[-1]*Y0.size)), dtype=np.float32)
    for i in np.arange(Y.shape[0]):
        y_arr = Y[i,:]
        j = 0
        for yi, y in enumerate(y_arr):
            for y0 in Y0:
                if method == 'ratio':
                    if y0 > 0:
                        averted_mat[i,j] = (100*((y0-y)/y0))
                elif method == 'difference':
                    averted_mat[i,j] = y0-y
                j += 1

    return averted_mat

def perform_linear_regression(X, Y):
    slope, intercept, r_value, p_value, std_err = stats.linregress(X, Y)
    return slope, intercept, r_value, p_value, std_err

def linear_fn(X, slope, intercept):
    return slope*X + intercept

@jit(nopython=True)
def generate_bs_samples(X, Y, reps):
    if len(X) != len(Y):
        raise Exception("generate_bs_samples: X and Y must be of the same length.")
    # remove nans
    mask = ~np.isnan(Y)
    X = X[mask]
    Y = Y[mask]
    bs_samples = np.zeros((reps, 2, len(X)), dtype=np.float32)
    for i in np.arange(reps):
        sampled_idx = np.random.choice(len(X), len(X), replace=True)
        bs_samples[i,0,:] = X[sampled_idx]
        bs_samples[i,1,:] = Y[sampled_idx]
    return bs_samples

@jit(nopython=True)
def get_jackknife_mean(arr):
    arr_idx = np.arange(len(arr))
    out_arr = np.zeros(len(arr))
    for i in arr_idx:
        out_arr[i] = np.mean(arr[arr_idx!=i])
    return np.mean(out_arr)

def get_bootstrap_ci(bs_samples, conf_level=0.95):
    slope_intercept_arr = np.zeros((bs_samples.shape[0], 2), dtype=np.float32)
    for i in np.arange(bs_samples.shape[0]):
        X = bs_samples[i,0,:]
        Y = bs_samples[i,1,:]
        slope, intercept = perform_linear_regression(X, Y)[:2]
        slope_intercept_arr[i,0] = slope
        slope_intercept_arr[i,1] = intercept
    ci_slope_lower, ci_slope_upper = np.percentile(slope_intercept_arr[:,0], [(1 - conf_level) / 2 * 100, (1 + conf_level) / 2 * 100])
    ci_intercept_lower, ci_intercept_upper = np.percentile(slope_intercept_arr[:,1], [(1 - conf_level) / 2 * 100, (1 + conf_level) / 2 * 100])
    return ci_slope_lower, ci_slope_upper, ci_intercept_lower, ci_intercept_upper

def get_linregress_plot_coords(X, Y, Yaux=None, Ymask=None, Y0=None, method='ratio', ci_method='bootstrap', bsrep=100):
    '''
    Get plotting coordiates for proportion averted
    '''
    # get matrix of averted values for each X value (row)
    if method == 'ratio':
        Ydat = get_averted_mat(Y, Y0, method='ratio')
    elif method == 'difference':
        Ydat = get_averted_mat(Y, Y0, method='difference')
    elif method == 'ratiodiff':
        Ydat = get_averted_mat(Y, Y0, Yaux=Yaux, method='ratiodiff')
    elif method == 'absolute':
        Ydat = Y
    else:
        raise Exception("get_linregress_plot_coords requires method input.")
    # return unique X values, preserving order
    plot_x = pd.unique(X)
    # regress using jackknife mean
    try:
        jackknife_y = np.asarray([get_jackknife_mean(Ydat[i,:][Ymask[i,:]]) for i in np.arange(len(plot_x))])
    except:
        jackknife_y = np.asarray([get_jackknife_mean(Ydat[i,:]) for i in np.arange(len(plot_x))])

    slope, intercept, r_value, p_value, std_err = perform_linear_regression(plot_x, jackknife_y)
    plot_y = linear_fn(plot_x, slope, intercept)
    plot_y[plot_y<0] = 0

    if ci_method == 'bootstrap':
        # compute CI bounds using bootstrapping method
        fitX = np.repeat(plot_x, Ydat.shape[-1])
        fitY = Ydat.flatten()
        """try:
            fitYmask = Ymask.flatten()
            fitY = fitY[fitYmask]
            fitX = fitX[fitYmask]
        except:
            pass"""
        bs_samples = generate_bs_samples(fitX, fitY, bsrep)
        ci_slope_lower, ci_slope_upper, ci_intercept_lower, ci_intercept_upper = get_bootstrap_ci(bs_samples)
        plot_y_lb = linear_fn(plot_x, ci_slope_lower, ci_intercept_lower)
        plot_y_ub = linear_fn(plot_x, ci_slope_upper, ci_intercept_upper)
    return plot_x, plot_y, plot_y_lb, plot_y_ub

def parse_sim_results(fpath, **kwargs):
    """
    Parse and read simulation result files
    """
    arr = np.load(fpath)
    # unpack arrays
    cumm_epi_arr = arr['cumm_epi_arr']
    new_epi_arr = arr['new_epi_arr']
    test_results = arr['test_results']
    # compute total number of invasive cases and deaths
    n_invasive = new_epi_arr[:,:,1].sum()
    mu_invasive_per_week = new_epi_arr[:,:,1].mean()
    n_deaths = new_epi_arr[:,:,-1].sum()
    # compute test performance
    n_tests = test_results.sum()
    n_TP, n_FN, n_TN, n_FP = test_results[0,:,:].sum(axis=0)
    if (n_TP+n_FP) > 0:
        ppv = n_TP/(n_TP+n_FP)
    else:
        ppv = None
    if (n_TN+n_FN) > 0:
        npv = n_TN/(n_TN+n_FN)
    else:
        npv = None
    n_corr_diag = n_TP + n_TN
    n_incorr_diag = n_FP + n_FN
    # tests over time
    tests_over_time = test_results[0,:,:].sum(axis=1)
    true_prevalence = (cumm_epi_arr[:,1:3].sum(axis=1)/cumm_epi_arr[0,0])

    pos_cases_over_time = test_results[0,:,0] + test_results[0,:,3]
    test_pos_rate = np.zeros(len(tests_over_time), dtype=np.float32)
    test_pos_rate[tests_over_time>0] = pos_cases_over_time[tests_over_time>0]/tests_over_time[tests_over_time>0]
    case_pos_rate = pos_cases_over_time/cumm_epi_arr[0,0]
    # compare prevalence
    surv_prevalence = test_pos_rate**0.5 * case_pos_rate**0.5

    try:
        rmse_prev = (100 * np.sqrt(mean_squared_error(true_prevalence, surv_prevalence)))
    except:
        rmse_prev = None

    try:
        mae_prev = (100 * mean_absolute_error(true_prevalence, surv_prevalence))
    except:
        mae_prev = None

    # compute vaccine-related parameters
    n_vacc = arr['vacc_counts'][0]
    vacc_trigger_day = -1
    if kwargs['baseline'] < 1:
        for t in np.arange(test_results.shape[1]):
            cumm_pos_cases_t = test_results[:,:t+1,0].sum() + test_results[:,:t+1,3].sum()
            if cumm_pos_cases_t >= kwargs['react_vacc_threshold']:
                vacc_trigger_day = t
                break

    result_row = {
        'paridx':kwargs['paridx'],
        'baseline':kwargs['baseline'],
        'runidx':kwargs['runidx'],
        'test_sens':kwargs['test_sens'],
        'test_spec':kwargs['test_spec'],
        'test_receptiveness':kwargs['test_receptiveness'],
        'n_invasive':n_invasive,
        'mu_invasive_per_week':mu_invasive_per_week,
        'n_deaths':n_deaths,
        'ppv':ppv, 'npv':npv,
        'n_corr_diag':n_corr_diag, 'n_incorr_diag':n_incorr_diag,
        'n_tests':n_tests,
        'n_vacc':n_vacc,
        'rmse_prev':rmse_prev, #'rmse_prev_ratio':rmse_prev_ratio,
        'mae_prev':mae_prev,
        'vacc_trigger_day':vacc_trigger_day
    }

    return result_row

def get_result_files():
    # sort result filenames
    paridx_to_fpath = {}
    for fname in os.listdir("./results"):
        try:
            paridx = int(re.search("^\d+", fname).group())
        except:
            continue
        try:
            paridx_to_fpath[paridx].append("./results/"+fname)
        except:
            paridx_to_fpath[paridx] = ["./results/"+fname]
    return paridx_to_fpath

def get_operating_params(paridx_list):
    # Load operating params dataframe
    operating_params_df = pd.read_csv("./data/operating_params_df.csv").set_index('paridx')
    # filter out paridx where we don't currently have result files yet
    operating_params_df = operating_params_df.loc[operating_params_df.index.isin(paridx_list)]
    # Get tunable parameter ranges
    baseline_trans_multiplier_arr = np.sort(np.around(operating_params_df['baseline_trans_multiplier'].unique().astype(np.float64), 2))
    baseline_trans_multiplier_arr = baseline_trans_multiplier_arr.tolist()
    sus_per_inv_arr = np.sort(operating_params_df['sus_per_inv'].unique().astype(np.int64)).tolist()

    test_sens_arr = np.sort(100*np.around(operating_params_df['test_sens'].unique(), 2)).astype(np.int64)
    test_sens_arr = test_sens_arr[test_sens_arr>=0]
    test_sens_arr = test_sens_arr.tolist()

    test_spec_arr = np.sort(100*np.around(operating_params_df['test_spec'].unique(), 2)).astype(np.int64)
    test_spec_arr = test_spec_arr[test_spec_arr>=0]
    test_spec_arr = test_spec_arr.tolist()

    test_receptiveness_arr = np.sort(100*np.around(operating_params_df['test_receptiveness'].unique(), 2)).astype(np.int64)
    test_receptiveness_arr = test_receptiveness_arr[test_receptiveness_arr>=0]
    test_receptiveness_arr = test_receptiveness_arr.tolist()

    test_turnaround_time_arr = np.sort(operating_params_df['test_turnaround_time'].unique().astype(np.int64))
    test_turnaround_time_arr = test_turnaround_time_arr[test_turnaround_time_arr>=0]
    test_turnaround_time_arr = test_turnaround_time_arr.tolist()

    react_vacc_threshold_arr = np.sort(operating_params_df['react_vacc_threshold'].unique().astype(np.int64))
    react_vacc_threshold_arr = react_vacc_threshold_arr[react_vacc_threshold_arr>=0]
    react_vacc_threshold_arr = react_vacc_threshold_arr.tolist()

    react_vacc_turnaround_time_arr = np.sort(operating_params_df['react_vacc_turnaround_time'].unique().astype(np.int64))
    react_vacc_turnaround_time_arr = react_vacc_turnaround_time_arr[react_vacc_turnaround_time_arr>=0]
    react_vacc_turnaround_time_arr = react_vacc_turnaround_time_arr.tolist()

    df = operating_params_df[['ini_vacc_prop', 'react_vacc_prop']].drop_duplicates()
    ini_react_vacc_arr = []
    for r, row in df.iterrows():
        ini_react_vacc_arr.append(tuple((100*row.to_numpy()).astype(np.int64)))

    return operating_params_df, baseline_trans_multiplier_arr, sus_per_inv_arr, test_sens_arr, test_spec_arr, test_receptiveness_arr, test_turnaround_time_arr, react_vacc_threshold_arr, react_vacc_turnaround_time_arr, ini_react_vacc_arr
