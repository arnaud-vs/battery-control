import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np


def fan_plot_qh(fc, hz, real, quantiles, alphas, start, end, method):
    real = real.loc[start:end]
    fc = fc.loc[start:end]

    fig, axs = plt.subplots(2, 4, sharex='col', sharey='all')
    for qh in range(1, hz+1):
        ax = axs[(qh-1) // 4, (qh-1) % 4]
        ax.plot(real[f'qh{qh}'], label='Real', color='k', linewidth=2, alpha=0.8)
        for i in range(len(quantiles)):
            if method in ['LEAR', 'XGB']:
                lower = fc[f'qh{qh}_q{quantiles[i]}']
                upper = fc[f'qh{qh}_q{1 - quantiles[i]}']
            else:
                lower = fc[f'q{quantiles[i]}_qh{qh}']
                upper = fc[f'q{1 - quantiles[i]}_qh{qh}']
            ax.fill_between(fc.index, lower, upper, color='C0', alpha=alphas[i], linewidth=2)
        if method in ['LEAR', 'XGB']:
            ax.plot(fc[f'qh{qh}_q0.5'], color='C0', label='Forecast', linewidth=2)
        else:
            ax.plot(fc[f'q0.5_qh{qh}'], color='C0', label='Forecast', linewidth=2)
        ax.set_xlabel('Time', fontsize=12)
        # ax.tick_params(axis='both', which='major', labelsize=12)
        # fig.subplots_adjust(left=0.15, bottom=0.15)
        # ax.xticks(rotation=45)
        ax.set_title(f'QH{qh}')
        ax.legend()
    fig.suptitle(f'Forecasts with {method}')
    fig.supylabel('Imbalance price [€/MWh]')
    plt.show()


def fan_plot_horizon(fc, real, quantiles, alphas, start, method, ax):
    frame = []
    fc = fc.loc[start]
    real_start = real['qh1'].loc[start-pd.Timedelta(minutes=15)]
    day_start = start-pd.Timedelta(hours=start.hour, minutes=start.minute)
    # hz_labels = [f'QH{qh}' for qh in range(1, hz+1)]
    hz_labels = pd.date_range(start-pd.Timedelta(minutes=15), start+pd.Timedelta(minutes=15*(hz-1)), freq='15min')
    real_hist = real.loc[day_start:start-pd.Timedelta(minutes=15)]
    real_fut = real.loc[start-pd.Timedelta(minutes=15):day_start+pd.Timedelta(days=1)]
    frame += ax.plot(real_hist['qh1'], label='Past imbalance price', color='k', linewidth=2, alpha=0.8)
    frame += ax.plot(real_fut['qh1'], label='Future imbalance price', color='k', linewidth=2, alpha=0.4)
    for i in range(len(quantiles)):
        if method in ['LEAR', 'XGB']:
            lower = [real_start]+[fc[f'qh{qh}_q{quantiles[i]}'] for qh in range(1, hz+1)]
            upper = [real_start]+[fc[f'qh{qh}_q{1 - quantiles[i]}']  for qh in range(1, hz+1)]
        else:
            lower = [real_start]+[fc[f'q{quantiles[i]}_qh{qh}'] for qh in range(1, hz+1)]
            upper = [real_start]+[fc[f'q{1 - quantiles[i]}_qh{qh}'] for qh in range(1, hz+1)]
        frame += [ax.fill_between(hz_labels, lower, upper, color='C0', alpha=alphas[i], linewidth=2)]
    if method in ['LEAR', 'XGB']:
        frame += ax.plot(hz_labels, [real_start]+[fc[f'qh{qh}_q0.5'] for qh in range(1, hz+1)], color='C0', label='Probabilistic forecast', linewidth=2)
    else:
        frame += ax.plot(hz_labels, [real_start]+[fc[f'q0.5_qh{qh}'] for qh in range(1, hz+1)], color='C0', label='Probabilistic forecast', linewidth=2)
    ax.set_xlabel('January 2, 2023', fontsize=12)
    ax.legend()
    # ax.set_title(f'Forecasts with {method}')
    ax.set_ylabel('Imbalance price [€/MWh]')
    return frame, ax


def fan_plot_horizon_blocks(fc, real, quantiles, alphas, start, end, method):
    fc = fc.loc[start:end]
    real = real.loc[start:end]['qh1']

    plt.plot(real, label='Imbalance price', color='k', linewidth=2, alpha=0.8)
    start_range = pd.date_range(start, end, freq='2h')
    for s in start_range:
        # s_real = s - pd.Timedelta(minutes=15)
        real_start = real.loc[s]
        horizon = pd.date_range(s, s+pd.Timedelta(hours=2), freq='15min')
        for i in range(len(quantiles)):
            if method in ['LEAR', 'XGB']:
                lower = [real_start]+[fc.at[s, f'qh{qh}_q{quantiles[i]}'] for qh in range(1, hz+1)]
                upper = [real_start]+[fc.at[s, f'qh{qh}_q{1 - quantiles[i]}']  for qh in range(1, hz+1)]
            else:
                lower = [real_start]+[fc.at[s, f'q{quantiles[i]}_qh{qh}'] for qh in range(1, hz+1)]
                upper = [real_start]+[fc.at[s, f'q{1 - quantiles[i]}_qh{qh}'] for qh in range(1, hz+1)]
            plt.fill_between(horizon, lower, upper, color='C0', alpha=alphas[i], linewidth=2)
        if method in ['LEAR', 'XGB']:
            plt.plot(horizon, [real_start]+[fc.at[s, f'qh{qh}_q0.5'] for qh in range(1, hz+1)], color='C0', label='Probabilistic forecast', linewidth=2)
        else:
            plt.plot(horizon, [real_start]+[fc.at[s, f'q0.5_qh{qh}'] for qh in range(1, hz+1)], color='C0', label='Probabilistic forecast', linewidth=2)
    plt.xlabel('Time', fontsize=12)
    plt.legend()
    # ax.set_title(f'Forecasts with {method}')
    plt.ylabel('Imbalance price [€/MWh]')
    plt.show()


def pinball_loss(y_true, y_pred, quantile):
    """
    Compute the pinball (quantile) loss.

    For each observation:
      - if y_true >= y_pred, loss = quantile*(y_true - y_pred)
      - if y_true < y_pred,  loss = (1 - quantile)*(y_pred - y_true)
    """
    error = y_true - y_pred
    return np.mean(np.maximum(quantile * error, (quantile - 1) * error))


def interval_coverage(y_true, lower, upper):
    """
    Compute the coverage of the prediction interval.
    """
    return np.mean((y_true >= lower) & (y_true <= upper))


def interval_width(lower, upper):
    """
    Compute the average width of the prediction interval.
    """
    return np.mean(upper - lower)


def get_cols_from_q_qh(q, qh, method):
    if method in ['LEAR', 'XGB']:
        cols = f'qh{qh}_q{q}', f'qh{qh}_q{1-q}'
    else:
        cols = f'q{q}_qh{qh}', f'q{1-q}_qh{qh}'
    return cols


def calc_metrics(target, merged_df, quantiles, qhs, method, metrics):
    """
    Compute probabilistic metrics for QR / quantile forecasts.

    - Pinball loss for each available quantile column
    - CRPS (approx.) from the set of quantiles
    - Interval coverage & width for all symmetric intervals (tau, 1-tau) that exist:
        (0.1,0.9), (0.2,0.8), ..., (<0.5, >0.5)
    - If interval_lower/interval_upper are passed, also reports those specifically.
    """
    for q in quantiles:
        for qh in qhs:
            lower_col, upper_col = get_cols_from_q_qh(q, qh, method)
            lower_loss = pinball_loss(target[f'qh{qh}'], merged_df[lower_col], q)
            upper_loss = pinball_loss(target[f'qh{qh}'], merged_df[upper_col], 1-q)
            metrics.at['pinball_loss', f'{method} QH{qh} q{q}'] = lower_loss
            metrics.at['pinball_loss', f'{method} QH{qh} q{1-q}'] = upper_loss

            cov = interval_coverage(target[f'qh{qh}'], merged_df[lower_col], merged_df[upper_col])
            wid = interval_width(merged_df[lower_col], merged_df[upper_col])
            metrics.at['coverage', f'{method} QH{qh} {q}-{1-q}'] = cov
            metrics.at['width', f'{method} QH{qh} {q}-{1-q}'] = wid

    return metrics


def plot_pinball_loss(df_eval, quantiles, qhs, methods):
    fig, axs = plt.subplots(2, 5, sharex='all', sharey='all')
    for j, q in enumerate(quantiles):
        for i, m in enumerate(methods):
            df_per_qh = pd.DataFrame()
            ax = axs[j // 5, j % 5]
            for qh in qhs:
                df_per_qh.at[qh, 'pinball_loss'] = df_eval.at['pinball_loss', f'{m} QH{qh} q{q}']
            ax.plot(df_per_qh['pinball_loss'], color=f'C{i}', linewidth=2, label=m)
        ax.set_xlabel('QH')
        ax.set_title(f'q{q}')
        ax.legend()
    fig.suptitle('Pinball loss of imbalance price forecasts')
    fig.supylabel('Pinball loss [€/MWh]')
    plt.show()


def plot_coverage(df_eval, quantiles, qhs, methods):
    fig, axs = plt.subplots(1, 5, sharex='all', sharey='all')
    for j, q in enumerate(quantiles):
        for i, m in enumerate(methods):
            df_per_qh = pd.DataFrame()
            ax = axs[j]
            for qh in qhs:
                df_per_qh.at[qh, 'coverage'] = df_eval.at['coverage', f'{m} QH{qh} {q}-{1-q}']
            ax.plot(df_per_qh['coverage'], color=f'C{i}', linewidth=2, label=m)
        ax.set_xlabel('QH')
        ax.set_title(f'{q}-{1-q}')
        ax.legend()
    fig.suptitle('Coverage of imbalance price forecasts')
    fig.supylabel('Coverage [%]')
    plt.show()


def plot_width(df_eval, quantiles, qhs, methods):
    fig, axs = plt.subplots(1, 5, sharex='all', sharey='all')
    for j, q in enumerate(quantiles):
        for i, m in enumerate(methods):
            df_per_qh = pd.DataFrame()
            ax = axs[j]
            for qh in qhs:
                df_per_qh.at[qh, 'width'] = df_eval.at['width', f'{m} QH{qh} {q}-{1-q}']
            ax.plot(df_per_qh['width'], color=f'C{i}', linewidth=2, label=m)
        ax.set_xlabel('QH')
        ax.set_title(f'{q}-{1-q}')
        ax.legend()
    fig.suptitle('Width of imbalance price forecasts')
    fig.supylabel('Width [€/MWh]')
    plt.show()


data_path = './Data/Prob_Prices/IP_CET/IP_'

ip = pd.read_csv(data_path + 'Real_Prices.csv', index_col='Unnamed: 0')
ip.index = pd.to_datetime(ip.index)
start_ip = pd.Timestamp(year=2023, month=1, day=1, hour=0)
ip = ip.loc[start_ip:]

# plt.plot(ip['qh1'])
# plt.show()

start = pd.Timestamp(year=2023, month=1, day=1, hour=0)
end = pd.Timestamp(year=2023, month=1, day=14, hour=0)
hz = 8
quantiles = [0.1, 0.2, 0.3, 0.4] #, 0.5]
alphas = [0.1, 0.2, 0.4, 0.6]
# methods = ['LEAR', 'QR', 'XGB']
methods = ['QR']
# starts = [pd.Timestamp(year=2023, month=1, day=1, hour=12),
#           pd.Timestamp(year=2023, month=1, day=1, hour=14),
#           pd.Timestamp(year=2023, month=1, day=1, hour=16)]
starts = pd.date_range(pd.Timestamp(year=2023, month=3, day=25, hour=21, minute=30),
                       pd.Timestamp(year=2023, month=3, day=25, hour=23, minute=45),
                       freq='15min')
df_eval = pd.DataFrame()
for m in methods:
    fc = pd.read_csv(data_path + m + '.csv', index_col='Date')
    fc.index = pd.to_datetime(fc.index)

    # fan_plot_qh(fc, hz, ip, quantiles, alphas, start, end, m)

    fan_plot_horizon_blocks(fc, ip, quantiles, alphas, start, end, m)

    # fig, ax = plt.subplots()
    # fan_plot_horizon(fc, ip, quantiles, alphas, start, m, ax)
    # plt.show()

    # artists = []
    # fig, ax = plt.subplots()
    # for s in starts:
    #     frame, ax = fan_plot_horizon(fc, ip, quantiles, alphas, s, m, ax)
    #     artists += [frame]
    # ani = animation.ArtistAnimation(fig=fig, artists=artists, interval=1000)
    # plt.show()

#     qhs = range(1, hz+1)
#     df_eval = calc_metrics(ip, fc, quantiles, qhs, m, df_eval)
#
# plot_coverage(df_eval, quantiles, qhs, methods)
# plot_width(df_eval, quantiles, qhs, methods)
#
# quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6 ,0.7, 0.8, 0.9]
# plot_pinball_loss(df_eval, quantiles, qhs, methods)

# ani.save(filename="./videos/imbalance_price_QR.gif", writer="pillow")
