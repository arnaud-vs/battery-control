import matplotlib.pyplot as plt
import pandas as pd
import time
from ip_results import load_results_parquet
from pathlib import Path

# Plotting functions
def plot_trading_decisions(results, labels, prefixes, forecast, real_prices, start, end, probabilistic, quantiles, alphas):
    forecast = forecast.loc[start:end]
    real_prices = real_prices.loc[start:end]

    fig, axs = plt.subplots(nrows=4, ncols=len(results), sharex='all', sharey='row')
    for i, r in enumerate(results):
        df, metadata = r
        df = df.loc[start:end]
        battery = metadata['battery']
        P = ( (df[f'{prefixes[i]}.e_ch_kwh']/0.25 / battery['p_charge_kw_max'])
              - (df[f'{prefixes[i]}.e_dis_kwh']/0.25 / battery['p_discharge_kw_max']) ) * 100
        soc = df[f'{prefixes[i]}.E_start_kwh'] / battery['energy_kwh'] * 100

        axs[0, i].grid(alpha=0.5)
        axs[0, i].step(real_prices.index, real_prices, where='post', label='Real imbalance price', color='C1')
        if probabilistic[i]:
            for q in range(len(quantiles)):
                lower = forecast[f'q{quantiles[q]:.2f}_qh1']
                upper = forecast[f'q{1 - quantiles[q]:.2f}_qh1']
                axs[0, i].fill_between(forecast.index, lower, upper, step='post', color='C0', alpha=alphas[q])
        axs[0, i].step(forecast.index, forecast['q0.50_qh1'], where='post', label='Forecast QH1', color='C0')
        axs[0, i].legend()
        axs[0, i].set_ylabel('Imbalance price [€/MWh]')
        axs[0, i].set_title(labels[i])
        axs[0, i].tick_params(labelleft=True)

        axs[1, i].grid(alpha=0.5)
        axs[1, i].bar(P.index[:-1], P.iloc[:-1], width=pd.Timedelta(minutes=15), align='edge', label='Battery setpoint', alpha=0.5, color='C0')
        axs[1, i].axhline(100, linestyle='--', color='C0', label='Power limits')
        axs[1, i].axhline(-100, linestyle='--', color='C0')
        axs[1, i].legend()
        axs[1, i].set_ylabel('Power [%]')
        axs[1, i].tick_params(labelleft=True)

        axs[2, i].grid(alpha=0.5)
        axs[2, i].plot(soc, label='Battery SoC', color='C0')
        axs[2, i].axhline(battery['soc_max']*100, linestyle='--', color='C0', label='SoC limits')
        axs[2, i].axhline(battery['soc_min']*100, linestyle='--', color='C0')
        axs[2, i].legend()
        axs[2, i].set_ylabel('State of Charge [%]')
        axs[2, i].tick_params(labelleft=True)

        axs[3, i].grid(alpha=0.5)
        axs[3, i].plot(df[f'{prefixes[i]}.profit_realized_eur'].cumsum(), color='C0')
        axs[3, i].set_ylabel('Cumulative profit [€]')
        axs[3, i].tick_params(labelleft=True)

    fig.suptitle('Battery decisions with model predictive control')
    fig.supxlabel('Time')
    plt.show()


def plot_cumulative_results(results, labels, prefixes):
    rbc = pd.read_csv('../../results/rbc/rbc.csv', index_col='datetime')
    rbc.index = pd.to_datetime(rbc.index)

    fig, axs = plt.subplots(1, 2, sharex='all')
    for i, r in enumerate(results):
        df, metadata = r

        axs[0].plot(df[f'{prefixes[i]}.profit_realized_eur'].cumsum(), label=labels[i])
        axs[0].plot(rbc['Profit'].cumsum(), label='Rule-based control')
        axs[0].legend()
        axs[0].set_ylabel('Cumulative profit [€]')
        axs[0].set_title('Cumulative profit')

        axs[1].plot((df[f'{prefixes[i]}.e_ch_kwh']+df[f'{prefixes[i]}.e_ch_kwh']).cumsum(), label=labels[i])
        axs[1].plot(1000*rbc['Volume'].abs().cumsum(), label='Rule-based control')
        axs[1].legend()
        axs[1].set_ylabel('Cumulative traded volume [kWh]')
        axs[1].set_title('Cumulative traded volume')
    fig.supxlabel('Time')
    plt.show()


def plot_profit_as_function_of_labels(results, labels, prefixes):
    tot_profits = []
    for i, r in enumerate(results):
        df, _ = r
        tot_profits += [df[f'{prefixes[i]}.profit_realized_eur'].sum()]
    plt.plot(labels, tot_profits)
    plt.xlabel('Case')
    plt.ylabel('Profit [€]')
    plt.show()


def plot_daily_profit_over_time(results, labels, prefixes):
    for i, r in enumerate(results):
        df, metadata = r
        daily_profit = df[f'{prefixes[i]}.profit_realized_eur'].groupby(pd.Grouper(freq='W')).sum()
        plt.plot(daily_profit, label=labels[i])
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Profit [€]')
    plt.title('Profit per day')
    plt.grid(alpha=0.5)
    plt.show()


def plot_trades_histogram(results, labels, prefixes):
    fig, axs = plt.subplots(len(results), 1, sharex='all', sharey='all')
    for i, r in enumerate(results):
        df, metadata = r
        daily_profit = df[f'{prefixes[i]}.profit_realized_eur'].groupby(pd.Grouper(freq='H')).sum()
        axs[i].hist(x=df[f'{prefixes[i]}.profit_realized_eur']) #, bins=range(-1000, 2000, 100))
        axs[i].grid(alpha=0.5)
        axs[i].set_title(labels[i])
    fig.supxlabel('Daily profit [€]')
    fig.supylabel('Days [-]')
    fig.suptitle('Daily profit histogram')
    plt.show()


# Specify paths to compare, and their labels for plotting
paths_to_compare = [
                    # 'ip_rolling_ce/ip_rolling_ce_qr_deterministic_forecast_TP_L1_0.01_CP_0__v1.parquet',
                    # 'ip_rolling_ce/ip_rolling_ce_qr_deterministic_forecast_TP_L1_0.03_CP_0__v1.parquet',
                    'ip_rolling_ce/ip_rolling_ce_qr_deterministic_forecast_TP_L1_0.1_CP_0__v1.parquet',
                    # 'ip_rolling_ce/ip_rolling_ce_qr_deterministic_forecast_TP_L1_0.3_CP_0__v1.parquet',
                    # 'ip_rolling_ce/ip_rolling_ce_qr_deterministic_forecast_TP_L2_0.001_CP_0__v1.parquet',
                    # 'ip_rolling_ce/ip_rolling_ce_qr_deterministic_forecast_TP_L2_0.003_CP_0__v1.parquet',
                    # 'ip_rolling_ce/ip_rolling_ce_qr_deterministic_forecast_TP_L2_0.01_CP_0__v1.parquet',
                    # 'ip_rolling_ce/ip_rolling_ce_qr_deterministic_forecast_TP_L2_0.03_CP_0__v1.parquet',
                    # 'ip_rolling_ce/ip_rolling_ce_qr_deterministic_perfect_foresight_TP_L1_0.1_CP_0__v1.parquet',
                    # 'ip_rolling_ce/ip_rolling_ce_qr_deterministic_perfect_foresight_TP_L1_0.01_CP_0__v1.parquet',
                    # 'ip_rolling_ce/ip_rolling_ce_qr_deterministic_perfect_foresight_TP_L1_0.3_CP_0__v1.parquet',
                    # 'ip_rolling_ce/ip_rolling_ce_qr_deterministic_perfect_foresight_TP_L1_0.03_CP_0__v1.parquet',
                    # 'ip_rolling_ce/ip_rolling_ce_qr_deterministic_perfect_foresight_TP_L2_0.01_CP_0__v1.parquet',
                    # 'ip_rolling_ce/ip_rolling_ce_qr_deterministic_perfect_foresight_TP_L2_0.001_CP_0__v1.parquet',
                    # 'ip_rolling_ce/ip_rolling_ce_qr_deterministic_perfect_foresight_TP_L2_0.03_CP_0__v1.parquet',
                    # 'ip_rolling_ce/ip_rolling_ce_qr_deterministic_perfect_foresight_TP_L2_0.003_CP_0__v1.parquet'
                    'ip_rolling_prob/ip_rolling_prob_qr_quantile_paths_cvar_a0p95_l0p1_TP_L1_0.1_CP_0_v1.parquet'
                    # 'ip_rolling_prob/ip_rolling_prob_qr_quantile_paths_cvar_a0p95_l0p3_TP_L1_0.1_CP_0_v1.parquet'
                    ]

# labels = ['QR forecast', 'Perfect foresight']
# labels = ['L1 0.01', 'L2 0.01']
# labels = ['Deterministic forecast', 'Perfect foresight', 'Probabilistic forecast']
labels = ['Deterministic forecast', 'Probabilistic forecast']
# labels = ['L1\n0.01', 'L1\n0.03', 'L1\n0.1', 'L1\n0.3', 'L2\n0.001', 'L2\n0.003', 'L2\n0.01', 'L2\n0.03']
# prefixes = ['qr.forecast', 'benchmark.perfect_foresight']
# prefixes = ['qr.forecast']*2
# prefixes = ['qr.forecast', 'benchmark.perfect_foresight', 'qr.prob.cvar_a0p95_l0p1']
prefixes = ['qr.forecast', 'qr.prob.cvar_a0p95_l0p1']
# prefixes = ['qr.forecast']*8
probabilistic = [0, 1]
forecast_path = 'IP_QR.csv'

# Load the desired results
results = []
results_folder_path = '../../results/'
print('Loading results...')
tic = time.time()
for p in paths_to_compare:
    result_path = Path(results_folder_path+p)
    result = load_results_parquet(result_path) # result is a tuple: (df, metadata)
    results += [result]
toc = time.time()
print(f'Done in {round(toc-tic, 1)} seconds.')

# Load imbalance price forecasts
forecast_folder_path = '../../Data/IP_CET/'
forecast_path = Path(forecast_folder_path+forecast_path)
real_prices_path = Path(forecast_folder_path+'IP_Real_Prices.csv')
forecast = pd.read_csv(forecast_path, index_col='Date')
real_prices = pd.read_csv(real_prices_path, index_col='Date')
forecast.index = pd.to_datetime(forecast.index)
real_prices.index = pd.to_datetime(real_prices.index)

# Make your favorite plots
start = pd.Timestamp(year=2023, month=1, day=25, hour=16)
end = pd.Timestamp(year=2023, month=1, day=25, hour=22)
quantiles = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4]
alphas = [0.01, 0.05, 0.1, 0.2, 0.4, 0.6]
# plot_trading_decisions(results, labels, prefixes, forecast, real_prices, start, end, probabilistic, quantiles, alphas)

plot_cumulative_results(results, labels, prefixes)

# plot_profit_as_function_of_labels(results, labels, prefixes)

# plot_daily_profit_over_time(results, labels, prefixes)

# plot_trades_histogram(results, labels, prefixes)