import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime as dt
import os
import pulp
from dataclasses import dataclass, field
from epftoolbox.evaluation import MAE


plt.style.use('seaborn-v0_8')

# Battery specs
@dataclass
class Battery:
    rated_capacity: float = field()
    rated_power: float = field()
    initial_soc: float = field(default=0.5)

    def __post_init__(self):
        if self.rated_capacity <= 0:
            raise ValueError("Rated capacity must be positive.")
        if self.rated_power <= 0:
            raise ValueError("Rated power must be positive.")

    @property
    def initial_charge(self) -> float:
        return self.rated_capacity * self.initial_soc

# Create battery object
battery = Battery(rated_capacity=4, rated_power=4)

# Function for plotting optimization results
def plot_results(battery_power: np.array, price: np.array, forecast: pd.DataFrame, battery: Battery):
    x = range(len(price))
    x_battery = range(len(price)+1)

    battery_load = battery_power * -1.0
    profit = battery_power * price * 0.25

    battery_charge = np.cumsum(np.insert(0.25*battery_load, 0, values=[battery.initial_charge]))

    fc = forecast.resample('h').first()
    fc = pd.concat([fc.loc[q] for q in fc.index], axis=0)
    if len(forecast.columns) > 4:
        fc = fc.drop(index=['qh5', 'qh6', 'qh7', 'qh8'])
    fc = fc.values

    # Create a 2x2 grid of subplots with shared x-axis for columns
    fig, axs = plt.subplots(2, 2, figsize=(16, 8), sharex="col")

    # Plot on the first subplot
    ax1 = axs[0, 0]
    ax1.bar(x, battery_load, label="Battery", alpha=0.5)
    ax1.plot(x, np.full_like(x, battery.rated_power), "--", lw=1, color='C1', label="Rated power")
    ax1.plot(x, np.full_like(x, -battery.rated_power), "--", lw=1, color='C1')
    ax1.set_ylabel("Power [MW]")
    ax1.legend(loc='upper left')
    ax1.set_axisbelow(True)

    # Create a twin y-axis sharing the same x-axis
    ax1_twin = ax1.twinx()
    ax1_twin.plot(x, fc, label="Forecast", lw=2)
    ax1_twin.set_ylabel("Imbalance price [€/MWh]")
    ax1_twin.legend(loc='upper right')
    ax1_twin.grid(False)

    # Plot on the second subplot
    ax2 = axs[0, 1]
    ax2.bar(x, battery_load, label="Power")
    ax2.set_ylabel("Power [MW]")
    ax2.legend(loc='upper left')

    # Create a twin y-axis sharing the same x-axis
    ax2_twin = ax2.twinx()
    ax2_twin.plot(x_battery, battery_charge, color='C1', label="SoC")
    ax2_twin.plot(x_battery, np.full_like(x_battery, battery.rated_capacity), "--", lw=1, color="C1", label="Rated capacity")
    ax2_twin.plot(x_battery, np.full_like(x_battery, 0), "--", lw=1, color="C1")
    ax2_twin.set_ylabel("Energy [MWh]")
    ax2_twin.legend(loc='upper right')
    ax2_twin.grid(False)

    # Plot on the third subplot with dual y-axis
    ax3 = axs[1, 0]
    ax3.plot(x, price, label="Price")
    ax3.set_ylabel("Price [€/MWh]")
    ax3.legend(loc='upper left')

    # Create a twin y-axis sharing the same x-axis
    ax3_twin = ax3.twinx()
    ax3_twin.bar(x, profit, label="Profit", color="C1", alpha=0.5)
    ax3_twin.plot(x, np.cumsum(profit), label="Cumulative profit", color="C1")
    ax3_twin.set_ylabel("Profit [€]")
    ax3_twin.legend(loc='upper right')
    ax3_twin.grid(False)

    # Plot on the fourth subplot
    # ax4 = axs[1, 1]
    # ax4.plot(x, price, label="Price")
    # for q in range(len(forecast.columns)):
    #     ax4.plot(x[q:], forecast.iloc[q].values[:len(forecast.columns)-q], label=f"Forecast QH{q+1}")
    # ax4.set_ylabel("Price [€/MWh]")
    # ax4.legend(loc='upper left')
    #
    # # Set x-axis label for the bottom subplots
    # axs[1, 0].set_xlabel("QH")
    # axs[1, 1].set_xlabel("QH")

    # Adjust layout
    plt.tight_layout()

    # Show plot
    plt.show()

### Load data

# Define input parameters to create unique file name
nlayers            = 2               # Number of layers in DNN
dataset            = 'basic_new'     # Name of csv file with input data
horizon            = 2               # Horizon in hours
lookback           = 1               # Lookback in hours
shuffle_train      = 0               # Boolean that selects whether the validation and training datasets are shuffled
data_augmentation  = 0               # Boolean that selects whether a data augmentation technique for DNNs is used
# calibration_window = 22+118*24         # Number of hours used in the training dataset for recalibration (change to 0 to include in hyperopt?)
calibration_window = 24*7*52
recalibration_freq = 24*30           # Number of hours between successive recalibrations of the forecasting model
experiment_id      = 1               # Unique identifier to read the trials file of hyperparameter optimization

# Load real imbalance price values
# real_values = pd.read_csv('results_forecast/real_values_elia_period_hz2.csv')
real_values = pd.read_csv('results_forecast/real_values_hz2.csv')
real_values = real_values.set_index('Unnamed: 0')
real_values.index = pd.to_datetime(real_values.index)

# Load ensemble forecast
path_recalibration_folder = "./results_forecast/"
forecast_file_name = 'ensembleTOT_forecast_NL' + str(nlayers) + '_DATA' + str(dataset) + \
                     '_SF' + str(shuffle_train) + '_DA' * data_augmentation + \
                     '_CW' + str(calibration_window) + '_RF' + str(recalibration_freq) + \
                     '_LB' + str(lookback) + '_HZ' + str(horizon) + '_' + str(experiment_id) + '.csv'
forecast_file_path = os.path.join(path_recalibration_folder, forecast_file_name)
forecast = pd.read_csv(forecast_file_path)
forecast = forecast.set_index('Unnamed: 0')
forecast.index = pd.to_datetime(forecast.index)

# Naive persistence forecast (not used)
persistence = real_values['qh1'].shift(periods=1).iloc[1:]

# Construct pattern forecast (number of periods can be changed)
pers_hour = real_values.shift(periods=8).iloc[8:]

# Load Elia forecast (not used)
forecast_file_name = 'ImbalancePriceForecast_historical_11112024.xlsx'
path_recalibration_folder = 'Data'
forecast_file_path = os.path.join(path_recalibration_folder, forecast_file_name)
forecast_elia = pd.read_excel(forecast_file_path)
forecast_elia = forecast_elia.set_index('Forecasted QH (utc)')
forecast_elia = forecast_elia.iloc[:-3-4*horizon]

# Load SI
inputs = pd.read_csv('./ip_data/' + dataset + '.csv', index_col='Unnamed: 0', usecols=['Unnamed: 0', 'si'])
inputs.index = pd.to_datetime(inputs.index).tz_localize(None)
inputs = inputs.loc[real_values.index]
si = pd.Series(inputs['si'])

# Drop first QHs because pattern forecast does not have a prediction
real_values = real_values.iloc[8:]
forecast = forecast.iloc[8:]
# da_fc = da_fc.iloc[4:]
forecast_elia = forecast_elia.iloc[8:]
si = si.iloc[8:]
persistence = persistence[3:]

# Define period of interest
# begin_test_date    = pd.to_datetime('17/9/2024 23:00', dayfirst=True)
# end_test_date      = pd.to_datetime('11/11/2024 18:15', dayfirst=True)
# begin_test_date    = pd.to_datetime('18/9/2024 6:00', dayfirst=True)
# end_test_date      = pd.to_datetime('18/9/2024 11:45', dayfirst=True)
# begin_test_date    = pd.to_datetime('18/9/2024 20:00', dayfirst=True)
# end_test_date      = pd.to_datetime('18/9/2024 21:45', dayfirst=True)
begin_test_date    = pd.to_datetime('1/1/2023 2:00', dayfirst=True)
end_test_date      = pd.to_datetime('31/12/2023 21:45', dayfirst=True)
# end_test_date      = pd.to_datetime('1/1/2023 3:45', dayfirst=True)
forecast    = forecast.loc[begin_test_date:end_test_date]
real_values = real_values.loc[begin_test_date:end_test_date]
# naive_fc    = naive_fc.loc[begin_test_date:end_test_date]
pers_hour   = pers_hour.loc[begin_test_date:end_test_date]
si          = si.loc[begin_test_date:end_test_date]

# Optimization horizon
optim_hz = 8

# Length of the problem in QHs
qhs = len(real_values)
assert qhs % optim_hz == 0, 'Considered period cannot be split into separate hours'
hours = int(qhs / optim_hz)

# Forecast loop
# fc_list = [real_values, naive_fc, pers_hour, forecast]
# label_list = ['Perfect', 'Naive: hybrid', 'Naive: persistence', 'Ensemble']
fc_list = [real_values, pers_hour, forecast]
label_list = ['Perfect', 'Pattern', 'Ensemble']
# fc_list = [forecast]
# label_list = ['Ensemble']
profit_list = []
si_side_list = []
pos_energy_list = []
neg_energy_list = []
idx = 0
qh = real_values.index[idx]
for fc in fc_list:
    power_solution = np.array([])

    for h in range(hours):

        # Solve optimization every QH (receding horizon or MPC)
        for q in range(1, optim_hz+1):
            # Forecast at considered QH
            fc_hz = fc.iloc[idx+q-1].values[:optim_hz]

            # Create a linear programming problem (maximize profit)
            lp = pulp.LpProblem("battery_optimization", pulp.LpMaximize)

            # Decision variables
            p = pulp.LpVariable.dicts("power_setpoint", range(optim_hz), -battery.rated_power, battery.rated_power)
            soc = pulp.LpVariable.dicts("state_of_charge", range(optim_hz+1), 0, battery.rated_capacity)

            # Objective function: maximize profit
            lp += pulp.lpSum([p[t] * fc_hz[t-q+1] for t in range(q-1, optim_hz)]) # Positive power = discharging

            # Starting capacity constraint
            lp += soc[0] == battery.initial_charge

            # State of charge update constraints
            for t in range(optim_hz):
                # Positive power = discharging
                lp += soc[t+1] == soc[t] - p[t] * 0.25 # 0.25 because of quarter hour
                # Ensure SOC remains within capacity limits at each hour
                lp += soc[t+1] <= battery.rated_capacity
                lp += soc[t+1] >= 0

            # Fix power levels of previous QHs
            for q_past in range(1, q):
                lp += p[q_past-1] == power_solution[-q+q_past]

            # End capacity constraint
            lp += soc[optim_hz] == battery.initial_charge

            # Solve the problem
            lp.solve(pulp.PULP_CBC_CMD(msg=False))

            # Print result
            # print("Total profit: €", pulp.value(lp.objective))

            # Access solution values
            lp_solution = np.array([p[t].varValue for t in range(optim_hz)])
            power_solution = np.append(power_solution, lp_solution[q-1])

        # Step forward with value of optimization horizon
        if idx+optim_hz == len(fc.index):
            idx = 0 # Reset idx if end is reached
            qh = real_values.index[idx]
        else:
            idx += optim_hz
            qh = fc.index[idx]

    # Store profit
    profit = power_solution * real_values['qh1'].values * 0.25 # 0.25 because of quarter hour
    profit_list += [profit]

    # System imbalance mitigation
    # Positive sign of power = discharging, positive sign of SI = excess energy
    # ==> We want SI and power to be of opposite sign
    si_side_sgn = [np.sign(si.iloc[t] * (-power_solution[t])) for t in range(qhs) if power_solution[t] != 0]
    si_side_binary = [(si_side_sgn[t]+1)/2 for t in range(len(si_side_sgn))]
    si_side_list += [np.sum(si_side_binary)*100/len(si_side_binary)]

    # Total (dis)charged energy
    pos_energy = [-power_solution[t] * 0.25 if power_solution[t] <= 0 else 0 for t in range(qhs)]
    neg_energy = [power_solution[t] * 0.25 if power_solution[t] > 0 else 0 for t in range(qhs)]
    pos_energy_list += [np.sum(pos_energy)]
    neg_energy_list += [np.sum(neg_energy)]

    # Plot results
    # plot_results(battery_power=power_solution, price=real_values['qh1'].values, forecast=fc, battery=battery)

# Plot cumulative profit
fig, ax = plt.subplots()
for i in range(len(fc_list)):
    ax.plot(real_values.index, np.cumsum(profit_list[i])*100/np.sum(profit_list[0]), label=label_list[i], linewidth=2)
plt.legend(loc='upper left', fontsize=18)
plt.xlabel('Time', fontsize=18)
plt.ylabel('Relative cumulative profit [%]', fontsize=18)
dates = [begin_test_date+pd.Timedelta(days=31*i) for i in range(0, 12, 3)]
ax.set_xticks(dates+[pd.to_datetime('1/1/2024 00:00', dayfirst=True)])
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax.tick_params(axis='both', which='major', labelsize=18)
plt.show()

# Print some statistics
print('Operating profits:')
for i, label in enumerate(label_list):
    op_profit = profit_list[i].sum()/len(profit_list[i])/battery.rated_power
    print(label + ': ' + str(round(op_profit, 2)) + ' €/MW/QH')
print('\nSI mitigation:')
for i, label in enumerate(label_list):
    print(label + ': ' + str(round(si_side_list[i], 2)) + ' %')
print('\nCharged energy:')
for i, label in enumerate(label_list):
    print(label + ': ' + str(round(pos_energy_list[i], 2)) + ' MWh')
print('\nDischarged energy:')
for i, label in enumerate(label_list):
    print(label + ': ' + str(round(neg_energy_list[i], 2)) + ' MWh')
















