import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('./Data/data_ip_min_23.csv', index_col='datetime')
df.index = pd.to_datetime(df.index)
ip_min = df['ip']
df_qh = pd.read_csv('./Data/data_ip_qh_23.csv', index_col='datetime')
df_qh.index = pd.to_datetime(df_qh.index)
ip_qh = df_qh['ip']

E = 4
P = 4
quantile_window = pd.Timedelta(days=3*30)
start = pd.Timestamp(year=2023, month=1, day=1, hour=0, tz='utc')
end = pd.Timestamp(year=2023, month=12, day=31, hour=23, tz='utc')
ip_min = ip_min.loc[start:end]

# actions: (SoC lower, SoC upper, price quantile lower, price quantile upper, action)
charge, wait, discharge = 1, 0, -1
actions = [(0,  5,   0,    1,    charge),
           (5,  30,  0,    0.4,  charge),
           (30, 60,  0,    0.25, charge),
           (60, 95,  0,    0.1,  charge),
           (5,  40,  0.9,  1,    discharge),
           (40, 70,  0.8,  1,    discharge),
           (70, 95,  0.7,  1,    discharge),
           (95, 100, 0,    1,    discharge)]

state = pd.DataFrame(index=ip_min.index, columns=['SoC', 'Profit', 'Volume', 'Q'])
state['month'] = state.index.month
soc = 50
month = start.month
for min in state.index:
    qh = min.floor('15min')
    ip_min_window = ip_qh.loc[qh - quantile_window:qh]
    q = sum(ip_min_window <= ip_min.loc[min]) / len(ip_min_window)

    decision = wait
    for a in actions:
        if (soc >= a[0]) & (soc < a[1]):
            if (q >= a[2]) & (q < a[3]):
                decision = a[4]

    volume = decision*P/60
    profit = -ip_qh.loc[qh]*volume
    soc_mwh = soc*E/100 + volume
    soc = soc_mwh*100 / E

    state.at[min, 'Q'] = q
    state.at[min, 'Volume'] = volume
    state.at[min, 'Profit'] = profit
    state.at[min, 'SoC'] = soc

    if min.month != month:
        print(f'Month {month} done')
        month = min.month

cycle = 2*E
n_days = len(state) / (24*60)
daily_cycle = state['Volume'].abs().sum() / (cycle*n_days)
print(f'On average, battery performs {daily_cycle} cycles per day.')

op_profit = state['Profit'].sum() / (P*len(state)/15)
print(f'On average, battery yields an operational profit of {op_profit} €/MW/QH.')

plt.hist(state['Q'])
plt.xlabel('Quantile')
plt.ylabel('Counts')
plt.show()

plt.hist(state['SoC'])
plt.xlabel('State of charge')
plt.ylabel('Counts')
plt.show()

plt.plot(state.groupby(pd.Grouper(freq='ME'))['Profit'].sum(), marker='*')
plt.xlabel('Month')
plt.ylabel('Revenue')
plt.show()















