# Base parameters
expected_cost    = 0.0 * (lot / 100000) 
assets           = asset_list(1) 
window           = 1000

# Trading parameters   
horizon          = 'H1'

# Indicator parameters
lookback         = 10
multiplier       = 3
atr_lookback     = 10

# Mass imports 
my_data = mass_import(0, horizon)

def signal(Data, close, super_trend_column, buy, sell):
    
    for i in range(len(Data)):
        if Data[i, close] > Data[i, super_trend_column] and Data[i - 1, close] < Data[i - 1, super_trend_column]:
            Data[i, buy] = 1
            
        elif Data[i, close] < Data[i, super_trend_column] and Data[i - 1, close] > Data[i - 1, super_trend_column]:
            Data[i, sell] = -1 
            
    return Data

##############################################################################   1

my_data = atr(my_data, lookback, 1, 2, 3, 4, genre = 'Smoothed')
my_data = supertrend(my_data, multiplier, 4, 3, 1, 2, 5)
my_data = adder(my_data, 10)
my_data = signal(my_data, 3, 5, 6, 7)

holding(my_data, 6, 7, 8, 9)
my_data_eq = equity_curve(my_data, 8, expected_cost, lot, investment)
performance(my_data_eq, 8, my_data, assets[0])

if sigchart == True:
    signal_chart_ohlc_color(my_data, assets[0], 3, 6, 7, window = 500)   
    plt.plot(my_data[-500:, 5])
    plt.axhline(y = upper_barrier, color = 'black', linewidth = 1, linestyle = '--')
    plt.axhline(y = lower_barrier, color = 'black', linewidth = 1, linestyle = '--')    

plt.plot(my_data_eq[:, 3], linewidth = 1, label = assets[0])
plt.grid()
plt.legend()
plt.axhline(y = investment, color = 'black', linewidth = 1)