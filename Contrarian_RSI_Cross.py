
# Base parameters
expected_cost    = 0.0 * (lot / 100000) 
assets           = asset_list(1) 
window           = 1000

# Trading parameters   
horizon          = 'H1'

# Indicator / Strategy parameters
lookback         = 14
upper_barrier    = 75
lower_barrier    = 25
ma_lookback      = 13

# Mass imports 
my_data = mass_import(0, horizon)

def signal(Data, rsi_col, ma_col, buy, sell):
    
    Data = adder(Data, 10)
    Data = rounding(Data, 5)
    
    for i in range(len(Data)):
        
        if Data[i, rsi_col] > Data[i, ma_col] and Data[i - 1, rsi_col] < Data[i - 1, ma_col]:
            
            Data[i, buy] = 1
            
        elif Data[i, rsi_col] < Data[i, ma_col] and Data[i - 1, rsi_col] > Data[i - 1, ma_col]:
            
            Data[i, sell] = -1    
    return Data


##############################################################################   1

my_data = rsi(my_data, lookback, 3, 4, genre = 'Smoothed')
my_data = ma(my_data, ma_lookback, 4, 5)
my_data = signal(my_data, 4, 5, 6, 7)

holding(my_data, 6, 7, 8, 9)
holding_fix(my_data)
risk_management_atr(my_data, stop, target, 4)
my_data_eq = equity_curve(my_data, expected_cost, lot, investment)
performance(my_data_eq, my_data, assets[0])

if sigchart == True:
    signal_chart_ohlc_color(my_data, assets[0], 3, 6, 7, window = 500)
    indicator_plot_double(my_data, 0, 1, 2, 3, 4, window = 250)
    plt.plot(my_data[-250:, 5])
    plt.axhline(y = upper_barrier, color = 'black', linewidth = 1, linestyle = '--')
    plt.axhline(y = lower_barrier, color = 'black', linewidth = 1, linestyle = '--')    

plt.plot(my_data_eq[:, 3], linewidth = 1, label = assets[0])
plt.grid()
plt.legend()
plt.axhline(y = investment, color = 'black', linewidth = 1)