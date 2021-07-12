
# Base parameters
expected_cost    = 0.2 * (lot / 10000) 
assets           = asset_list(1) 
window           = 1000

# Trading parameters   
horizon          = 'H1'

# Indicator / Strategy parameters
lookback         = 10
multiplier       = 1
atr_lookback     = 10
rsi_lookback     = 2
upper_barrier = 75
lower_barrier = 25

# Mass imports 
my_data = mass_import(0, horizon)

def signal(Data, rsi_col, supertrend_col, buy, sell):
    
    Data = adder(Data, 10)
  
    for i in range(len(Data)):
        if Data[i, rsi_col] < lower_barrier and Data[i - 1, buy] == 0 and \
            Data[i - 2, buy] == 0 and Data[i - 3, buy] == 0 and Data[i - 4, buy] == 0 and Data[i, 2] > Data[i, supertrend_col]:
            Data[i, buy] = 1
            
        elif Data[i, rsi_col] > upper_barrier and Data[i - 1, sell] == 0 and \
            Data[i - 2, sell] == 0 and Data[i - 3, sell] == 0 and Data[i - 4, sell] == 0  and Data[i, 1] < Data[i, supertrend_col]:
            Data[i, sell] = -1    
            
    return Data


##############################################################################   1

my_data = atr(my_data, 10, 1, 2, 3, 4)
my_data = supertrend(my_data, multiplier, 4, 3, 1, 2, 5)
my_data = deleter(my_data, 4, 1)
my_data = rsi(my_data, rsi_lookback, 3, 5, genre = 'Smoothed')
my_data = signal(my_data, 5, 4, 6, 7) 

if sigchart == True:
    signal_chart_ohlc_color(my_data, assets[0], 3, 6, 7, window = 250)
    plt.plot(my_data[-250:, 4])

holding(my_data, 6, 7, 8, 9)
my_data_eq = equity_curve(my_data, 8, expected_cost, lot, investment)
performance(my_data_eq, 8, my_data, assets[0])  

plt.plot(my_data_eq[:, 3], linewidth = 1, label = assets[0])
plt.grid()
plt.legend()
plt.axhline(y = investment, color = 'black', linewidth = 1)