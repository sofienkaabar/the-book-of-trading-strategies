
# Base parameters
expected_cost    = 0.2 * (lot / 10000) 
assets           = asset_list(1) 
window           = 1000

# Trading parameters   
horizon          = 'H1'

# Indicator / Strategy parameters
rsi_lookback        = 3
ma_lookback         = 3
stochastic_lookback = 6
upper_barrier       = 80
lower_barrier       = 20

# Mass imports 
my_data = mass_import(0, horizon)

def signal(Data, rsi_column, stochastic_column, stochastic_ma_column, ma_column, buy, sell):
    
    Data = adder(Data, 10)
    
    for i in range(len(Data)):
        if Data[i, rsi_column] < lower_barrier and Data[i, stochastic_column] > Data[i, stochastic_ma_column] and Data[i - 1, stochastic_column] < Data[i - 1, stochastic_ma_column] and Data[i, 3] > Data[i, ma_column]:
               Data[i, buy] = 1
               
        elif Data[i, rsi_column] > upper_barrier and Data[i, stochastic_column] < Data[i, stochastic_ma_column] and Data[i - 1, stochastic_column] > Data[i - 1, stochastic_ma_column] and Data[i, 3] < Data[i, ma_column]:
            Data[i, sell] = -1   
            
    return Data

# The Data variable refers to the OHLC data with the calculated indicators
# The rsi_column variable is the index of the column where the RSI is stored
# The stochastic_column variable is the index of the column where the Stochastic Oscillator is stored
# The stochastic_ma_column variable is the index of the column where the Moving Average of the Stochastic Oscillator is stored
# The Closing price is indexed at 3


##############################################################################   1

my_data = rsi(my_data, rsi_lookback, 3, 4, genre = 'Smoothed')
my_data = stochastic(my_data, stochastic_lookback, 3, 5, genre = 'High-Low')
my_data = ma(my_data, ma_lookback, 5, 6)
my_data = ma(my_data, 150, 3, 7)
my_data = signal(my_data, 4, 5, 6, 7, 8, 9)
my_data = deleter(my_data, 6, 2)

if sigchart == True:
    signal_chart_ohlc_color(my_data, assets[0], 3, 6, 7, window = 250)

holding(my_data, 6, 7, 8, 9)
my_data_eq = equity_curve(my_data, 8, expected_cost, lot, investment)
performance(my_data_eq, 8, my_data, assets[0])  

plt.plot(my_data_eq[:, 3], linewidth = 1, label = assets[0])
plt.grid()
plt.legend()
plt.axhline(y = investment, color = 'black', linewidth = 1)

