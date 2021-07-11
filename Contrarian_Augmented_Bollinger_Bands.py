
def augmented_BollingerBands(Data, boll_lookback, standard_distance, high, low, where):
      
    Data = adder(Data, 10)
    
    # Calculating means
    Data = ema(Data, 2, boll_lookback, high, where)
    Data = ema(Data, 2, boll_lookback, low, where + 1)
    
    Data = volatility(Data, boll_lookback, high, where + 2)
    Data = volatility(Data, boll_lookback, low, where + 3)
    
    Data[:, where + 4] = Data[:, where] + (standard_distance * Data[:, where + 2])
    Data[:, where + 5] = Data[:, where + 1] - (standard_distance * Data[:, where + 3])
    
    Data = jump(Data, boll_lookback)
    
    Data = deleter(Data, where, 4)
        
    return Data

# Base parameters
expected_cost    = 0.0 * (lot / 100000) 
assets           = asset_list(1) 
window           = 1000

# Trading parameters   
horizon          = 'H1'

# Indicator parameters
lookback = 20
standard_deviation = 2

# Mass imports 
my_data = mass_import(0, horizon)

# Signal
def signal(Data, close, upper_boll, lower_boll, buy, sell):
    for i in range(len(Data)):
        if Data[i, close] < Data[i, lower_boll] and Data[i - 1, close] > Data[i - 1, lower_boll]:
            Data[i, buy] = 1
        if  Data[i, close] > Data[i, upper_boll] and Data[i - 1, close] < Data[i - 1, upper_boll]:
            Data[i, sell] = -1
    return Data

##############################################################################   1

my_data = augmented_BollingerBands(my_data, lookback, standard_deviation, 1, 2, 4)
my_data = adder(my_data, 10)
my_data = signal(my_data, 3, 4, 5, 6, 7)

holding(my_data, 6, 7, 8, 9)
my_data_eq = equity_curve(my_data, 8, expected_cost, lot, investment)
performance(my_data_eq, 8, my_data, assets[0])

if sigchart == True:
    signal_chart_ohlc_color(my_data, assets[0], 3, 6, 7, window = 500)   
    plt.plot(my_data[-500:, 4])    
    plt.plot(my_data[-500:, 5])    

plt.plot(my_data_eq[:, 3], linewidth = 1, label = assets[0])
plt.grid()
plt.legend()
plt.axhline(y = investment, color = 'black', linewidth = 1)