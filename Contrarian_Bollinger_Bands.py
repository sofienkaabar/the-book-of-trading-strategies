
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

my_data = BollingerBands(my_data, lookback, standard_deviation, 3, 4)
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