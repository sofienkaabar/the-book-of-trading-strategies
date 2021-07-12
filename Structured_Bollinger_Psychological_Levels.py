
# Base parameters
expected_cost    = 0.2 * (lot / 10000) 
assets           = asset_list(1) 
window           = 1000

# Trading parameters   
horizon          = 'H1'

# Indicator / Strategy parameters
boll_lookback     = 20
standard_deviation      = 2
trend = 8

# Mass imports 
my_data = mass_import(0, horizon)


def signal(Data, upper_bollinger_column, lower_bollinger_column, psychological_level_signal_column, buy, sell):
    Data = adder(Data, 10)
    for i in range(len(Data)):
        if Data[i, 3] <= Data[i, lower_bollinger_column] and Data[i, psychological_level_signal_column] == 1:
            Data[i, buy] = 1
        elif Data[i, 3] >= Data[i, upper_bollinger_column] and Data[i, psychological_level_signal_column] == 1:
            Data[i, sell] = -1
    return Data


    

##############################################################################   1

my_data = BollingerBands(my_data, boll_lookback, standard_deviation, 3, 4)
my_data = psychological_levels_scanner(my_data, trend, 6)
my_data = signal(my_data, 4, 5, 6, 7, 8)

if sigchart == True:
    signal_chart_ohlc_color(my_data, assets[0], 3, 7, 8, window = 250)
    plt.plot(my_data[-250:, 4])
    plt.plot(my_data[-250:, 5])

my_data = deleter(my_data, 4, 1)
holding(my_data, 6, 7, 8, 9)
my_data_eq = equity_curve(my_data, 8, expected_cost, lot, investment)
performance(my_data_eq, 8, my_data, assets[0])  

plt.plot(my_data_eq[:, 3], linewidth = 1, label = assets[0])
plt.grid()
plt.legend()
plt.axhline(y = investment, color = 'black', linewidth = 1)