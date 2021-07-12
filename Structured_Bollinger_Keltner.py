
# Base parameters
expected_cost    = 0.2 * (lot / 10000) 
assets           = asset_list(1) 
window           = 1000

# Trading parameters   
horizon          = 'H1'

# Indicator / Strategy parameters
boll_lookback     = 20
standard_deviation      = 2
keltner_lookback    = 10
atr_lookback   = 10
multiplier    = 2

# Mass imports 
my_data = mass_import(0, horizon)


def signal(Data, close, upper_boll, lower_boll, upper_keltner, lower_keltner, buy, sell):
    
    Data = adder(Data, 10)
    
    for i in range(len(Data)):
        
        if Data[i, close] < Data[i, lower_boll] and Data[i, close] < Data[i, lower_keltner] and Data[i - 1, buy] == 0:
            Data[i, buy] = 1
            
        elif Data[i, close] > Data[i, upper_boll] and Data[i, close] > Data[i, upper_keltner] and Data[i - 1, sell] == 0:
            Data[i, sell] = -1
            
    return Data

    

##############################################################################   1

my_data = BollingerBands(my_data, boll_lookback, standard_deviation, 3, 4)
my_data = keltner_channel(my_data, keltner_lookback, atr_lookback, multiplier, 3, 6)
my_data = signal(my_data, 3, 4, 5, 6, 7, 8, 9) 

if sigchart == True:
    signal_chart_ohlc_color(my_data, assets[0], 3, 8, 9, window = 250)
    plt.plot(my_data[-250:, 4])
    plt.plot(my_data[-250:, 5])
    plt.plot(my_data[-250:, 6])
    plt.plot(my_data[-250:, 7])

my_data = deleter(my_data, 4, 2)
holding(my_data, 6, 7, 8, 9)
my_data_eq = equity_curve(my_data, 8, expected_cost, lot, investment)
performance(my_data_eq, 8, my_data, assets[0])  

plt.plot(my_data_eq[:, 3], linewidth = 1, label = assets[0])
plt.grid()
plt.legend()
plt.axhline(y = investment, color = 'black', linewidth = 1)