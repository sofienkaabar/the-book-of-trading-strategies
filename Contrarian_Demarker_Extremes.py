
# Base parameters
expected_cost    = 0.0 * (lot / 100000) 
assets           = asset_list(1) 
window           = 1000

# Trading parameters   
horizon          = 'H1'

# Indicator / Strategy parameters
lookback         = 14
upper_barrier    = 0.70
lower_barrier    = 0.30

# Mass imports 
my_data = mass_import(0, horizon)

def signal(Data, what, buy, sell):
    
    Data = adder(Data, 10)
    Data = rounding(Data, 5)
    
    for i in range(len(Data)):
            
        if Data[i, what] <= lower_barrier and Data[i - 1, buy] == 0 and \
            Data[i - 2, buy] == 0 and Data[i - 3, buy] == 0 and Data[i - 4, buy] == 0:
            Data[i, buy] = 1
            
        elif Data[i, what] >= upper_barrier and Data[i - 1, sell] == 0 and \
            Data[i - 2, sell] == 0 and Data[i - 3, sell] == 0 and Data[i - 4, sell] == 0:
            Data[i, sell] = -1    
            
    return Data  

##############################################################################   1

my_data = demarker(my_data, lookback, 1, 2, 4)
my_data = signal(my_data, 4, 6, 7)

holding(my_data, 6, 7, 8, 9)
my_data_eq = equity_curve(my_data, 8, expected_cost, lot, investment)
performance(my_data_eq, 8, my_data, assets[0])

if sigchart == True:
    signal_chart_ohlc_color(my_data, assets[0], 3, 6, 7, window = 500)
    indicator_plot_double(my_data, 0, 1, 2, 3, 4, window = 250)
    plt.axhline(y = upper_barrier, color = 'black', linewidth = 1, linestyle = '--')
    plt.axhline(y = lower_barrier, color = 'black', linewidth = 1, linestyle = '--')    

plt.plot(my_data_eq[:, 3], linewidth = 1, label = assets[0])
plt.grid()
plt.legend()
plt.axhline(y = investment, color = 'black', linewidth = 1)