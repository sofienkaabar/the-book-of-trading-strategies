# Base parameters
expected_cost    = 0.0 * (lot / 100000) 
assets           = asset_list(1) 
window           = 1000

# Trading parameters   
horizon          = 'H1'

# Parameters
short_term_ma    = 50
medium_term_ma   = 100
long_term_ma     = 200
my_data = ma(my_data, short_term_ma,   3, 4)
my_data = ma(my_data, medium_term_ma,  3, 5)
my_data = ma(my_data, long_term_ma,    3, 6)

# Mass imports 
my_data = mass_import(0, horizon)

def signal(Data, short_term_ma, medium_term_ma, long_term_ma, buy, sell):
    Data = adder(Data, 10)
    
    for i in range(len(Data)):
        if Data[i, short_term_ma] > Data[i, medium_term_ma] and Data[i, short_term_ma] > Data[i, long_term_ma] and \
           Data[i - 1, short_term_ma] < Data[i - 1, medium_term_ma]:
            Data[i, buy] = 1
            
        elif Data[i, short_term_ma] < Data[i, medium_term_ma] and Data[i, short_term_ma] < Data[i, long_term_ma] and \
             Data[i - 1, short_term_ma] > Data[i - 1, medium_term_ma]:
            Data[i, sell] = -1  
            
    return Data  

##############################################################################   1

my_data = ema(my_data, 2, short_term_ma, 3, 4)
my_data = ema(my_data, 2, medium_term_ma, 3, 5)
my_data = ema(my_data, 2, long_term_ma, 3, 6)
my_data = adder(my_data, 10)
my_data = signal(my_data, 4, 5, 6, 7, 8)
my_data = deleter(my_data, 5, 1)

holding(my_data, 6, 7, 8, 9)
my_data_eq = equity_curve(my_data, 8, expected_cost, lot, investment)
performance(my_data_eq, 8, my_data, assets[0])

if sigchart == True:
    signal_chart_ohlc_color(my_data, assets[0], 3, 6, 7, window = 500)    
    plt.axhline(y = upper_barrier, color = 'black', linewidth = 1, linestyle = '--')
    plt.axhline(y = lower_barrier, color = 'black', linewidth = 1, linestyle = '--')    

plt.plot(my_data_eq[:, 3], linewidth = 1, label = assets[0])
plt.grid()
plt.legend()
plt.axhline(y = investment, color = 'black', linewidth = 1)