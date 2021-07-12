
# Base parameters
expected_cost    = 0.2 * (lot / 10000) 
assets           = asset_list(1) 
window           = 1000

# Trading parameters   
horizon          = 'H1'

# Indicator / Strategy parameters
upper_barrier = 2.24
lower_barrier = -2.24

# Mass imports 
my_data = mass_import(0, horizon)

def signal(Data, what, buy, sell):
    for i in range(len(Data)):
        if Data[i, what] < lower_barrier and Data[i - 1, what] > lower_barrier and Data[i - 2, what] > lower_barrier :
            Data[i, buy] = 1
        if Data[i, what] > upper_barrier and Data[i - 1, what] < upper_barrier and Data[i - 2, what] < upper_barrier :
            Data[i, sell] = -1    
    return Data

##############################################################################   1

where = 4
for i in range(3, 30):
    my_data = fisher_transform(my_data, i, 3, where)
    where = where + 1
my_data = adder(my_data, 1)
for i in range(len(my_data)):
    my_data[i, -1] = np.sum(my_data[i, 4:4+ 30 - 3])
    my_data[i, -1] = my_data[i, - 1] / (30 - 3)
my_data = deleter(my_data, 4, 27)
my_data = adder(my_data, 10)
my_data = signal(my_data, 4, 6, 7) 

if sigchart == True:
    signal_chart_ohlc_color(my_data, assets[0], 3, 6, 7, window = 250)
    indicator_plot_double(my_data, 0, 1, 2, 3, 4, window = 250)

holding(my_data, 6, 7, 8, 9)
my_data_eq = equity_curve(my_data, 8, expected_cost, lot, investment)
performance(my_data_eq, 8, my_data, assets[0])  

plt.plot(my_data_eq[:, 3], linewidth = 1, label = assets[0])
plt.grid()
plt.legend()
plt.axhline(y = investment, color = 'black', linewidth = 1)