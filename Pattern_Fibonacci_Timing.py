
# Base parameters
expected_cost    = 0.5 * (lot / 10000) 
assets           = asset_list(1) 
window           = 1000

# Trading parameters   
horizon          = 'H1'

# Mass imports 
my_data = mass_import(0, horizon)

def signal(Data):

    # Bullish signal
    for i in range(len(Data)):
        if Data[i, 4] == -count:
            Data[i, 6] = 1
    # Bearish signal
    for i in range(len(Data)):
        if Data[i, 5] == count:
            Data[i, 7] = -1    
    return Data

count = 8
step = 5
step_two = 3
step_three = 2

          
##############################################################################   1

my_data = adder(my_data, 10)
my_data = fibonacci_timing_pattern(my_data, count, step, step_two, step_three, 3, 4, 5)
my_data = signal(my_data)

if sigchart == True:
    signal_chart_ohlc_color(my_data, assets[0], 3, 6, 7, window = 250)

holding(my_data, 6, 7, 8, 9)
my_data_eq = equity_curve(my_data, 8, expected_cost, lot, investment)
performance(my_data_eq, 8, my_data, assets[0])  

plt.plot(my_data_eq[:, 3], linewidth = 1, label = assets[0])
plt.grid()
plt.legend()
plt.axhline(y = investment, color = 'black', linewidth = 1)