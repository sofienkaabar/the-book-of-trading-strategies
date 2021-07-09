
# Base parameters
expected_cost    = 0.5 * (lot / 10000) 
assets           = asset_list(1) 
window           = 1000

# Trading parameters   
horizon          = 'H1'

# Mass imports 
my_data = mass_import(0, horizon)

def signal(Data):

    # True Low Calculation
    for i in range(len(Data)):
        Data[i, 4] = min(Data[i, 2], Data[i - 2, 2])
    # True High Calculation
    for i in range(len(Data)):
        Data[i, 5] = max(Data[i, 1], Data[i - 2, 1])    
    # Bullish signal
    for i in range(len(Data)):
        if Data[i, 3] < Data[i - 1, 3] and Data[i, 3] > Data[i, 0] and Data[i, 2] < Data[i - 2, 4]:
            Data[i, 6] = 1
    # Bearish signal
    for i in range(len(Data)):
        if Data[i, 3] > Data[i - 1, 3] and Data[i, 3] < Data[i, 0] and Data[i, 1] > Data[i - 2, 5]:
            Data[i, 7] = -1    
        return Data


          
##############################################################################   1

my_data = adder(my_data, 10)
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