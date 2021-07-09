
# Base parameters
expected_cost    = 0.5 * (lot / 10000) 
assets           = asset_list(1) 
window           = 1000

# Trading parameters   
horizon          = 'H1'

# Mass imports 
my_data = mass_import(0, horizon)

def signal(Data, what, true_low, true_high, buy, sell, differential = 1):
    
    if differential == 1:
        for i in range(len(Data)):
            # True low
            Data[i, true_low] = min(Data[i, 2], Data[i - 1, what])
            Data[i, true_low] = Data[i, what] - Data[i, true_low]
            # True high  
            Data[i, true_high] = max(Data[i, 1], Data[i - 1, what])
            Data[i, true_high] = Data[i, what] - Data[i, true_high]
            
            # TD Differential
            if Data[i, what] < Data[i - 1, what] and Data[i - 1, what] < Data[i - 2, what] and Data[i, true_low] > Data[i - 1, true_low] and Data[i, true_high] < Data[i - 1, true_high]: 
                   Data[i, buy] = 1 
            if Data[i, what] > Data[i - 1, what] and Data[i - 1, what] > Data[i - 2, what] and Data[i, true_low] < Data[i - 1, true_low] and Data[i, true_high] > Data[i - 1, true_high]: 
                   Data[i, sell] = -1
                   
    if differential == 2:
        for i in range(len(Data)):
            # True low
            Data[i, true_low] = min(Data[i, 2], Data[i - 1, what])
            Data[i, true_low] = Data[i, what] - Data[i, true_low]
            # True high  
            Data[i, true_high] = max(Data[i, 1], Data[i - 1, what])
            Data[i, true_high] = Data[i, what] - Data[i, true_high]
            
            # TD Reverse Differential
            if Data[i, what] < Data[i - 1, what] and Data[i - 1, what] < Data[i - 2, what] and Data[i, true_low] < Data[i - 1, true_low] and Data[i, true_high] > Data[i - 1, true_high]: 
                   Data[i, buy] = 1 
            if Data[i, what] > Data[i - 1, what] and Data[i - 1, what] > Data[i - 2, what] and Data[i, true_low] > Data[i - 1, true_low] and Data[i, true_high] < Data[i - 1, true_high]: 
                   Data[i, sell] = -1   
                   
    if differential == 3: # TD Anti-Differential
        for i in range(len(Data)):
            if Data[i, what] < Data[i - 1, what] and Data[i - 1, what] > Data[i - 2, what] and Data[i - 2, what] < Data[i - 3, what] and Data[i - 3, what] < Data[i - 4, what]: 
                   Data[i, buy] = 1 
            if Data[i, what] > Data[i - 1, what] and Data[i - 1, what] < Data[i - 2, what] and Data[i - 2, what] > Data[i - 3, what] and Data[i - 3, what] > Data[i - 4, what]: 
                   Data[i, sell] = -1           

    return Data


          
##############################################################################   1

my_data = adder(my_data, 10)
my_data = signal(my_data, 3, 4, 5, 6, 7, 1)

if sigchart == True:
    signal_chart_ohlc_color(my_data, assets[0], 3, 6, 7, window = 250)

holding(my_data, 6, 7, 8, 9)
my_data_eq = equity_curve(my_data, 8, expected_cost, lot, investment)
performance(my_data_eq, 8, my_data, assets[0])  

plt.plot(my_data_eq[:, 3], linewidth = 1, label = assets[0])
plt.grid()
plt.legend()
plt.axhline(y = investment, color = 'black', linewidth = 1)