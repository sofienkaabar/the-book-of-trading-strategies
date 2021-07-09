
# Base parameters
expected_cost    = 0.5 * (lot / 10000) 
assets           = asset_list(1) 
window           = 1000

# Trading parameters   
horizon          = 'H1'

# Mass imports 
my_data = mass_import(0, horizon)

# Defining the minimum width of the first and third candles
side_body = 0.0010

# Defining the maximum width of the second candle
middle_body = 0.0003

# Signal function
def signal(Data):
    for i in range(len(Data)):        
       # Morning Star
       if Data[i - 2, 3] < Data[i - 2, 0] and (Data[i - 2, 0] - Data[i - 2, 3]) > side_body and (abs(Data[i - 1, 3] - Data[i - 1, 0])) <= middle_body and Data[i, 3] > Data[i, 0] and abs(Data[i, 3] - Data[i, 0]) > side_body:        
                Data[i, 6] = 1          
       # Evening Star
       if Data[i - 2, 3] > Data[i - 2, 0] and (Data[i - 2, 3] - Data[i - 2, 0]) > side_body and (abs(Data[i - 1, 3] - Data[i - 1, 0])) <= middle_body and Data[i, 3] < Data[i, 0] and abs(Data[i, 3] - Data[i, 0]) > side_body:
               Data[i, 7] = -1

##############################################################################   1

my_data = adder(my_data, 10)
my_data = rounding(my_data, 4)
signal(my_data)

if sigchart == True:
    signal_chart_ohlc_color(my_data, assets[0], 3, 6, 7, window = 500)

holding(my_data, 6, 7, 8, 9)
my_data_eq = equity_curve(my_data, 8, expected_cost, lot, investment)
performance(my_data_eq, 8, my_data, assets[0])  

plt.plot(my_data_eq[:, 3], linewidth = 1, label = assets[0])
plt.grid()
plt.legend()
plt.axhline(y = investment, color = 'black', linewidth = 1)