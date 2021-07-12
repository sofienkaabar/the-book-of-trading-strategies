
# Base parameters
expected_cost    = 0.2 * (lot / 10000) 
assets           = asset_list(1) 
window           = 1000

# Trading parameters   
horizon          = 'H1'

# Indicator / Strategy parameters
ma_lookback      = 100

# Mass imports 
my_data = mass_import(0, horizon)


def signal(Data, close, psar, hull_ma, buy, sell):
    
    Data = adder(Data, 10)
    
    for i in range(len(Data)):
        if Data[i, close] > Data[i, psar] and Data[i, close] > Data[i, hull_ma] and Data[i - 1, close] < Data[i - 1, hull_ma]:
                Data[i, buy] = 1
         
        elif Data[i, close] < Data[i, psar] and Data[i, close] < Data[i, hull_ma] and Data[i - 1, close] > Data[i - 1, hull_ma]:
                Data[i, sell] = -1
                
    return Data
    

##############################################################################   1

# Converting to a pandas Data frame
my_data = pd.DataFrame(my_data)
# Renaming columns to fit the function
my_data.columns = ['open','high','low','close']
# Calculating the Parabolic SAR
Parabolic = sar(my_data, 0.02, 0.2)
# Converting the Parabolic values back to an array
Parabolic = np.array(Parabolic)
# Reshaping
Parabolic = np.reshape(Parabolic, (-1, 1))
# Concatenating with the OHLC Data
my_data = np.concatenate((my_data, Parabolic), axis = 1)
my_data = hull_moving_average(my_data, 3, ma_lookback, 5)
my_data = signal(my_data, 3, 4, 5, 6, 7) 

if sigchart == True:
    signal_chart_ohlc_color(my_data, assets[0], 3, 6, 7, window = 250)
    plt.plot(my_data[-250:, 4])
    plt.plot(my_data[-250:, 5])

holding(my_data, 6, 7, 8, 9)
my_data_eq = equity_curve(my_data, 8, expected_cost, lot, investment)
performance(my_data_eq, 8, my_data, assets[0])  

plt.plot(my_data_eq[:, 3], linewidth = 1, label = assets[0])
plt.grid()
plt.legend()
plt.axhline(y = investment, color = 'black', linewidth = 1)