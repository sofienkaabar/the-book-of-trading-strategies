
# Requires the definition of Simple Moving Average & Exponential Moving Average

def atr(Data, lookback, high, low, close, where):
    
    # Adding the required columns
    Data = adder(Data, 1)
    
    # True Range Calculation
    for i in range(len(Data)):
        try:
            
            Data[i, where] =   max(Data[i, high] - Data[i, low],
                               abs(Data[i, high] - Data[i - 1, close]),
                               abs(Data[i, low] - Data[i - 1, close]))
            
        except ValueError:
            pass
        
    Data[0, where] = 0   
    
    # Average True Range Calculation
    Data = ema(Data, 2, lookback, where, where + 1)
    
    # Cleaning
    Data = deleter(Data, where, 1)
    Data = jump(Data, lookback)

    return Data

# Trading parameters   
horizon          = 'H1'

# Mass imports 
my_data = mass_import(0, horizon)
          
# Calculating the ATR
my_data = atr(my_data, 10, 1, 2, 3, 4)

if sigchart == True:
    indicator_plot_double(my_data, 3, 4, name = assets[2], name_ind = '', window = 250)    

