
# Base parameters
expected_cost    = 0.0 * (lot / 100000) 
assets           = asset_list(1) 
window           = 1000

# Trading parameters   
horizon          = 'H1'

# Indicator / Strategy parameters
lookback         = 14
upper_barrier    = 75
lower_barrier    = 25
width            = 20

# Mass imports 
my_data = mass_import(0, horizon)

def divergence(Data, indicator, lower_barrier, upper_barrier, width, buy, sell):
    
    Data = adder(Data, 10)
    
    for i in range(len(Data)):
        
        try:
            if Data[i, indicator] < lower_barrier:
                
                for a in range(i + 1, i + width):
                    
                    # First trough
                    if Data[a, indicator] > lower_barrier:
                        
                        for r in range(a + 1, a + width):
                            
                            if Data[r, indicator] < lower_barrier and \
                            Data[r, indicator] > Data[i, indicator] and Data[r, 3] < Data[i, 3]:
                                
                                for s in range(r + 1, r + width):
                                    
                                    # Second trough
                                    if Data[s, indicator] > lower_barrier:
                                        Data[s, buy] = 1
                                        break
                                    
                                    else:
                                        break
                            else:
                                break
                        else:
                            break
                    else:
                        break
                    
        except IndexError:
            pass
      
    for i in range(len(Data)):
        
        try:
            if Data[i, indicator] > upper_barrier:
                
                for a in range(i + 1, i + width):
                    
                    # First trough
                    if Data[a, indicator] < upper_barrier:
                        for r in range(a + 1, a + width):
                            if Data[r, indicator] > upper_barrier and \
                            Data[r, indicator] < Data[i, indicator] and Data[r, 3] > Data[i, 3]:
                                for s in range(r + 1, r + width):
                                    
                                    # Second trough
                                    if Data[s, indicator] < upper_barrier:
                                        Data[s, sell] = -1
                                        break
                                    else:
                                        break
                            else:
                                break
                        else:
                            break
                    else:
                        break
        except IndexError:
            pass 
    return Data


##############################################################################   1

my_data = rsi(my_data, lookback, 3, 4, genre = 'Smoothed')
my_data = divergence(my_data, 4, lower_barrier, upper_barrier, width, 6, 7)

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