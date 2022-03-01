
import datetime
import pytz
import pandas                    as pd
import MetaTrader5               as mt5
import matplotlib.pyplot         as plt
import numpy                     as np
import statistics                as stats
from scipy.stats                 import pearsonr
from scipy.stats                 import spearmanr
from scipy.ndimage.interpolation import shift
from scipy                       import stats

investment       = 100000                  
lot              = 100000
expected_cost    = 0.00000 # 1/10 of a Pip
sigchart         = False
signal_quality_period = 3

frameDict = {  
    'M1'  : [mt5.TIMEFRAME_M1,          1],
    'M2'  : [mt5.TIMEFRAME_M2,          2],
    'M3'  : [mt5.TIMEFRAME_M3,          3],
    'M4'  : [mt5.TIMEFRAME_M4,          4],
    'M5'  : [mt5.TIMEFRAME_M5,          5],
    'M6'  : [mt5.TIMEFRAME_M6,          6],
    'M10' : [mt5.TIMEFRAME_M10,        10],
    'M12' : [mt5.TIMEFRAME_M12,        12],
    'M15' : [mt5.TIMEFRAME_M15,        15],
    'M30' : [mt5.TIMEFRAME_M30,        30],
    'H1'  : [mt5.TIMEFRAME_H1,         60],
    'H2'  : [mt5.TIMEFRAME_H2,     2 * 60],
    'H3'  : [mt5.TIMEFRAME_H3,     3 * 60],
    'H4'  : [mt5.TIMEFRAME_H4,     4 * 60],
    'H6'  : [mt5.TIMEFRAME_H6,     6 * 60],
    'H8'  : [mt5.TIMEFRAME_H8,     8 * 60],
    'H12' : [mt5.TIMEFRAME_H12,   12 * 60],
    'D1'  : [mt5.TIMEFRAME_D1,       1440],
    'W1'  : [mt5.TIMEFRAME_W1,   7 * 1440],
    'MN1' : [mt5.TIMEFRAME_MN1, 30 * 1440]
}

now = datetime.datetime.now()

def asset_list(asset_set):
   
    if asset_set == 1:
        
        assets = ['EURUSD', 'USDCHF', 'GBPUSD', 'AUDUSD', 'NZDUSD',
                  'USDCAD', 'EURCAD', 'EURGBP', 'EURCHF', 'AUDCAD',
                  'EURNZD', 'NZDCHF', 'NZDCAD', 'EURAUD','AUDNZD',
                  'GBPCAD', 'AUDCHF', 'GBPAUD', 'GBPCHF', 'GBPNZD']
       
    elif asset_set == 'CRYPTO':
        
        assets = ['BTCUSD', 'ETHUSD', 'XRPUSD', 'LTCUSD']
       
    elif asset_set == 'COMMODITIES':
        
        assets = ['XAUUSD', 'XAGUSD', 'XPTUSD', 'XPDUSD']    
       
    return assets

def mass_import(asset, horizon):

    frameVal = frameDict.get(horizon)
    #Calculate 50000 bars 
    startSampleDate = datetime.datetime.now() - datetime.timedelta(minutes = 50001 * frameVal[1])
    data = get_quotes(frameVal[0], startSampleDate.year, startSampleDate.month, startSampleDate.day, asset = assets[asset])
    data = data.iloc[:, 1:5].values
    data = data.round(decimals = 5)    
    # Diagnostics 
    #print(assets[asset] + " " + horizon + " got data len = " + str(len(data)))
    #print("Start date: " + startSampleDate.strftime("%m/%d/%Y"))
    return data 
    

def get_quotes(time_frame, year = 2005, month = 1, day = 1, asset = "EURUSD"):
        
    # Establish connection to MetaTrader 5 
    if not mt5.initialize():
        print("initialize() failed, error code =", mt5.last_error())
        quit()
    
    timezone = pytz.timezone("Europe/Paris")
    
    utc_from = datetime.datetime(year, month, day, tzinfo = timezone)
    utc_to = datetime.datetime.now(timezone) + datetime.timedelta(days=1)
    
    rates = mt5.copy_rates_range(asset, time_frame, utc_from, utc_to)
    
    rates_frame = pd.DataFrame(rates)

    return rates_frame
    
def count_annotation(Data, name, onwhat, what_bull, what_bear, td, window = 50):
    
    Plottable = Data[-window:, ]
    
    fig, ax = plt.subplots(figsize = (10, 5))
    ax.grid()
    
    ax.plot(Plottable[:, onwhat], color = 'black', linewidth = 1.5, label = name)
    
    for i in range(len(Plottable)):
        
        if Plottable[i, what_bull] < 0 and Plottable[i, what_bull] != -td:
            
            x = i
            y = Plottable[i, onwhat]
        
            ax.annotate(int(Plottable[i, what_bull]), xy = (x, y), textcoords = "offset points", xytext = (0, - 10), ha = 'center',
                        color = 'blue')
            
        elif Plottable[i, what_bull] == -td:
            
            x = i
            y = Plottable[i, onwhat]
        
            ax.annotate(int(Plottable[i, what_bull]), xy = (x, y), textcoords = "offset points", xytext = (0, - 10), ha = 'center',
                        color = 'red')
            
        elif Plottable[i, what_bear] > 0 and Plottable[i, what_bear] != td:
            
            x = i
            y = Plottable[i, onwhat]
        
            ax.annotate(int(Plottable[i, what_bear]), xy = (x, y), textcoords = "offset points", xytext = (0, 10), ha = 'center',
                        color = 'blue' )

        elif Plottable[i, what_bear] == td:
            
            x = i
            y = Plottable[i, onwhat]
        
            ax.annotate(int(Plottable[i, what_bear]), xy = (x, y), textcoords = "offset points", xytext = (0, 10), ha = 'center',
                        color = 'red' )
                     
    ax.set_facecolor((0.95, 0.95, 0.95)) 
    plt.legend()
    
def adder(Data, times):
    
    for i in range(1, times + 1):
    
        new = np.zeros((len(Data), 1), dtype = float)
        Data = np.append(Data, new, axis = 1)

    return Data

def deleter(Data, index, times):
    
    for i in range(1, times + 1):
    
        Data = np.delete(Data, index, axis = 1)

    return Data
   
def jump(Data, jump):
    
    Data = Data[jump:, ]
    
    return Data

def rounding(Data, how_far):
    
    Data = Data.round(decimals = how_far)
    
    return Data
            







def rolling_correlation(Data, first_data, second_data, lookback, where):
    
    # Adding an extra column
    Data = adder(Data, 1)
    
    for i in range(len(Data)):
        
        try:
            Data[i, where] = pearsonr(Data[i - lookback + 1:i + 1, first_data], Data[i - lookback + 1:i + 1, second_data])[0]
            
             
        except ValueError:
            pass
    
    Data = jump(Data, lookback) 
    
    return Data

def auto_correlation(Data, first_data, second_data, shift_degree, lookback, where):
    
    new_array = shift(Data[:, first_data], shift_degree, cval = 0)
    new_array = np.reshape(new_array, (-1, 1))
    
    Data = np.concatenate((Data, new_array), axis = 1)
    Data = adder(Data, 1)
    
    for i in range(len(Data)):
        
        try:
            Data[i, where] = pearsonr(Data[i - lookback + 1:i + 1, first_data], Data[i - lookback + 1:i + 1, second_data])[0]
            
            
        except ValueError:
            pass
    
    Data = jump(Data, lookback) 
    Data = deleter(Data, where - 1, 1)
    
    return Data

def volatility(Data, lookback, what, where):
    
    # Adding an extra column
    Data = adder(Data, 1)
    
    for i in range(len(Data)):
        
        try:
            Data[i, where] = (Data[i - lookback + 1:i + 1, what].std())
    
        except IndexError:
            pass
     
    # Cleaning
    Data = jump(Data, lookback)    
     
    return Data

def normalizer(data, lookback, close, where):
            
    data = adder(data, 1)
        
    for i in range(len(data)):
            
        try:
            
            data[i, where] = (data[i, close] - min(data[i - lookback + 1:i + 1, close])) / (max(data[i - lookback + 1:i + 1, close]) - min(data[i - lookback + 1:i + 1, close]))
            
        except ValueError:
            
            pass
        
    data[:, where] = data[:, where] * 100  
            
    data = deleter(data, lookback)

    return data
   
def ma(Data, lookback, close, where): 
    
    Data = adder(Data, 1)
    
    for i in range(len(Data)):
           
            try:
                Data[i, where] = (Data[i - lookback + 1:i + 1, close].mean())
            
            except IndexError:
                pass
            
    # Cleaning
    Data = jump(Data, lookback)
    
    return Data

def ema(Data, alpha, lookback, what, where):
    
    alpha = alpha / (lookback + 1.0)
    beta  = 1 - alpha
    
    # First value is a simple SMA
    Data = ma(Data, lookback, what, where)
    
    # Calculating first EMA
    Data[lookback + 1, where] = (Data[lookback + 1, what] * alpha) + (Data[lookback, where] * beta)

    # Calculating the rest of EMA
    for i in range(lookback + 2, len(Data)):
            try:
                Data[i, where] = (Data[i, what] * alpha) + (Data[i - 1, where] * beta)
        
            except IndexError:
                pass
            
    return Data 
   
def lwma(Data, lookback, what):
    
    weighted = []
    for i in range(len(Data)):
            try:
                total = np.arange(1, lookback + 1, 1)
                
                matrix = Data[i - lookback + 1: i + 1, what:what + 1]
                matrix = np.ndarray.flatten(matrix)
                matrix = total * matrix
                wma = (matrix.sum()) / (total.sum())
                weighted = np.append(weighted, wma)

            except ValueError:
                pass
    
    Data = Data[lookback - 1:, ]
    weighted = np.reshape(weighted, (-1, 1)) 
    Data = np.concatenate((Data, weighted), axis = 1)   
    
    return Data

def kama(Data, what, where, lookback):
    
    Data = adder(Data, 10)
    
    # lookback from previous period
    for i in range(len(Data)):
        Data[i, where] = abs(Data[i, what] - Data[i - 1, what])
    
    Data[0, where] = 0
    
    # Sum of lookbacks
    for i in range(len(Data)):
        Data[i, where + 1] = (Data[i - lookback + 1:i + 1, where].sum())   
        
    # Volatility    
    for i in range(len(Data)):
        Data[i, where + 2] = abs(Data[i, what] - Data[i - lookback, what])
        
    Data = Data[lookback + 1:, ]
    
    # Efficiency Ratio
    Data[:, where + 3] = Data[:, where + 2] / Data[:, where + 1]
    
    for i in range(len(Data)):
        Data[i, where + 4] = np.square(Data[i, where + 3] * 0.6666666666666666667)
        
    for i in range(len(Data)):
        Data[i, where + 5] = Data[i - 1, where + 5] + (Data[i, where + 4] * (Data[i, what] - Data[i - 1, where + 5]))
        Data[11, where + 5] = 0
        
    Data = deleter(Data, where, 5)
    Data = jump(Data, lookback * 2)
    
    return Data

def BollingerBands(Data, boll_lookback, standard_distance, what, where):
       
    # Adding a few columns
    Data = adder(Data, 2)
    
    # Calculating means
    Data = ma(Data, boll_lookback, what, where)

    Data = volatility(Data, boll_lookback, what, where + 1)
    
    Data[:, where + 2] = Data[:, where] + (standard_distance * Data[:, where + 1])
    Data[:, where + 3] = Data[:, where] - (standard_distance * Data[:, where + 1])
    
    Data = jump(Data, boll_lookback)
    
    Data = deleter(Data, where, 2)
        
    return Data

def augmented_BollingerBands(Data, boll_lookback, standard_distance, high, low, where):
      
    Data = adder(Data, 10)
    
    # Calculating means
    Data = ema(Data, 2, boll_lookback, high, where)
    Data = ema(Data, 2, boll_lookback, low, where + 1)
    
    Data = volatility(Data, boll_lookback, high, where + 2)
    Data = volatility(Data, boll_lookback, low, where + 3)
    
    Data[:, where + 4] = Data[:, where] + (standard_distance * Data[:, where + 2])
    Data[:, where + 5] = Data[:, where + 1] - (standard_distance * Data[:, where + 3])
    
    Data = jump(Data, boll_lookback)
    
    Data = deleter(Data, where, 4)
        
    return Data

def atr(Data, lookback, high, low, close, where, genre = 'Smoothed'):
    
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
    
    if genre == 'Smoothed':
        
        # Average True Range Calculation
        Data = ema(Data, 2, lookback, where, where + 1)
    
    if genre == 'Simple':
    
        # Average True Range Calculation
        Data = ma(Data, lookback, where, where + 1)
    
    # Cleaning
    Data = deleter(Data, where, 1)
    Data = jump(Data, lookback)

    return Data

def pure_pupil(Data, lookback, high, low, where):
        
    volatility(Data, lookback, high, where)
    volatility(Data, lookback, low,  where + 1)   
         
    Data[:, where + 2] = (Data[:, where] + Data[:, where + 1]) / 2
    
    Data = jump(Data, lookback)  
    Data = ema(Data, 2, lookback, where + 2, where + 3)
    
    Data = jump(Data, lookback)
    Data = deleter(Data, where, 3)
    
    return Data

def rsi(Data, lookback, close, where, width = 1, genre = 'Smoothed'):
    
    # Adding a few columns
    Data = adder(Data, 5)
    
    # Calculating Differences
    for i in range(len(Data)):
        
        Data[i, where] = Data[i, close] - Data[i - width, close]
     
    # Calculating the Up and Down absolute values
    for i in range(len(Data)):
        
        if Data[i, where] > 0:
            
            Data[i, where + 1] = Data[i, where]
            
        elif Data[i, where] < 0:
            
            Data[i, where + 2] = abs(Data[i, where])
            
    # Calculating the Smoothed Moving Average on Up and Down absolute values    
    if genre == 'Smoothed':
        lookback = (lookback * 2) - 1 # From exponential to smoothed
        Data = ema(Data, 2, lookback, where + 1, where + 3)
        Data = ema(Data, 2, lookback, where + 2, where + 4)
    
    if genre == 'Simple':
        Data = ma(Data, lookback, where + 1, where + 3)
        Data = ma(Data, lookback, where + 2, where + 4)
        
    if genre == 'Hull':
        hull_moving_average(Data, where + 1, lookback, where + 3)
        hull_moving_average(Data, where + 2, lookback, where + 4)
    
    # Calculating the Relative Strength
    Data[:, where + 5] = Data[:, where + 3] / Data[:, where + 4]
    
    # Calculate the Relative Strength Index
    Data[:, where + 6] = (100 - (100 / (1 + Data[:, where + 5])))

    # Cleaning
    Data = deleter(Data, where, 6)
    Data = jump(Data, lookback)

    return Data
    

def fractal_indicator(Data, high, low, ema_lookback, min_max_lookback, where):
    
    Data = adder(Data, 20)
    
    Data = ema(Data, 2, ema_lookback, high, where)
    Data = ema(Data, 2, ema_lookback, low, where + 1)
    
    Data = volatility(Data, ema_lookback, high, where + 2)
    Data = volatility(Data, ema_lookback, low, where + 3)
    
    Data[:, where + 4] = Data[:, high] - Data[:, where]
    Data[:, where + 5] = Data[:, low]  - Data[:, where + 1]

    for i in range(len(Data)):
        try:
            Data[i, where + 6] = max(Data[i - min_max_lookback + 1:i + 1, where + 4])
        
        except ValueError:
            pass              
        
    for i in range(len(Data)):
        try:
            Data[i, where + 7] = min(Data[i - min_max_lookback + 1:i + 1, where + 5])
        
        except ValueError:
            pass  
         
    Data[:, where + 8] =  (Data[:, where +  2] + Data[:, where +  3]) / 2
    Data[:, where + 9] = (Data[:, where + 6] - Data[:, where + 7]) / Data[:, where + 8]
     
    Data = deleter(Data, 5, 9)
    Data = jump(Data, min_max_lookback)
    
    return Data


                  
def stochastic(Data, lookback, close, where, genre = 'High-Low'):
        
    # Adding a column
    Data = adder(Data, 1)
    
    if genre == 'High-Low':
        
        for i in range(len(Data)):
            
            try:
                Data[i, where] = (Data[i, close] - min(Data[i - lookback + 1:i + 1, 2])) / (max(Data[i - lookback + 1:i + 1, 1]) - min(Data[i - lookback + 1:i + 1, 2]))
            
            except ValueError:
                pass
        
        Data[:, where] = Data[:, where] * 100  
        Data = jump(Data, lookback)

    if genre == 'Normalization':
        
        for i in range(len(Data)):
            
            try:
                Data[i, where] = (Data[i, close] - min(Data[i - lookback + 1:i + 1, close])) / (max(Data[i - lookback + 1:i + 1, close]) - min(Data[i - lookback + 1:i + 1, close]))
            
            except ValueError:
                pass
        
        Data[:, where] = Data[:, where] * 100  
        Data = jump(Data, lookback)

    return Data           


                
def divergence(Data, indicator, lower_barrier, upper_barrier, width, buy, sell):
    
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

def hidden_divergence(Data, lower_barrier, upper_barrier, width):
    
    for i in range(len(Data)):
        
        try:
            if Data[i, 5] < lower_barrier and Data[i - 1, 5] > lower_barrier:
                
                for a in range(i + 1, i + width):
                    
                    # First trough
                    if Data[a, 5] > lower_barrier:
                        
                        for r in range(a + 1, a + width):
                            
                            if Data[r, 5] < lower_barrier and \
                            Data[r, 5] < Data[i, 5] and Data[r, 3] > Data[i, 3]:
                                
                                for s in range(r + 1, r + width):
                                    
                                    # Second trough
                                    if Data[s, 5] > lower_barrier:
                                        Data[s, 6] = 1
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
            if Data[i, 5] > upper_barrier  and Data[i - 1, 5] < upper_barrier:
                
                for a in range(i + 1, i + width):
                    
                    # First trough
                    if Data[a, 5] < upper_barrier:
                        for r in range(a + 1, a + width):
                            if Data[r, 5] > upper_barrier and \
                            Data[r, 5] > Data[i, 5] and Data[r, 3] < Data[i, 3]:
                                for s in range(r + 1, r + width):
                                    
                                    # Second trough
                                    if Data[s, 5] < upper_barrier:
                                        Data[s, 7] = -1
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
        
def vami(Data, lookback, moving_average_lookback, what, where):
    
    for i in range(len(Data)):
        
        Data[i, where] = Data[i, what] - Data[i - lookback, what]
        
    volatility(Data, lookback, what, where + 1)
        
    Data = jump(Data, lookback)
        
    Data[:, where + 2] = (Data[:, where] - Data[:, where + 1]) * 1000
    
    Data = ema(Data, 2, moving_average_lookback, where + 2, where + 3)       

    Data = jump(Data, moving_average_lookback)
    
    Data = deleter(Data, 5, 3)
    
    return Data
                  
def sar(s, af = 0.02, amax = 0.2):
    
    high, low = s.high, s.low

    # Starting values
    sig0, xpt0, af0 = True, high[0], af
    sar = [low[0] - (high - low).std()]

    for i in range(1, len(s)):
        sig1, xpt1, af1 = sig0, xpt0, af0

        lmin = min(low[i - 1], low[i])
        lmax = max(high[i - 1], high[i])

        if sig1:
            sig0 = low[i] > sar[-1]
            xpt0 = max(lmax, xpt1)
        else:
            sig0 = high[i] >= sar[-1]
            xpt0 = min(lmin, xpt1)

        if sig0 == sig1:
            sari = sar[-1] + (xpt1 - sar[-1])*af1
            af0 = min(amax, af1 + af)

            if sig0:
                af0 = af0 if xpt0 > xpt1 else af1
                sari = min(sari, lmin)
            else:
                af0 = af0 if xpt0 < xpt1 else af1
                sari = max(sari, lmax)
        else:
            af0 = af
            sari = xpt0

        sar.append(sari)

    return sar

def rri(Data, lookback, where):
    
    # Adding a column
    Data = adder(Data, 1)
    
    for i in range(len(Data)):
        
        Data[i, where] = (Data[i, 3] - Data[i - lookback, 0]) / (Data[i - lookback, 1] - Data[i - lookback, 2])
        
        if Data[i - lookback, 1] == Data[i - lookback, 2]:
            Data[i, where] = 0
    
    return Data

def macd(Data, what, long_ema, short_ema, signal_ema, where):
    
    
    Data = adder(Data, 1)
    
    Data = ema(Data, 2, long_ema,  what, where)
    Data = ema(Data, 2, short_ema, what, where + 1)
    
    Data[:, where + 2] = Data[:, where + 1] - Data[:, where]

    Data = jump(Data, long_ema)
    Data = ema(Data, 2, signal_ema, where + 2, where + 3)
    
    Data = deleter(Data, where, 2)   
    Data = jump(Data, signal_ema)
    
    return Data
            
def maci(Data, lookback, normalization_lookback, what, where):
    
    Data = adder(Data, 1)
    
    Data = ema(Data, 2, lookback, what, where)
    
    Data[:, where + 1] = Data[:, what] - Data[:, where]
    
    Data = stochastic(Data, normalization_lookback, where + 1, where + 2, genre = 'Normalization')
    
    Data = jump(Data, lookback)
    
    Data = deleter(Data, where, 2)

    return Data
    
def rainbow(Data, ma1, ma2, ma3, ma4, ma5, ma6, ma7, what, where):
    
    # Converting Exponential lookback to Smoothed Lookback
    ma1 = (ma1 * 2) - 1
    ma2 = (ma2 * 2) - 1
    ma3 = (ma3 * 2) - 1
    ma4 = (ma4 * 2) - 1
    ma5 = (ma5 * 2) - 1
    ma6 = (ma6 * 2) - 1
    ma7 = (ma7 * 2) - 1
    
    # Calculating the Smoothed Moving Averages A.K.A The Rainbow Moving Average
    Data = ema(Data, 2, ma1, what, where)
    Data = ema(Data, 2, ma2, what, where + 1)
    Data = ema(Data, 2, ma3, what, where + 2)
    Data = ema(Data, 2, ma4, what, where + 3)
    Data = ema(Data, 2, ma5, what, where + 4)
    Data = ema(Data, 2, ma6, what, where + 5)
    Data = ema(Data, 2, ma7, what, where + 6)
    
    Data = jump(Data, ma7)
    
    # The Rainbow Oscillator
    Data[:, where + 7] = Data[:, where] - Data[:, where + 6]
         
    return Data
    
def momentum_indicator(Data, lookback, what, where):
    
    Data = adder(Data, 1)
    
    for i in range(len(Data)):
        Data[i, where] = Data[i, what] / Data[i - lookback, what] * 100       
    
    Data = jump(Data, lookback)
    
    return Data
    
    
def fmi(Data, what, where, lookback, boll_lookback, standard_distance):
    
    for i in range(len(Data)):
        Data[i, where] = Data[i, what] / Data[i - lookback, what] * 100
    
    Data = BollingerBands(Data, boll_lookback, standard_distance, where, where + 1)
    
    Data[:, where + 3] = Data[:, where + 1] - Data[:, where]
    Data[:, where + 4] = Data[:, where] - Data[:, where + 2]
        
    Data = jump(Data, lookback)
        
    return Data    
    
def adx(Data, high, low, close, lookback, where):

    # DM+
    for i in range(len(Data)):
        if (Data[i, high] - Data[i - 1, high]) > (Data[i - 1, low] - Data[i, low]):
            Data[i, where] = Data[i, high] - Data[i - 1, high]
            
        else:
            Data[i, where] = 0
        
    # DM-
    for i in range(len(Data)):
        if (Data[i, high] - Data[i - 1, high]) < (Data[i - 1, low] - Data[i, low]):
            Data[i, where + 1] = Data[i - 1, low] - Data[i, low]
            
        else:
            Data[i, where + 1] = 0
        
    # Smoothing DI+   
    Data = ema(Data, 2, (lookback * 2 - 1), where, where + 2)
    
    # Smoothing DI-
    Data = ema(Data, 2, (lookback * 2 - 1), where + 1, where + 3)
    
    # Smoothing ATR
    Data = atr(Data, (lookback * 2 - 1), high, low, close, where + 4)
    
    Data = jump(Data, lookback)
    
    # DI+
    Data[:, where + 5] = Data[:, where + 2] / Data[:, where + 4]
    
    # DI-
    Data[:, where + 6] = Data[:, where + 3] / Data[:, where + 4]
    
    # ADX
    for i in range(len(Data)):
        Data[i, where + 7] = abs(Data[i, where + 5] - Data[i, where + 6]) / abs(Data[i, where + 5] + Data[i, where + 6]) * 100
    
    Data = ema(Data, 2, (lookback * 2 - 1), where + 7, where + 8)

    Data = jump(Data, lookback)

    Data = deleter(Data, where, 5)    
    
    return Data
    
def donchian(Data, low, high, lookback, where, median = 1):
    
    for i in range(len(Data)):
        try:
            Data[i, where] = max(Data[i - lookback:i + 1, high])
        
        except ValueError:
            pass
        
    for i in range(len(Data)):
        try:
            Data[i, where + 1] = min(Data[i - lookback:i + 1, low]) 
        
        except ValueError:
            pass
        
    if median == 1:
        
        for i in range(len(Data)): 
            try:
                Data[i, where + 2] = (Data[i, where] + Data[i, where + 1]) / 2 
            
            except ValueError:
                pass      
        
    Data = jump(Data, lookback)
    
    return Data

def ichimoku(Data, close, high, low, kijun_lookback, 
             tenkan_lookback,
             chikou_lookback,
             senkou_span_projection,
             senkou_span_b_lookback,
             where):
    
    Data = adder(Data, 3)
    
    # Kijun-sen
    for i in range(len(Data)):
        try:
            Data[i, where] = max(Data[i - kijun_lookback:i + 1, high]) + min(Data[i - kijun_lookback:i + 1, low])
    
        except ValueError:
            pass
        
    Data[:, where] = Data[:, where] / 2
    
    # Tenkan-sen
    for i in range(len(Data)):
        try:
            Data[i, where + 1] = max(Data[i - tenkan_lookback:i + 1, high]) + min(Data[i - tenkan_lookback:i + 1, low])
    
        except ValueError:
            pass
        
    Data[:, where + 1] = Data[:, where + 1] / 2

    # Senkou-span A
    senkou_span_a = (Data[:, where] + Data[:, where + 1]) / 2
    senkou_span_a = np.reshape(senkou_span_a, (-1, 1))

    # Senkou-span B
    for i in range(len(Data)):
        try:
            Data[i, where + 2] = max(Data[i - senkou_span_b_lookback:i + 1, high]) + min(Data[i - senkou_span_b_lookback:i + 1, low])
    
        except ValueError:
            pass
    
    Data[:, where + 2] = Data[:, where + 2] / 2  
    senkou_span_b = Data[:, where + 2]
    senkou_span_b = np.reshape(senkou_span_b, (-1, 1))
    kumo = np.concatenate((senkou_span_a, senkou_span_b), axis = 1)
    
    Data = deleter(Data, where + 2, 1)
    
    # Creating the Cloud
    Data = np.concatenate((Data, kumo), axis = 1)
    Data = Data[senkou_span_b_lookback:, ]

    for i in range (1, 7):
        
        new_array = shift(Data[:, 0], -senkou_span_projection, cval = 0)
        new_array = np.reshape(new_array, (-1, 1))
        Data = np.concatenate((Data, new_array), axis = 1)
        Data = deleter(Data, 0, 1)

    kumo = Data[:, 0:2]
    Data = deleter(Data, 0, 2)
    Data = np.concatenate((Data, kumo), axis = 1)
    
    Data = adder(Data, 1)   
    
    for i in range(len(Data)):  
        try:   
            Data[i, 8] = Data[i + chikou_lookback, 3]
        except IndexError:
            pass
    
    Data[-senkou_span_projection:, 0] = Data[-senkou_span_projection:, 0] / 0
    Data[-senkou_span_projection:, 1] = Data[-senkou_span_projection:, 1] / 0
    Data[-senkou_span_projection:, 2] = Data[-senkou_span_projection:, 2] / 0
    Data[-senkou_span_projection:, 3] = Data[-senkou_span_projection:, 3] / 0
    Data[-senkou_span_projection:, 4] = Data[-senkou_span_projection:, 4] / 0
    Data[-senkou_span_projection:, 5] = Data[-senkou_span_projection:, 5] / 0
    Data[-52:, 8] = Data[-52:, 8] / 0
    
    return Data

def kma(Data, high, low, lookback, where):
    
    Data = ma(Data, lookback, high, where)
    Data = ma(Data, lookback, low, where + 1)
    
    Data = jump(Data, lookback)
    
    return Data

def supertrend(Data, multiplier, atr_col, close, high, low, where):
    
    Data = adder(Data, 6)
    
    for i in range(len(Data)):
        
            # Average Price
            Data[i, where] = (Data[i, high] + Data[i, low]) / 2
            # Basic Upper Band
            Data[i, where + 1] = Data[i, where] + (multiplier * Data[i, atr_col])
            # Lower Upper Band
            Data[i, where + 2] = Data[i, where] - (multiplier * Data[i, atr_col])
    
    # Final Upper Band
    for i in range(len(Data)):
        
        if i == 0:
            Data[i, where + 3] = 0
            
        else:  
            if (Data[i, where + 1] < Data[i - 1, where + 3]) or (Data[i - 1, close] > Data[i - 1, where + 3]):
                Data[i, where + 3] = Data[i, where + 1]
            
            else:
                Data[i, where + 3] = Data[i - 1, where + 3]
    
    # Final Lower Band
    for i in range(len(Data)):
        
        if i == 0:
            Data[i, where + 4] = 0
            
        else:  
            if (Data[i, where + 2] > Data[i - 1, where + 4]) or (Data[i - 1, close] < Data[i - 1, where + 4]):
                Data[i, where + 4] = Data[i, where + 2]
            
            else:
                Data[i, where + 4] = Data[i - 1, where + 4]
      
    # SuperTrend
    for i in range(len(Data)):
        
        if i == 0:
            Data[i, where + 5] = 0
        
        elif (Data[i - 1, where + 5] == Data[i - 1, where + 3]) and (Data[i, close] <= Data[i, where + 3]):
            Data[i, where + 5] = Data[i, where + 3]
        
        elif (Data[i - 1, where + 5] == Data[i - 1, where + 3]) and (Data[i, close] >  Data[i, where + 3]):
            Data[i, where + 5] = Data[i, where + 4]
        
        elif (Data[i - 1, where + 5] == Data[i - 1, where + 4]) and (Data[i, close] >= Data[i, where + 4]):
            Data[i, where + 5] = Data[i, where + 4]
        
        elif (Data[i - 1, where + 5] == Data[i - 1, where + 4]) and (Data[i, close] <  Data[i, where + 4]):
            Data[i, where + 5] = Data[i, where + 3]   
            
    # Cleaning columns
    Data = deleter(Data, where, 5)        
    
    return Data

def differentials(Data, what, true_low, true_high, buy, sell, differential = 1):
    
    
    Data = adder(Data, 4)
    
    if differential == 1:
            
        for i in range(len(Data)):
            
            # True low
            Data[i, true_low] = min(Data[i, 2], Data[i - 1, what])
            Data[i, true_low] = Data[i, what] - Data[i, true_low]
                
            # True high  
            Data[i, true_high] = max(Data[i, 1], Data[i - 1, what])
            Data[i, true_high] = Data[i, what] - Data[i, true_high]
            
            # TD Differential
            if Data[i, what] < Data[i - 1, what] and Data[i - 1, what] < Data[i - 2, what] and \
               Data[i, true_low] > Data[i - 1, true_low] and Data[i, true_high] < Data[i - 1, true_high]: 
                   Data[i, buy] = 1 
    
            if Data[i, what] > Data[i - 1, what] and Data[i - 1, what] > Data[i - 2, what] and \
               Data[i, true_low] < Data[i - 1, true_low] and Data[i, true_high] > Data[i - 1, true_high]: 
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
            if Data[i, what] < Data[i - 1, what] and Data[i - 1, what] < Data[i - 2, what] and \
               Data[i, true_low] < Data[i - 1, true_low] and Data[i, true_high] > Data[i - 1, true_high]: 
                   Data[i, buy] = 1 
    
            if Data[i, what] > Data[i - 1, what] and Data[i - 1, what] > Data[i - 2, what] and \
               Data[i, true_low] > Data[i - 1, true_low] and Data[i, true_high] < Data[i - 1, true_high]: 
                   Data[i, sell] = -1            
        
    if differential == 3:

        for i in range(len(Data)):
            
            if Data[i, what] < Data[i - 1, what] and Data[i - 1, what] > Data[i - 2, what] and \
               Data[i - 2, what] < Data[i - 3, what] and Data[i - 3, what] < Data[i - 4, what]: 
                   Data[i, buy] = 1 
    
            if Data[i, what] > Data[i - 1, what] and Data[i - 1, what] < Data[i - 2, what] and \
               Data[i - 2, what] > Data[i - 3, what] and Data[i - 3, what] > Data[i - 4, what]: 
                   Data[i, sell] = -1           
    
    Data = deleter(Data, 5, 1)    
    
    return Data

def fib_stoch(Data, volatility_lookback, what, where):
        
    Data = volatility(Data, volatility_lookback, what, where)
    Data = normalizer(Data, volatility_lookback, where, where + 1)

    for i in range(len(Data)):
        
        Data[i, where + 1] = round(Data[i, where + 1], 0) 
    
    for i in range(len(Data)):
        
        if Data[i, where + 1] >= 0 and Data[i, where + 1] <= 10 :
            Data[i, where + 1] = 144
        if Data[i, where + 1] > 10 and Data[i, where + 1] <= 20 :
            Data[i, where + 1] = 89            
        if Data[i, where + 1] > 20 and Data[i, where + 1] <= 30 :
            Data[i, where + 1] = 55
        if Data[i, where + 1] > 30 and Data[i, where + 1] <= 40 :
            Data[i, where + 1] = 34
        if Data[i, where + 1] > 40 and Data[i, where + 1] <= 50 :
            Data[i, where + 1] = 21
        if Data[i, where + 1] > 50 and Data[i, where + 1] <= 60 :
            Data[i, where + 1] = 13
        if Data[i, where + 1] > 60 and Data[i, where + 1] <= 70 :
            Data[i, where + 1] = 8
        if Data[i, where + 1] > 70 and Data[i, where + 1] <= 80 :
            Data[i, where + 1] = 5
        if Data[i, where + 1] > 80 and Data[i, where + 1] <= 90 :
            Data[i, where + 1] = 3
        if Data[i, where + 1] > 90 and Data[i, where + 1] <= 100 :            
            Data[i, where + 1] = 2            
    
    Data = jump(Data, volatility_lookback)
        
    for i in range(len(Data)):
        
        try:
            lookback = int(Data[i, where + 1])
            Data[i, where + 2] = (Data[i, what] - min(Data[i - lookback + 1:i + 1, 2])) / (max(Data[i - lookback + 1:i + 1, 1]) - min(Data[i - lookback + 1:i + 1, 2]))
        
        except ValueError:
            pass
    
    Data[:, where + 2] = Data[:, where + 2] * 100  
    Data = deleter(Data, where, 2)

    return Data           

def dynamic_rsi(Data, momentum_lookback, corr_lookback, what, where):
        
    for i in range(len(Data)):
        Data[i, where] = Data[i, what] / Data[i - momentum_lookback, what] * 100
    
    Data = jump(Data, momentum_lookback)    

    Data = rolling_correlation(Data, what, where, corr_lookback, where + 1)
    
    for i in range(len(Data)):
        
        if Data[i, where + 1] >= -1.00 and Data[i, where + 1] <= 0.10 :
            Data[i, where + 1] = 14
        if Data[i, where + 1] > 0.10 and Data[i, where + 1] <= 0.20 :
            Data[i, where + 1] = 10            
        if Data[i, where + 1] > 0.20 and Data[i, where + 1] <= 0.30 :
            Data[i, where + 1] = 9
        if Data[i, where + 1] > 0.30 and Data[i, where + 1] <= 0.40 :
            Data[i, where + 1] = 8
        if Data[i, where + 1] > 0.40 and Data[i, where + 1] <= 0.50 :
            Data[i, where + 1] = 7
        if Data[i, where + 1] > 0.50 and Data[i, where + 1] <= 0.60 :
            Data[i, where + 1] = 6
        if Data[i, where + 1] > 0.60 and Data[i, where + 1] <= 0.70 :
            Data[i, where + 1] = 5
        if Data[i, where + 1] > 0.70 and Data[i, where + 1] <= 0.80 :
            Data[i, where + 1] = 4
        if Data[i, where + 1] > 0.80 and Data[i, where + 1] <= 0.90 :
            Data[i, where + 1] = 3
        if Data[i, where + 1] > 0.90 and Data[i, where + 1] <= 1.00 :            
            Data[i, where + 1] = 2             
    
    Data = rsi(Data, 14, 3, 0)
    Data = rsi(Data, 10, 3, 0)
    Data = rsi(Data, 9, 3, 0)
    Data = rsi(Data, 8, 3, 0)
    Data = rsi(Data, 7, 3, 0)
    Data = rsi(Data, 6, 3, 0)
    Data = rsi(Data, 5, 3, 0)
    Data = rsi(Data, 4, 3, 0)
    Data = rsi(Data, 3, 3, 0)
    Data = rsi(Data, 2, 3, 0)
    
    Data = adder(Data, 1)

    for i in range(len(Data)):
        
        if Data[i, where + 1] == 14:
            Data[i, where + 12] = Data[i, where + 2]
        if Data[i, where + 1] == 10:
            Data[i, where + 12] = Data[i, where + 3]            
        if Data[i, where + 1] == 9:
            Data[i, where + 12] = Data[i, where + 4]
        if Data[i, where + 1] == 8:
            Data[i, where + 12] = Data[i, where + 5]
        if Data[i, where + 1] == 7:
            Data[i, where + 12] = Data[i, where + 6]
        if Data[i, where + 1] == 6:
            Data[i, where + 12] = Data[i, where + 7]
        if Data[i, where + 1] == 5:
            Data[i, where + 12] = Data[i, where + 8]
        if Data[i, where + 1] == 4:
            Data[i, where + 12] = Data[i, where + 9]
        if Data[i, where + 1] == 3:
            Data[i, where + 12] = Data[i, where + 10]
        if Data[i, where + 1] == 2:            
            Data[i, where + 12] = Data[i, where + 11]   
              
    Data = deleter(Data, where, 12)

    return Data

def keltner_channel(Data, ma_lookback, atr_lookback, multiplier, what, where):
    
    # Adding a few columns
    Data = adder(Data, 3)
    
    Data = ema(Data, 2, ma_lookback, what, where)
    
    Data = atr(Data, atr_lookback, 2, 1, 3, where + 1)
    
    Data[:, where + 2] = Data[:, where] + (Data[:, where + 1] * multiplier)
    Data[:, where + 3] = Data[:, where] - (Data[:, where + 1] * multiplier)

    Data = deleter(Data, where, 2)
    Data = jump(Data, ma_lookback)

    return Data

def mawi(Data, short_ma, long_ma, normalization_lookback, what, where):
    
    Data = ma(Data, short_ma, what, where)
    Data = ma(Data, long_ma, what, where + 1)

    # MAWI line (Width) 
    Data[:, where + 2] = Data[:, where] - Data[:, where + 1]

    # MAWI normalized
    Data = normalizer(Data, normalization_lookback, where + 2, where + 3) 
    Data[:, where + 3] =  Data[:, where + 3]
    
    Data = jump(Data, long_ma + normalization_lookback)
    Data = deleter(Data, where, 2)
    
    return Data

def vri(Data, lookback, what, high, low, where):
    
    Data = volatility(Data, lookback, what, where)

    for i in range(len(Data)):    
        Data[i, where + 1] = Data[i, what] - Data[i - lookback, 0]

    for i in range(len(Data)):
        try:
            Data[i, where + 2] = max(Data[i - lookback:i + 1, high]) 
        
        except ValueError:
            pass

    for i in range(len(Data)):
        try:
            Data[i, where + 3] = min(Data[i - lookback:i + 1, low]) 
        
        except ValueError:
            pass
        
    Data[:, where + 4] = Data[:, where + 1] / (Data[:, where + 2] - Data[:, where + 3])    
        
    Data[:, where + 4] = Data[:, where + 4] * Data[:, where] * 1000   
 
    Data = jump(Data, lookback)
    Data = deleter(Data, where, 4)
       
    return Data

def modified_td_flip(Data, td, step, high, low, where_long, where_short):
        
    # Timing buy signal
    counter = -1 

    for i in range(len(Data)):    
        if Data[i, low] < Data[i - step, low]:
            Data[i, where_long] = counter
            counter += -1       
            if counter == -td - 1:
                counter = 0
            else:
                continue        
        elif Data[i, low] >= Data[i - step, low]:
            counter = -1 
            Data[i, where_long] = 0 
    
    if Data[8, where_long] == -td:
        Data = Data[9:,]
    elif Data[7, where_long] == -td + 1:
        Data = Data[8:,]
    elif Data[6, where_long] == -td + 2:
        Data = Data[7:,]
    elif Data[5, where_long] == -td + 3:
        Data = Data[6:,]
    elif Data[4, where_long] == -td + 4:
        Data = Data[5:,]
        
    # Timing sell signal       
    counter = 1 
    
    for i in range(len(Data)):
        if Data[i, high] > Data[i - step, high]: 
            Data[i, where_short] = counter 
            counter += 1        
            if counter == td + 1: 
                counter = 0            
            else:
                continue        
        elif Data[i, high] <= Data[i - step, high]: 
            counter = 1 
            Data[i, where_short] = 0 
    
    if Data[8, where_short] == td:
        Data = Data[9:,]
    elif Data[7, where_short] == td - 1:
        Data = Data[8:,]
    elif Data[6, where_short] == td - 2:
        Data = Data[7:,]
    elif Data[5, where_short] == td - 3:
        Data = Data[6:,]
    elif Data[4, where_short] == td - 4:
        Data = Data[5:,] 
        
    return Data

def stationary_indicator(Data, lag, what, where, cutoff):
    
    for i in range(len(Data)):
        Data[i, where] = (Data[i, what] - Data[i - lag, what])
    
    Data = jump(Data, lag)
    
    for i in range(len(Data)):
    
        if Data[i, where] > cutoff:
            Data[i, where] = cutoff
            
        if Data[i, where] < -cutoff:
            Data[i, where] = -cutoff

    return Data

def stationary_extreme_indicator(Data, lag, high, low, where, cutoff):
    
    for i in range(len(Data)):
        Data[i, where] = (Data[i, high] - Data[i - lag, high]) * 10000

    for i in range(len(Data)):
        Data[i, where + 1] = (Data[i, low] - Data[i - lag, low]) * 10000
    
    Data = jump(Data, lag)
    
    for i in range(len(Data)):
    
        if Data[i, where] > cutoff:
            Data[i, where] = cutoff
            
        if Data[i, where] < -cutoff:
            Data[i, where] = -cutoff

    for i in range(len(Data)):
    
        if Data[i, where + 1] > cutoff:
            Data[i, where + 1] = cutoff
            
        if Data[i, where + 1] < -cutoff:
            Data[i, where + 1] = -cutoff
            
    return Data

def fib(n):
   if n == 1:
      return 1
  
   elif n == 0:   
      return 0 
           
   else:                      
      return fib(n - 1) + fib(n - 2) 

def fibonnaci_moving_average(Data, where):
    
    # Adding Columns
    Data = adder(Data, 40)
    
    # Calculating Different Moving Averages
    Data = ema(Data, 2, 3,    1, where)
    Data = ema(Data, 2, 5,    1, where + 1)    
    Data = ema(Data, 2, 8,    1, where + 2)    
    Data = ema(Data, 2, 13,   1, where + 3)    
    Data = ema(Data, 2, 21,   1, where + 4)    
    Data = ema(Data, 2, 34,   1, where + 5)    
    Data = ema(Data, 2, 55,   1, where + 6)    
    Data = ema(Data, 2, 89,   1, where + 7)    
    Data = ema(Data, 2, 144,  1, where + 8)    
    Data = ema(Data, 2, 233,  1, where + 9)    
    Data = ema(Data, 2, 377,  1, where + 10)    
    Data = ema(Data, 2, 610,  1, where + 11)    
    Data = ema(Data, 2, 987,  1, where + 12)    
    Data = ema(Data, 2, 1597, 1, where + 13) 
    Data = ema(Data, 2, 2584, 1, where + 14) 
    Data = ema(Data, 2, 4181, 1, where + 15) 
    Data = ema(Data, 2, 6765, 1, where + 16) 
    

    Data[:, where + 17] = Data[:, where] +  Data[:, where + 1] + Data[:, where + 2] + Data[:, where + 3] + Data[:, where + 4] + \
                          Data[:, where + 5] + Data[:, where + 6] + Data[:, where + 7] + Data[:, where + 8] + Data[:, where + 9] + Data[:, where + 10] + Data[:, where + 11] + \
                          Data[:, where + 12] + Data[:, where + 13] + Data[:, where + 14] + Data[:, where + 15] + Data[:, where + 16]   
    

    Data[:, where + 17] = Data[:, where + 17] / 17

    Data = deleter(Data, 4, 17)
 
    # Calculating Different Moving Averages
    Data = ema(Data, 2, 3,    2, where + 1)
    Data = ema(Data, 2, 5,    2, where + 2)    
    Data = ema(Data, 2, 8,    2, where + 3)    
    Data = ema(Data, 2, 13,   2, where + 4)    
    Data = ema(Data, 2, 21,   2, where + 5)    
    Data = ema(Data, 2, 34,   2, where + 6)    
    Data = ema(Data, 2, 55,   2, where + 7)    
    Data = ema(Data, 2, 89,   2, where + 8)    
    Data = ema(Data, 2, 144,  2, where + 9)    
    Data = ema(Data, 2, 233,  2, where + 10)    
    Data = ema(Data, 2, 377,  2, where + 11)    
    Data = ema(Data, 2, 610,  2, where + 12)    
    Data = ema(Data, 2, 987,  2, where + 13)    
    Data = ema(Data, 2, 1597, 2, where + 14) 
    Data = ema(Data, 2, 2584, 2, where + 15) 
    Data = ema(Data, 2, 4181, 2, where + 16) 
    Data = ema(Data, 2, 6765, 2, where + 17)  
    
    Data[:, where + 18] = Data[:, where + 1] + Data[:, where + 2] + Data[:, where + 3] + Data[:, where + 4] + \
                          Data[:, where + 5] + Data[:, where + 6] + Data[:, where + 7] + Data[:, where + 8] + Data[:, where + 9] + Data[:, where + 10] + Data[:, where + 11] + \
                          Data[:, where + 12] + Data[:, where + 13] + Data[:, where + 14] + Data[:, where + 15] + Data[:, where + 16] + Data[:, where + 17]  
    

    Data[:, where + 18] = Data[:, where + 18] / 17

    Data = deleter(Data, 5, 17)
    
    return Data

def cci(Data, lookback, what, where, constant):
    
    # Calculating Typical Price
    Data[:, where] = (Data[:, 1] + Data[:, 2] + Data[:, 3]) / 3
    
    # Calculating the Absolute Mean Deviation
    specimen = Data[:, where]
    MAD_Data = pd.Series(specimen)
    
    for i in range(len(Data)):
            
            Data[i, where + 1] = MAD_Data[i - lookback:i].mad()
    
    # Calculating Mean of Typical Price 
    Data = ma(Data, lookback, where, where + 2)
    
    # CCI
    for i in range(len(Data)):
        Data[i, where + 3] = (Data[i, where] - Data[i, where + 2]) / (constant * Data[i, where + 1]) 
    
    Data = jump(Data, lookback)   
    Data = deleter(Data, where, 3)
    
    return Data

def long_range_indicator(Data, lookback, high, low, where):
    
    Data = normalizer(Data, lookback, high, where)
    Data = normalizer(Data, lookback, low, where + 1)
    
    Data = jump(Data, lookback)

    return Data      

def money_flow_multiplier(Data, what, high, low, where):
    
    Data[:, where]     = Data[:, what] - Data[:, low]
    Data[:, where + 1] = Data[:, high] - Data[:, what]
    
    Data[:, where + 2] = Data[:, where] - Data[:, where + 1]
    
    Data[:, where + 3] = Data[:, high] - Data[:, low]
    for i in range(len(Data)):
        if Data[i, where + 3] == 0:
            Data[i, where + 3] = 0.0001
    
    Data[:, where + 4] = (Data[:, where + 2] / Data[:, where + 3]) * 100
    Data = deleter(Data, where, 4)
    
    return Data
    
def spiral_indicator(Data, opening, close, high, low, where):
    
    Data[:, where]     = Data[:, high] - Data[:, opening]
    Data[:, where + 1] = Data[:, high] - Data[:, low]
    Data[:, where + 2] = Data[:, high] - Data[:, close]
    Data[:, where + 3] = Data[:, where] + Data[:, where + 1] + Data[:, where + 2]
    
    Data[:, where + 4] = Data[:, opening] - Data[:, low]
    Data[:, where + 5] = Data[:, high] - Data[:, low]
    Data[:, where + 6] = Data[:, close] - Data[:, low] 
    Data[:, where + 7] = Data[:, where + 4] + Data[:, where + 5] + Data[:, where + 6]
    
    Data[:, where + 8] = Data[:, where + 3] - Data[:, where + 7]
    
    Data = deleter(Data, where, 8)
 
    return Data    

def eRSI(Data, rsi_lookback, high1, high2, low1, low2):    
    
    rsi_lookback = (rsi_lookback * 2) - 1 # From exponential to smoothed
          
    # Get the difference in price from previous step
    delta = []
   
    for i in range(len(Data)):
        try:
            diff = Data[i, high1] - Data[i - 1, high1] 
            delta = np.append(delta, diff)                  
        except IndexError:
            pass
        
    delta = np.insert(delta, 0, 0, axis = 0)               
    delta = delta[1:] 
    
    # Make the positive gains (up) and negative gains (down) Series
    up, down = delta.copy(), delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0
    
    up = np.array(up)
    down = np.array(down)
    
    roll_up = up
    roll_down = down
    
    roll_up = np.reshape(roll_up, (-1, 1))
    roll_down = np.reshape(roll_down, (-1, 1))
    
    roll_up = adder(roll_up, 3)
    roll_down = adder(roll_down, 3)
    
    roll_up = ema(roll_up, 2, rsi_lookback, high2, 1)
    roll_down = ema(abs(roll_down), 2, rsi_lookback, high2, 1)
    
    # Calculate the SMA
    roll_up = roll_up[rsi_lookback:, 1:2]
    roll_down = roll_down[rsi_lookback:, 1:2]
    Data = Data[rsi_lookback + 1:,]
    
    # Calculate the RSI based on SMA
    RS = roll_up / roll_down
    RSI = (100.0 - (100.0 / (1.0 + RS)))
    RSI = np.array(RSI)
    RSI = np.reshape(RSI, (-1, 1))
    RSI = RSI[1:,]
    
    Data = np.concatenate((Data, RSI), axis = 1) 
    
    # Get the difference in price from previous step
    delta = []
   
    for i in range(len(Data)):
        try:
            diff = Data[i, low1] - Data[i - 1, low1] 
            delta = np.append(delta, diff)                  
        except IndexError:
            pass
        
    delta = np.insert(delta, 0, 0, axis = 0)               
    delta = delta[1:] 
    
    # Make the positive gains (up) and negative gains (down) Series
    up, down = delta.copy(), delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0
    
    up = np.array(up)
    down = np.array(down)
    
    roll_up = up
    roll_down = down
    
    roll_up = np.reshape(roll_up, (-1, 1))
    roll_down = np.reshape(roll_down, (-1, 1))
    
    roll_up = adder(roll_up, 3)
    roll_down = adder(roll_down, 3)
    
    roll_up = ema(roll_up, 2, rsi_lookback, low2, 1)
    roll_down = ema(abs(roll_down), 2, rsi_lookback, low2, 1)
    
    # Calculate the SMA
    roll_up = roll_up[rsi_lookback:, 1:2]
    roll_down = roll_down[rsi_lookback:, 1:2]
    Data = Data[rsi_lookback + 1:,]
    
    # Calculate the RSI based on SMA
    RS = roll_up / roll_down
    RSI = (100.0 - (100.0 / (1.0 + RS)))
    RSI = np.array(RSI)
    RSI = np.reshape(RSI, (-1, 1))
    RSI = RSI[1:,]
    
    Data = np.concatenate((Data, RSI), axis = 1)    
    
    return Data

def ma_steepness(Data, lookback, steepness_period, what, where):
    
    Data = ma(Data, lookback, what, where)
    Data = ma(Data, lookback, where, where + 1)
    Data = deleter(Data, where, 1)
    
    for i in range(len(Data)):
        Data[i, where + 1] = (Data[i, where] - Data[i - steepness_period, where]) / (i - (i - steepness_period))
        
    Data = deleter(Data, where, 1)
    Data = jump(Data, lookback + steepness_period)
    
    return Data
    
def slope_indicator(Data, lookback, what, where):
        
    for i in range(len(Data)):
        Data[i, where] = (Data[i, what] - Data[i - lookback, what]) / (i - (i - lookback))
        
    Data = jump(Data, lookback)
    
    return Data    
    
def volatility_moving_average(Data, lookback, what, where):

    Data = volatility(Data, lookback, what, where)
    Data = ma(Data, lookback, what, where + 1)

    for i in range(len(Data)):
        if Data[i, what] > Data[i, where + 1]:
            Data[i, where + 1] = Data[i, where + 1] - (Data[i, where] * 0.25)
        
        if Data[i, what] < Data[i, where + 1]:
            Data[i, where + 1] = Data[i, where + 1] + (Data[i, where] * 0.25)
    
    Data = jump(Data, lookback)
    Data = deleter(Data, where, 1)
    
    return Data
            
def mirror_slope_indicator(Data, lookback, high, low, where):
        
    for i in range(len(Data)):
        Data[i, where] = (Data[i, high] - Data[i - lookback, high]) / (i - (i - lookback))
    
    for i in range(len(Data)):
        Data[i, where + 1] = (Data[i, low] - Data[i - lookback, low]) / (i - (i - lookback))
        
    Data = jump(Data, lookback)
    
    return Data             

def relative_volatility_index(Data, lookback, close, where):

    # Calculating Volatility
    Data = volatility(Data, lookback, close, where)
    
    # Calculating the RSI on Volatility
    Data = rsi(Data, lookback, where, where + 1, width = 1, genre = 'Smoothed') 
    
    # Cleaning
    Data = deleter(Data, where, 1)
    
    return Data       

def dynamic_relative_strength_index(Data, lookback, close, where):
    
    # Calculating the Relative Volatility Index
    Data = relative_volatility_index(Data, lookback, close, where)
    
    # Calculating the Relative Strength Index
    Data = rsi(Data, lookback, close, where + 1)
    
    # Calculating the Lookback Periods on the Dynamic Relative Strength Index
    for i in range(len(Data)):
        
        if Data[i, where] >= 0 and Data[i, where] <= 10 :
            Data[i, where + 1] = 0.90 * Data[i, where + 1]
            
        if Data[i, where] > 10 and Data[i, where] <= 20 :
            Data[i, where + 1] = 0.80 * Data[i, where + 1]
            
        if Data[i, where] > 20 and Data[i, where] <= 30 :
            Data[i, where + 1] = 0.70 * Data[i, where + 1]
            
        if Data[i, where] > 30 and Data[i, where] <= 40 :
            Data[i, where + 1] = 0.60 * Data[i, where + 1]
            
        if Data[i, where] > 40 and Data[i, where] <= 50 :
            Data[i, where + 1] = 0.50 * Data[i, where + 1]
            
        if Data[i, where] > 50 and Data[i, where] <= 60 :
            Data[i, where + 1] = 0.40 * Data[i, where + 1]
            
        if Data[i, where] > 60 and Data[i, where] <= 70 :
            Data[i, where + 1] = 0.30 * Data[i, where + 1]
            
        if Data[i, where] > 70 and Data[i, where] <= 80 :
            Data[i, where + 1] = 0.20 * Data[i, where + 1]
            
        if Data[i, where] > 80 and Data[i, where] <= 90 :
            Data[i, where + 1] = 0.10 * Data[i, where + 1]
            
        if Data[i, where] > 90 and Data[i, where] <= 100 :
            Data[i, where + 1] = 0.05 * Data[i, where + 1]
            
    # Cleaning
    Data = deleter(Data, where, 1)
    
    return Data
   
def fractals(Data, high, low, buy, sell):
    
    # Fractal up - bullish breakout signal
    for i in range(len(Data)):
        if Data[i, high] < Data[i - 2, high] and Data[i - 1, high] < Data[i - 2, high] and \
            Data[i - 2, high] > Data[i - 3, high] and Data[i - 2, high] > Data[i - 4, high]:
                Data[i - 2, buy] = 1
   
    # Fractal down - bearish breakout signal
    for i in range(len(Data)):
        if Data[i, low] > Data[i - 2, low] and Data[i - 1, low] > Data[i - 2, low] and \
            Data[i - 2, low] < Data[i - 3, low] and Data[i - 2, low] < Data[i - 4, low]:
                Data[i - 2, sell] = -1   
                
    return Data

   

def ohlc_plot_candles(Data, window):
      
    Chosen = Data[-window:, ]
    
    for i in range(len(Chosen)):
        
        plt.vlines(x = i, ymin = Chosen[i, 2], ymax = Chosen[i, 1], color = 'black', linewidth = 1)  
        
        if Chosen[i, 3] > Chosen[i, 0]:
            color_chosen = 'mediumseagreen'
            plt.vlines(x = i, ymin = Chosen[i, 0], ymax = Chosen[i, 3], color = color_chosen, linewidth = 3)  

        if Chosen[i, 3] < Chosen[i, 0]:
            color_chosen = 'maroon'
            plt.vlines(x = i, ymin = Chosen[i, 3], ymax = Chosen[i, 0], color = color_chosen, linewidth = 3)  
            
        if Chosen[i, 3] == Chosen[i, 0]:
            color_chosen = 'black'
            plt.vlines(x = i, ymin = Chosen[i, 3], ymax = Chosen[i, 0], color = color_chosen, linewidth = 3)  
            
    plt.grid()

def ohlc_plot_candles_k(Data, opening, high, low, close, window):
      
    Chosen = Data[-window:, ]
    
    for i in range(len(Chosen)):
        
        plt.vlines(x = i, ymin = Chosen[i, low], ymax = Chosen[i, high], color = 'black', linewidth = 1)  
        
        if Chosen[i, close] > Chosen[i, opening]:
            color_chosen = 'mediumseagreen'
            plt.vlines(x = i, ymin = Chosen[i, opening], ymax = Chosen[i, close], color = color_chosen, linewidth = 3)  

        if Chosen[i, close] < Chosen[i, opening]:
            color_chosen = 'maroon'
            plt.vlines(x = i, ymin = Chosen[i, close], ymax = Chosen[i, opening], color = color_chosen, linewidth = 3)  
            
        if Chosen[i, close] == Chosen[i, opening]:
            color_chosen = 'black'
            plt.vlines(x = i, ymin = Chosen[i, close], ymax = Chosen[i, opening], color = color_chosen, linewidth = 3)  
            
    plt.grid()
       
def ohlc_plot_k(Data, window, name):
      
    Chosen = Data[-window:, ]
    
    for i in range(len(Chosen)):
        
        plt.vlines(x = i, ymin = Chosen[i, 7], ymax = Chosen[i, 6], color = 'black', linewidth = 1)  
        
        if Chosen[i, 8] > Chosen[i, 5]:
            color_chosen = 'blue'
            plt.vlines(x = i, ymin = Chosen[i, 5], ymax = Chosen[i, 8], color = color_chosen, linewidth = 3)  

        if Chosen[i, 8] < Chosen[i, 5]:
            color_chosen = 'brown'
            plt.vlines(x = i, ymin = Chosen[i, 8], ymax = Chosen[i, 5], color = color_chosen, linewidth = 3)  
            
        if Chosen[i, 8] == Chosen[i, 5]:
            color_chosen = 'black'
            plt.vlines(x = i, ymin = Chosen[i, 8], ymax = Chosen[i, 5], color = color_chosen, linewidth = 3)  
            
    plt.grid()
    plt.title(name)
       
def ohlc_plot_bars(Data, window):
     
    Chosen = Data[-window:, ]
    
    for i in range(len(Chosen)):
        
        plt.vlines(x = i, ymin = Chosen[i, 2], ymax = Chosen[i, 1], color = 'black', linewidth = 1)  
        plt.vlines(x = i, ymin = Chosen[i, 2], ymax = Chosen[i, 1], color = 'black', linewidth = 1)
        
        if Chosen[i, 3] > Chosen[i, 0]:
            color_chosen = 'black'
            plt.vlines(x = i, ymin = Chosen[i, 0], ymax = Chosen[i, 3], color = color_chosen, linewidth = 1.00)  

        if Chosen[i, 3] < Chosen[i, 0]:
            color_chosen = 'black'
            plt.vlines(x = i, ymin = Chosen[i, 3], ymax = Chosen[i, 0], color = color_chosen, linewidth = 1.00)  
            
        if Chosen[i, 3] == Chosen[i, 0]:
            color_chosen = 'black'
            plt.vlines(x = i, ymin = Chosen[i, 3], ymax = Chosen[i, 0], color = color_chosen, linewidth = 1.00)  
            
    plt.grid()       
   
def vertical_horizontal_indicator(Data, lookback, what, where):
    
    for i in range(len(Data)):
        Data[i, where] = Data[i, what] - Data[i - 1, what]
    
    Data = jump(Data, 1)    
       
    Data[:, where] = abs(Data[:, where])

    for i in range(len(Data)):
        Data[i, where + 1] = Data[i - lookback + 1:i + 1, where].sum()
    
    for i in range(len(Data)):
        try:
            Data[i, where + 2] = max(Data[i - lookback + 1:i + 1, what]) - min(Data[i - lookback + 1:i + 1, what])
        except ValueError:
            pass
    Data = jump(Data, lookback)    
    Data[:, where + 3] = Data[:, where + 2] / Data[:, where + 1]
    
    Data = deleter(Data, where, 3)
    
    return Data
 


def smi(Data, lookback, what, where):

    for i in range(len(Data)):
        
        try:
            Data[i, where] = min(Data[i - lookback + 1:i + 1, 2]) + max(Data[i - lookback + 1:i + 1, 1])
        
        except ValueError:
            pass
        
    Data[:, where] = Data[:, where] / 2    
            
    for i in range(len(Data)):
        
        try:
            Data[i, where + 1] = (Data[i, what] - Data[i, where]) / (min(Data[i - lookback + 1:i + 1, 2]) - max(Data[i - lookback + 1:i + 1, 1]))
        
        except ValueError:
            pass
    
    Data[:, where + 1] = Data[:, where + 1] * 100  
    Data = jump(Data, lookback)
    Data = deleter(Data, where, 1)
    
    return Data      
   
def hull_moving_average(Data, what, lookback, where):
    
    Data = lwma(Data, lookback, what)
    
    second_lookback = round((lookback / 2), 1)
    second_lookback = int(second_lookback) 
    
    Data = lwma(Data, second_lookback, what)
    
    Data = adder(Data, 1)
    Data[:, where + 2] = ((2 * Data[:, where + 1]) - Data[:, where])

    third_lookback = round(np.sqrt(lookback), 1)
    third_lookback = int(third_lookback) 

    Data = lwma(Data, third_lookback, where + 2)
    Data = deleter(Data, where, 3)

    return Data
           

   
def rsi_delta(Data, lookback, what, where):
    
    Data = rsi(Data, lookback, what, 0)
    
    for i in range(len(Data)):
        Data[i, where] = Data[i, where - 1] - Data[i - lookback, where - 1]
        
    return Data
   
def efficiency_ratio(Data, lookback, what, where):
    
    # Change from previous period
    for i in range(len(Data)):
        Data[i, where] = abs(Data[i, what] - Data[i - lookback, what])
    
    Data = jump(Data, lookback)
    
    # Sum of changes
    for i in range(len(Data)):
        Data[i, where + 1] = (Data[i - lookback + 1:i + 1, where].sum())   
    
    # Efficiency Ratio
    Data[:, where + 2] = Data[:, where] / Data[:, where + 1]
    
    Data = jump(Data, lookback)
    Data = deleter(Data, where, 2)
    
    return Data

def trix(Data, lookback, what, where):
    
    # First EMA
    Data = ema(Data, 2, lookback, what, where)
    Data = jump(Data, lookback)
    
    # Second EMA
    Data = ema(Data, 2, lookback, where, where + 1)
    Data = jump(Data, lookback)
       
    # Third EMA
    Data = ema(Data, 2, lookback, where + 1, where + 2)  
    Data = jump(Data, lookback)
    
    # TRIX
    for i in range(len(Data)):
        Data[i, where + 3] = (Data[i, where + 2] / Data[i - 1, where + 2]) - 1
   
    Data = deleter(Data, where, 3)
   
    return Data
   
def heiken_ashi(Data, opening, high, low, close, where):
    
    Data = adder(Data, 4)
    
    # Heiken-Ashi Open
    try:
        for i in range(len(Data)):
            Data[i, where] = (Data[i - 1, opening] + Data[i - 1, close]) / 2
    except:
        pass
    
    # Heiken-Ashi Close
    for i in range(len(Data)):
        Data[i, where + 3] = (Data[i, opening] + Data[i, high] + Data[i, low] + Data[i, close]) / 4
    
    
   
    # Heiken-Ashi High
    for i in range(len(Data)):    
        Data[i, where + 1] = max(Data[i, where], Data[i, where + 3], Data[i, high])
    
    
    # Heiken-Ashi Low    
    for i in range(len(Data)):    
        Data[i, where + 2] = min(Data[i, where], Data[i, where + 3], Data[i, low])      
    
    return Data
   
def ohlc_heiken_plot(Data, window, name):
    
    
    Chosen = Data[-window:, ]
    
    for i in range(len(Chosen)):
        
        plt.vlines(x = i, ymin = Chosen[i, 7], ymax = Chosen[i, 6], color = 'black', linewidth = 1)  
        plt.vlines(x = i, ymin = Chosen[i, 7], ymax = Chosen[i, 6], color = 'black', linewidth = 1)
        
        if Chosen[i, 8] > Chosen[i, 5]:
            color_chosen = 'green'
            plt.vlines(x = i, ymin = Chosen[i, 8], ymax = Chosen[i, 5], color = color_chosen, linewidth = 4)  

        if Chosen[i, 8] < Chosen[i, 5]:
            color_chosen = 'red'
            plt.vlines(x = i, ymin = Chosen[i, 8], ymax = Chosen[i, 5], color = color_chosen, linewidth = 4)  
            
        if Chosen[i, 8] == Chosen[i, 5]:
            color_chosen = 'black'
            plt.vlines(x = i, ymin = Chosen[i, 8], ymax = Chosen[i, 5], color = color_chosen, linewidth = 4)  
            
    plt.grid()
    plt.title(name)   

def fisher_transform(Data, lookback, close, where):
    
   Data = adder(Data, 1)
   
   Data = stochastic(Data, lookback, close, where)
   
   Data[:, where] = Data[:, where] / 100
   Data[:, where] = (2 * Data[:, where]) - 1
   
   for i in range(len(Data)):
       
       if Data[i, where] == 1:
           Data[i, where] = 0.999
       if Data[i, where] == -1:
           Data[i, where] = -0.999
           
   for i in range(len(Data)):
       
      Data[i, where + 1] = 0.5 * (np.log((1 + Data[i, where]) / (1 - Data[i, where])))
   
   Data = deleter(Data, where, 1)   
   
   return Data


def custom_normalization(Data, lookback, upper_bound, lower_bound, what, where):
    
    for i in range(len(Data)):
        
        try:
            Data[i, where] =(upper_bound - lower_bound) * ((Data[i, what] - min(Data[i - lookback + 1:i + 1, what])) / (max(Data[i - lookback + 1:i + 1, what]) - min(Data[i - lookback + 1:i + 1, what]))) + (lower_bound)
        
        except ValueError:
            pass
        
    return Data

def high_low_index(Data, lookback, high, low, where):
    
    for i in range(len(Data)):
        
        Data[i, where] = Data[i, high] - Data[i - lookback, low]
        
    return Data

def q_stick(Data, ema_lookback, opening, close, where):
    
    for i in range(len(Data)):
        
        Data[i, where] = Data[i, close] - Data[i, opening]
        
    Data = ema(Data, 2, ema_lookback, where, where + 1)
    Data = deleter(Data, where, 1)
    
    return Data
        
def fibonacci_retracement(Data, retracement, indicator, upper_barrier, lower_barrier, where): 

    for i in range(len(Data)):
     
        if Data[i, indicator] > lower_barrier and Data[i - 1, indicator] < lower_barrier:
            
            for a in range(i + 1, len(Data)):
                
                if Data[a, indicator] < upper_barrier and Data[a - 1, indicator] > upper_barrier:
                    
                    Data[a - 1, where] = 1 # Marking the top
                    Data[a - 1, where + 1] = (Data[a - 1, indicator] - Data[i - 1, indicator])
                    Data[a - 1, where + 1] = (Data[a - 1, where + 1] * retracement) + Data[i - 1, indicator]
                    break
                
                else:
                    continue
        else:
            continue

    for i in range(len(Data)):
     
        if Data[i, indicator] < upper_barrier and Data[i - 1, indicator] > upper_barrier:
            
            for a in range(i + 1, len(Data)):
                
                if Data[a, indicator] > lower_barrier and Data[a - 1, indicator] < lower_barrier:
                    
                    Data[a - 1, where + 2] = -1 # Marking the bottom
                    Data[a - 1, where + 3] = (Data[i - 1, indicator] - Data[a - 1, indicator])
                    Data[a - 1, where + 3] = Data[a - 1, indicator] + (Data[a - 1, where + 3] * retracement) 
                    break
                
                else:
                    continue
        else:
            continue

    for i in range(len(Data)):

        if Data[i, where] == 1:
            
            for a in range(i + 1, len(Data)):
                if Data[a, indicator] <= Data[i, where + 1]:
                    Data[a, where + 4] = 1
                    break
                else:
                    continue
        else:
            continue        

    for i in range(len(Data)):

        if Data[i, where + 2] == -1:
            
            for a in range(i + 1, len(Data)):
                if Data[a, indicator] >= Data[i, where + 3]:
                    Data[a, where + 5] = -1
                    break
                else:
                    continue
        else:
            continue   


    return Data

def roc(Data, lookback, what, where):
    
    for i in range(len(Data)):
        Data[i, where] = ((Data[i, what] - Data[i - lookback, what]) / Data[i - lookback, what]) * 100             
    
    return Data

def stc(Data, st_ema, lt_ema, stoch_lookback, what, where):
    
    
    Data = ema(Data, 2, st_ema, what, where)
    Data = ema(Data, 2, lt_ema, what, where + 1)
    
    # MACD Line
    Data[:, where + 2] = Data[:, where] - Data[:, where + 1]

    # %K
    for i in range(len(Data)):
            
            try:
                Data[i, where + 3] = 100 * (Data[i, where + 2] - min(Data[i - stoch_lookback + 1:i + 1, where + 2])) / (max(Data[i - stoch_lookback + 1:i + 1, where + 2]) - min(Data[i - stoch_lookback + 1:i + 1, where + 2]))
            
            except ValueError:
                pass
    
    # %D        
    Data = ma(Data, 3, where + 3, where + 4)
    Data = deleter(Data, 5, 4) 
    
    return Data

def cross_indicator(Data, short_lookback, long_lookback, what, where):
    
    Data = ma(Data, short_lookback, what, where)
    Data = ma(Data, long_lookback, what, where + 1)
    
    Data[:, where + 2] = Data[:, where] - Data[:, where + 1]
    
    Data = deleter(Data, where, 2)
    
    Data = jump(Data, long_lookback)
    
    return Data

def rvi(Data, lookback, opening, high, low, close, where):

    # Numerator
    for i in range(len(Data)):
        
        Data[i, where] = ((Data[i, close] - Data[i, opening]) + \
                          (2 * (Data[i, close] - Data[i - 1, opening])) + \
                           (2 * (Data[i, close] - Data[i - 2, opening])) + \
                            (Data[i, close] - Data[i - 2, opening])) / 6
    
    Data = ma(Data, lookback, where, where + 1)
    
    # Denominator
    for i in range(len(Data)):
        
        Data[i, where + 2] = ((Data[i, high] - Data[i, low]) + \
                          (2 * (Data[i, high] - Data[i - 1, low])) + \
                           (2 * (Data[i, high] - Data[i - 2, low])) + \
                            (Data[i, high] - Data[i - 2, low])) / 6
            
    Data = ma(Data, lookback, where + 2, where + 3)
    
    # RVI
    Data[:, where + 4] = Data[:, where + 1] / Data[:, where + 3]
    
    # Signal
    for i in range(len(Data)):
    
        Data[i, where + 5] = ((Data[i, where + 4]) + \
                          (2 * (Data[i - 1, where + 4])) + \
                           (2 * (Data[i - 2, where + 4])) + \
                            (Data[i - 3, where + 4])) / 6 
    
    Data = deleter(Data, where, 4)
    Data = jump(Data, lookback + 10)
    
    return Data

def demarker(Data, lookback, high, low, where):
    
    Data = adder(Data, 3)
    
    # Calculating DeMAX
    for i in range(len(Data)):
        
        if Data[i, high] > Data[i - 1, high]:
            Data[i, where] = Data[i, high] - Data[i - 1, high]
        else:
            Data[i, where] = 0
    
    # Calculating the Moving Average on DeMAX
    Data = ma(Data, lookback, where, where + 1)        
            
    # Calculating DeMIN
    for i in range(len(Data)):
        
        if Data[i - 1, low] > Data[i, low]:
            Data[i, where + 2] = Data[i - 1, low] - Data[i, low]
        else:
            Data[i, where + 2] = 0    
    
    # Calculating the Moving Average on DeMIN
    Data = ma(Data, lookback, where + 2, where + 3)        
   
    
    # Calculating DeMarker
    for i in range(len(Data)):
        
        Data[i, where + 4] = Data[i, where + 1] / (Data[i, where + 1] + Data[i, where + 3]) 
    
    # Removing Excess Columns
    Data = deleter(Data, where, 4)
    
    return Data

def pendulum_indicator(Data, lookback, lookback_ma, what, where):
    
    # Range
    for i in range(len(Data)):
        
        Data[i, where] = Data[i, what] - Data[i - lookback, what]
        
    # Volatility
    Data = volatility(Data, lookback, what, where + 1)
    
    # Pendulum Ratio
    Data[:, where + 2] = Data[:, where] / Data[:, where + 1]
    
    # Pendulum Indicator
    Data = ma(Data, lookback_ma, where + 2, where + 3)
    
    # Removing Excess Columns
    Data = deleter(Data, where, 3)
    
    return Data

def awesome_oscillator(Data, high, low, long_ma, short_ma, where):
    
    # Adding columns
    Data = adder(Data, 10)
    
    # Mid-point Calculation
    Data[:, where] = (Data[:, high] + Data[:, low]) / 2
    
    # Calculating the short-term Simple Moving Average
    Data = ma(Data, short_ma, where, where + 1)
    
    # Calculating the long-term Simple Moving Average
    Data = ma(Data, long_ma, where, where + 2)
    
    # Calculating the Awesome Oscillator
    Data[:, where + 3] = Data[:, where + 1] - Data[:, where + 2]

    # Removing Excess Columns/Rows
    Data = jump(Data, long_ma)  
    Data = deleter(Data, where, 3)   
    
    return Data

def indicator_plot_double_awesome(Data, first, second, name = '', name_ind = '', window = 250):

    fig, ax = plt.subplots(2, figsize = (10, 5))

    Chosen = Data[-window:, ]
    
    for i in range(len(Chosen)):
        
        ax[0].vlines(x = i, ymin = Chosen[i, 2], ymax = Chosen[i, 1], color = 'black', linewidth = 1)  
   
    ax[0].grid() 

    for i in range(len(Chosen)):
        
        if Chosen[i, 5] > Chosen[i - 1, 5]:
            ax[1].vlines(x = i, ymin = 0, ymax = Chosen[i, 5], color = 'green', linewidth = 1)  
        
        if Chosen[i, 5] < Chosen[i - 1, 5]:
            ax[1].vlines(x = i, ymin = Chosen[i, 5], ymax = 0, color = 'red', linewidth = 1)  
            
    ax[1].grid() 
    ax[1].axhline(y = 0, color = 'black', linewidth = 0.5, linestyle = '--')

def indicator_plot_double_macd(Data, first, second, name = '', name_ind = '', window = 250):

    fig, ax = plt.subplots(2, figsize = (10, 5))

    Chosen = Data[-window:, ]
    
    for i in range(len(Chosen)):
        
        ax[0].vlines(x = i, ymin = Chosen[i, 2], ymax = Chosen[i, 1], color = 'black', linewidth = 1)  
   
    ax[0].grid() 
    ax[0].plot(Data[-window:, 6])

    for i in range(len(Chosen)):
        
        if Chosen[i, 5] > 0:
            ax[1].vlines(x = i, ymin = 0, ymax = Chosen[i, second], color = 'green', linewidth = 1)  
        
        if Chosen[i, 5] < 0:
            ax[1].vlines(x = i, ymin = Chosen[i, second], ymax = 0, color = 'red', linewidth = 1)  

        if Chosen[i, 5] == 0:
            ax[1].vlines(x = i, ymin = Chosen[i, second], ymax = 0, color = 'black', linewidth = 1)  
            
    ax[1].grid() 
    ax[1].axhline(y = 0, color = 'black', linewidth = 0.5, linestyle = '--')
    
def ssl(Data, lookback, what, high, low, where):
    
    # Calculating the High-MA
    Data = ma(Data, lookback, high, where)
    
    # Calculating the Low-Ma
    Data = ma(Data, lookback, low, where + 1)
    
    # Combining into one Column
    for i in range(len(Data)):
        
        if Data[i, what] < Data[i, where]:
            Data[i, where + 2] = Data[i, where]
    
        if Data[i, what] > Data[i, where + 1]:
            Data[i, where + 2] = Data[i, where + 1]            

    # Removing Excess Columns/Rows
    Data = jump(Data, lookback)
    Data = deleter(Data, where, 2)
            
    return Data

def stochastic_smoothing_oscillator(Data, high, low, close, lookback, where):
    
    # Adding columns
    Data = adder(Data, 4)
    
    Data = ema(Data, 2, 2, high, where)
    Data = ema(Data, 2, 2, low, where + 1)
    Data = ema(Data, 2, 2, close, where + 2)
    
    for i in range(len(Data)):
        
        try:
            Data[i, where + 3] = (Data[i, where + 2] - min(Data[i - lookback + 1:i + 1, where + 1])) / (max(Data[i - lookback + 1:i + 1, where]) - min(Data[i - lookback + 1:i + 1, where + 1]))
        
        except ValueError:
            pass
    
    Data[:, where + 3] = Data[:, where + 3] * 100
    Data = deleter(Data, where, 3)
    
    Data = jump(Data, lookback)
    
    return Data


def hurst(Data, lookback, what, where):

    for i in range(len(Data)):
        try:
            new = Data[i - lookback:i, what]
            
            Data[i, where] = compute_Hc(Data[i - lookback:i + 1, what])[0]
        except ValueError:
            pass
        
    return Data

def fractal_dimension_index(Data, lookback, what, where):

    for i in range(len(Data)):
        try:
            new = Data[i - lookback:i, what]
            
            Data[i, where] = compute_Hc(Data[i - lookback:i + 1, what])[0]
            Data[i, where] = 2 - Data[i, where]
            
        except ValueError:
            pass
        
    return Data

def kairi_indicator(Data, lookback, what, where):
    
    Data = ma(Data, lookback, what, where)
    
    for i in range(len(Data)):
        
        Data[i, where + 1] = (Data[i, what] - Data[i, where]) / Data[i, where]
    
    Data = deleter(Data, where, 1)
    
    return Data
    
def pure_pupil_bands(Data, boll_lookback, standard_distance, what, high, low, where):
       
    # Calculating means
    ma(Data, boll_lookback, what, where)

    Data = pure_pupil(Data, boll_lookback, high, low, where + 1)
    
    Data[:, where + 2] = Data[:, where] + (standard_distance * Data[:, where + 1])
    Data[:, where + 3] = Data[:, where] - (standard_distance * Data[:, where + 1])
    
    Data = jump(Data, boll_lookback)
    
    Data = deleter(Data, where, 2)
        
    return Data

def ultimate_oscillator(Data, high, low, close, where):

    Data = adder(Data, 7)

    # Buying pressure
    for i in range(len(Data)):
        
        Data[i, where] = Data[i, close] - min(Data[i, low], Data[i - 1, close])
    
    # True range
    for i in range(len(Data)):
        
        Data[i, where + 1] = max(Data[i, high], Data[i - 1, close]) - min(Data[i, low], Data[i - 1, close]) 
        if Data[i, where + 1] == 0:
            Data[i, where + 1] = 0.01
    
    # BP / TR
    Data[:, where + 2] = Data[:, where] / Data[:, where + 1]
    
    # A5
    Data = ema(Data, 2, 5, where + 2, where + 3)
    
    # A13
    Data = ema(Data, 2, 13, where + 2, where + 4)  
    
    # A21
    Data = ema(Data, 2, 21, where + 2, where + 5)
      
    # Ultimate Oscillator
    for i in range(len(Data)):
        
        Data[i, where + 6] = (Data[i, where + 3] * 4) +  (Data[i, where + 4] * 2) + (Data[i, where + 5])
        Data[i, where + 6] = (Data[i, where + 6] / 7) * 100
        
    Data = deleter(Data, where, 6)
    
    return Data 

def trend_intensity_indicator(Data, lookback, what, where):


    Data = adder(Data, 5)
    
    # Calculating the Moving Average
    Data = ma(Data, lookback, what, where)

    # Deviations
    for i in range(len(Data)):
        
        if Data[i, what] > Data[i, where]:
            Data[i, where + 1] = Data[i, what] - Data[i, where]
        
        if Data[i, what] < Data[i, where]:
            Data[i, where + 2] = Data[i, where] - Data[i, what]
            
        
    # Trend Intensity Index
    for i in range(len(Data)):
            
        Data[i, where + 3] = np.count_nonzero(Data[i - lookback + 1:i + 1, where + 1])
            
    for i in range(len(Data)):
            
        Data[i, where + 4] = np.count_nonzero(Data[i - lookback + 1:i + 1, where + 2])
        
    for i in range(len(Data)):
        
        Data[i, where + 5] = ((Data[i, where + 3]) / (Data[i, where + 3] + Data[i, where + 4])) * 100
        
    Data = deleter(Data, where, 5)
     
    return Data
        
def envelopes(Data, lookback, deviation, what, where):


    # Calculating the moving average
    Data = ma(Data, lookback, what, where) 

    # Upper Envelope
    Data[:, where + 1] = Data[:, where] + (Data[:, where] * deviation)

    # Lower Envelope       
    Data[:, where + 2] = Data[:, where] - (Data[:, where] * deviation)
    
    return Data
        
def percentile_range_indicator(Data, lookback, what, where):

    for i in range(len(Data)):    
   
        Data[i, where] = stats.percentileofscore(Data[i - lookback + 1:i, what], Data[i, what])
    
    return Data        
    
def hull_bands(Data, lookback, standard_distance, what, where):
       
    # Calculating means
    Data = hull_moving_average(Data, what, lookback, where)
    
    Data = adder(Data, 3)

    volatility(Data, lookback, what, where + 1)
    
    Data[:, where + 2] = Data[:, where] + (standard_distance * Data[:, where + 1])
    Data[:, where + 3] = Data[:, where] - (standard_distance * Data[:, where + 1])
    
    Data = jump(Data, lookback)
    
    Data = deleter(Data, where, 2)
        
    return Data

def fibonacci_rsi(Data, high, low, close, where):
    
    # Taking price differences
    Data = adder(Data, 6)
    
    for i in range(len(Data)):
        
        Data[i, where] = Data[i, high] - Data[i - 1, high]
        
    for i in range(len(Data)):
        
        Data[i, where + 1] = Data[i, low] - Data[i - 1, low]
        
    Data[:, where + 2] = 0.5 * (Data[:, where] + Data[:, where + 1])
    
    Data = deleter(Data, where, 2)
    
    # Finding up and down columns
    for i in range(len(Data)):
        
        if Data[i, where] > 0:
            Data[i, where + 1] = Data[i, where]
        elif Data[i, where] < 0:
            Data[i, where + 1] = 0

        if Data[i, where] < 0:
            Data[i, where + 2] = Data[i, where]
        elif Data[i, where] > 0:
            Data[i, where + 2] = 0    
            
    Data[:, where + 2] = abs(Data[:, where + 2])
    
    # Taking the Fibonacci Moving Average
    for i in range(3, 15):
            
            Data = adder(Data, 1)
            lookback = fib(i)
            Data = ma(Data, lookback, where + 1, -1)
        
    Data = adder(Data, 1)
    Data = jump(Data, lookback)
    
    for i in range(len(Data)):
        Data[i, -1] = np.sum(Data[i, where + 3:where + 15 - 3])
        Data[i, -1] = Data[i, - 1] / (15 - 3)
    
    Data = deleter(Data, where + 3, 15 - 3)
        

    for i in range(3, 15):
            
            Data = adder(Data, 1)
            lookback = fib(i)
            Data = ma(Data, lookback, where + 2, -1)
        
    Data = adder(Data, 1)
    Data = jump(Data, lookback)
    
    for i in range(len(Data)):
        Data[i, -1] = np.sum(Data[i, where + 4:where + 15 - 3])
        Data[i, -1] = Data[i, - 1] / (15 - 3)
    
    Data = deleter(Data, where + 4, 15 - 3)
    
    # Calculating the Fibonacci RSI
    Data = adder(Data, 2)
    
    Data[:, where + 5] = Data[:, where + 3] / Data[:, where + 4]
    
    for i in range(len(Data)):
        
        Data[i, where + 6] = (100.0 - (100.0 / (1.0 + Data[i, where + 5])))
    
    Data = deleter(Data, where, 6)
    
    return Data

def frama(Data, lookback, what, where, fdi_column):

    # First Exponential Moving Average
    Data = ema(Data, 2, lookback, what, where)
    Data = jump(Data, lookback + 100)

    # Keeping the first Exponential Moving Average in the column
    Data[1:, where] = 0    
    
    # Calculating FRAMA
    for i in range(1, len(Data)):
        
        a = np.exp(-4.6 * (Data[i, fdi_column] - 1))       
        Data[i, where] = a * Data[i, what] + ((1 - a) * Data[i - 1, where])
    
    return Data

def fractal_volatility_bands(Data, boll_lookback, standard_distance, what, frama_col, where):
       
    # Calculating Volatility
    volatility(Data, boll_lookback, what, where)
    
    Data[:, where + 1] = Data[:, frama_col] + (standard_distance * Data[:, where])
    Data[:, where + 2] = Data[:, frama_col] - (standard_distance * Data[:, where])
    
    Data = jump(Data, boll_lookback)
    
    Data = deleter(Data, where, 1)
        
    return Data    
    
def adaptive_rsi(Data, lookback):
    
    # Adding columns
    Data = adder(Data, 20)
    
    for i in range(len(Data)):
        
        Data[i, 5] = Data[i, 3] - Data[i - 1, 3]
    
    # Finding up and down columns
    for i in range(len(Data)):
        
        if Data[i, 5] > 0:
            Data[i, 6] = Data[i, 5]
        elif Data[i, 5] < 0:
            Data[i, 6] = 0

    Data = Data[1:, ]  
    Data = deleter(Data, 5, 1)

    # lookback from previous period
    for i in range(len(Data)):
        Data[i, 6] = abs(Data[i, 5] - Data[i - 1, 5])
    
    Data[0, 6] = 0
    
    # Sum of lookbacks
    for i in range(len(Data)):
        Data[i, 7] = (Data[i - lookback + 1:i + 1, 6].sum())   
        if Data[i, 7] == 0:
            Data[i, 7] = Data[i - 1, 7]
            
    # Volatility    
    for i in range(len(Data)):
        Data[i, 8] = abs(Data[i, 5] - Data[i - lookback, 5])
        if Data[i, 8] == 0:
            Data[i, 8] = Data[i - 1, 8]
        
    Data = Data[lookback + 1:, ]
    
    # Efficiency Ratio
    Data[:, 9] = Data[:, 8] / Data[:, 7]
    
    for i in range(len(Data)):
        Data[i, 10] = np.square(Data[i, 9] * 0.6666666666666666667)
        
    for i in range(len(Data)):
        Data[i, 11] = Data[i - 1, 11] + (Data[i, 10] * (Data[i, 5] - Data[i - 1, 11]))
    
    Data[0, 11] = Data[0, 10]
        
    Data = deleter(Data, 5, 6)

    for i in range(len(Data)):
        
        Data[i, 6] = Data[i, 3] - Data[i - 1, 3]
    
    for i in range(len(Data)):

        if Data[i, 6] < 0:
            Data[i, 7] = Data[i, 6]
        elif Data[i, 6] > 0:
            Data[i, 7] = 0    
            
    Data[:, 7] = abs(Data[:, 7])    
    
    Data = Data[1:, ]  
    Data = deleter(Data, 6, 1)

    # lookback from previous period
    for i in range(len(Data)):
        Data[i, 7] = abs(Data[i, 6] - Data[i - 1, 6])
    
    Data[0, 7] = 0
    
    # Sum of lookbacks
    for i in range(len(Data)):
        Data[i, 8] = (Data[i - lookback + 1:i + 1, 7].sum())   
        if Data[i, 8] == 0:
            Data[i, 8] = Data[i - 1, 8]  
            
    # Volatility    
    for i in range(len(Data)):
        Data[i, 9] = abs(Data[i, 6] - Data[i - lookback, 6])
        if Data[i, 9] == 0:
            Data[i, 9] = Data[i - 1, 9]
            
    Data = Data[lookback + 1:, ]
    
    # Efficiency Ratio
    Data[:, 10] = Data[:, 9] / Data[:, 8]
    
    for i in range(len(Data)):
        Data[i, 11] = np.square(Data[i, 10] * 0.6666666666666666667)
        
    for i in range(len(Data)):
        Data[i, 12] = Data[i - 1, 12] + (Data[i, 11] * (Data[i, 6] - Data[i - 1, 12]))

    Data[0, 12] = Data[0, 11]
        
    Data = deleter(Data, 6, 6)    
    
    # Calculating the Adaptive RSI
    Data = jump(Data, lookback * 4)
    Data = adder(Data, 2)
    
    Data[:, 7] = Data[:, 5] / Data[:, 6]
    
    for i in range(len(Data)):
        
        Data[i, 8] = (100.0 - (100.0 / (1.0 + Data[i, 7])))
    
    Data = deleter(Data, 5, 3)
    
    return Data

def hull_rsi(Data, lookback, close, where):

    # Taking price differences
    Data = adder(Data, 3)
    
    for i in range(len(Data)):
        
        Data[i, where] = Data[i, close] - Data[i - 1, close]
    
    Data = Data[1:, ]
    
    # Finding up and down columns
    for i in range(len(Data)):
        
        if Data[i, where] > 0:
            Data[i, where + 1] = Data[i, where]
        elif Data[i, where] < 0:
            Data[i, where + 1] = 0

        if Data[i, where] < 0:
            Data[i, where + 2] = Data[i, where]
        elif Data[i, where] > 0:
            Data[i, where + 2] = 0    
            
    Data[:, where + 2] = abs(Data[:, where + 2])
    
    # Taking the Hull Moving Average of UP Periods
    Data = lwma(Data, lookback, 6)
    
    second_lookback = round((lookback / 2), 1)
    second_lookback = int(second_lookback) 
    
    Data = lwma(Data, second_lookback, 6)
    
    Data = adder(Data, 1)
    Data[:, 10] = ((2 * Data[:, 9]) - Data[:, 6])

    third_lookback = round(np.sqrt(lookback), 1)
    third_lookback = int(third_lookback) 

    Data = lwma(Data, third_lookback, 10)
    Data[:, 11] = abs(Data[:, 11])
    
    for i in range(len(Data)):
        
        if Data[i, 11] == 0:
            Data[i, 11] = Data[i - 1, 11]
            
    Data = deleter(Data, 5, 1)
    Data = deleter(Data, 7, 3)
    
    # Taking the Hull Moving Average of DOWN Periods
    Data = lwma(Data, lookback, 6)
    
    second_lookback = round((lookback / 2), 1)
    second_lookback = int(second_lookback) 
    
    Data = lwma(Data, second_lookback, 6)
    
    Data = adder(Data, 1)
    Data[:, 10] = ((2 * Data[:, 9]) - Data[:, 8])

    third_lookback = round(np.sqrt(lookback), 1)
    third_lookback = int(third_lookback) 

    Data = lwma(Data, third_lookback, 10)
    Data[:, 11] = abs(Data[:, 11])

    for i in range(len(Data)):
        
        if Data[i, 11] == 0:
            Data[i, 11] = Data[i - 1, 11]
    
    Data = deleter(Data, 8, 3)   
    Data = deleter(Data, 5, 2)
    
    # Calculating the Hull RSI
    Data = adder(Data, 2)
    
    Data[:, 7] = Data[:, 5] / Data[:, 6]
    
    for i in range(len(Data)):
        
        Data[i, 8] = (100.0 - (100.0 / (1.0 + Data[i, 7])))
    
    Data = deleter(Data, 5, 3)
    
    return Data

def signal_chart(Data, close, what_bull, what_bear, window = 500):   
     
    Plottable = Data[-window:, ]
    
    fig, ax = plt.subplots(figsize = (10, 5))
    
    ohlc_plot_candles(Data, window)    

    for i in range(len(Plottable)):
        
        if Plottable[i, what_bull] == 1:
            
            x = i
            y = Plottable[i, close]
        
            ax.annotate(' ', xy = (x, y), 
                        arrowprops = dict(width = 9, headlength = 11, headwidth = 11, facecolor = 'green', color = 'green'))
        
        elif Plottable[i, what_bear] == -1:
            
            x = i
            y = Plottable[i, close]
        
            ax.annotate(' ', xy = (x, y), 
                        arrowprops = dict(width = 9, headlength = -11, headwidth = -11, facecolor = 'red', color = 'red'))

def signal_chart_k(Data, close, what_bull, what_bear, window = 500):   
     
    Plottable = Data[-window:, ]
    
    fig, ax = plt.subplots(figsize = (10, 5))
    
    ohlc_plot_candles_k(Data, 4, 5, 6, 7, window)   

    for i in range(len(Plottable)):
        
        if Plottable[i, what_bull] == 1:
            
            x = i
            y = Plottable[i, close]
        
            ax.annotate(' ', xy = (x, y), 
                        arrowprops = dict(width = 9, headlength = 11, headwidth = 11, facecolor = 'green', color = 'green'))
        
        elif Plottable[i, what_bear] == -1:
            
            x = i
            y = Plottable[i, close]
        
            ax.annotate(' ', xy = (x, y), 
                        arrowprops = dict(width = 9, headlength = -11, headwidth = -11, facecolor = 'red', color = 'red'))


def signal_chart_bars(Data, close, what_bull, what_bear, window = 500):   
     
    Plottable = Data[-window:, ]
    
    fig, ax = plt.subplots(figsize = (10, 5))
    
    ohlc_plot_bars(Data, window)    

    for i in range(len(Plottable)):
        
        if Plottable[i, what_bull] == 1:
            
            x = i
            y = Plottable[i, close]
        
            ax.annotate(' ', xy = (x, y), 
                        arrowprops = dict(width = 9, headlength = 11, headwidth = 11, facecolor = 'green', color = 'green'))
        
        elif Plottable[i, what_bear] == -1:
            
            x = i
            y = Plottable[i, close]
        
            ax.annotate(' ', xy = (x, y), 
                        arrowprops = dict(width = 9, headlength = -11, headwidth = -11, facecolor = 'red', color = 'red'))  

def fibonacci_bands(Data, fibonacci_column, what, where):
    
    # Calculating Volatility
    volatility(Data, boll_lookback, what, where)
    
    Data[:, where + 1] = Data[:, fibonacci_column] + (standard_distance * Data[:, where])
    Data[:, where + 2] = Data[:, fibonacci_column] - (standard_distance * Data[:, where])
    
    Data = jump(Data, boll_lookback)
    
    Data = deleter(Data, where, 1)

    return Data    


def time_up(Data, width, what, where):
    
    # Adding the required columns
    Data = adder(Data, 4)

    # Calculating the difference in prices
    for i in range(len(Data)):
        
        Data[i, where] = Data[i, what] - Data[i - width, what]
        
    # Upward Timing    
    for i in range(len(Data)):
        
        Data[0, where + 1] = 1
        
        if Data[i, where] > 0:
            Data[i, where + 1] = Data[i - width, where + 1] + 1
            
        else:
            Data[i, where + 1] = 0
            
    # Downward Timing    
    for i in range(len(Data)):
        
        Data[0, where + 2] = 1
        
        if Data[i, where] < 0:
            Data[i, where + 2] = Data[i - width, where + 2] + 1
            
        else:
            Data[i, where + 2] = 0

    # Changing signs
    Data[:, where + 2] = -1 * Data[:, where + 2]
    
    # Time's Up Indicator
    Data[:, where + 3] = Data[:, where + 1] + Data[:, where + 2]
    
    # Cleaning rows/columns
    Data = deleter(Data, where, 3)
    
    return Data



def double_differencing_indicator(Data, lag, what, where):

    # Adding columns
    Data = adder(Data, 2)
    
    for i in range(len(Data)):
        Data[i, where] = (Data[i, what] - Data[i - lag, what])

    for i in range(len(Data)):
        Data[i, where + 1] = (Data[i, where] - Data[i - lag, where])
        
    Data = deleter(Data, where, 1)
    Data = jump(Data, lag * 2)

    return Data

def decycler(Data, lookback, what, where):
    
    # Adding a column
    Data = adder(Data, 3)
    
    # Defining Pi
    pi = np.pi

    # Defining AlphaArg
    AlphaArg = (2 * pi) / (lookback * np.sqrt(2))

    # Defining Alpha
    if np.cos(AlphaArg) != 0:
        alpha = (np.cos(AlphaArg) + np.sin(AlphaArg) - 1) / np.cos(AlphaArg)

    # Calculating HP
    # square(1 - (alpha / 2)) * (src - 2 * src[1] + src[2]) + 2 * (1 - alpha) * hp[1] - square(1 - alpha) * hp[2]
    for i in range(len(Data)):
        
        Data[0, where] = 0
        Data[i, where] = np.sqrt(1 - (alpha / 2)) * (Data[i, what] - 2 * Data[i - 1, what] + Data[i - 2, what]) + 2 * (1 - alpha) * Data[i - 1, where] - np.sqrt(1 - alpha) * Data[i - 2, where]    
    
    # Decycler
    Data[:, where + 1] = Data[:, what] - Data[:, where] 
    
    # Normalizing the Decycler
    Data = normalizer(Data, 100, where + 1, where + 2)
        
    # Deleting column
    Data = deleter(Data, where, 2)               
        
    return Data

def outstretched_indicator(Data, lookback, lookback_ma, close, where):
    
    # Adding a few columns
    Data = adder(Data, 6)    
    
    # Calculating the Stretch
    for i in range(len(Data)):
        
        # Positive Stretch
        if Data[i, close] > Data[i - lookback, close]:
            Data[i, where] = 1
            
        # Negative Stretch
        else:
            Data[i, where + 1] = 1 

    # Positive Stretch Summation
    for i in range(len(Data)):
        
        Data[i, where + 2] = Data[i - lookback + 1:i + 1, where].sum()    

    # Negative Stretch Summation 
    for i in range(len(Data)):
        
        Data[i, where + 3] = Data[i - lookback + 1:i + 1, where + 1].sum()    
    
    # Calculating the Raw Outstretch
    Data[:, where + 4] = Data[:, where + 2] - Data[:, where + 3]
    
    # Calculating the Outstretched Indicator
    Data = ema(Data, 2, lookback_ma, where + 4, where + 5)
    
    # Deleting columns
    Data = deleter(Data, where, 5)
    
    return Data

def heiken_ashi_plot(Data, first, second, name = '', name_ind = '', window = 250):
    
    fig, ax = plt.subplots(2, figsize = (10, 5))

    Chosen = Data[-window:, ]
    
    for i in range(len(Chosen)):
        
        ax[0].vlines(x = i, ymin = Chosen[i, 2], ymax = Chosen[i, 1], color = 'black', linewidth = 1)  
   
    ax[0].grid() 

    for i in range(len(Chosen)):
        
        if Chosen[i, 8] > Chosen[i, 5]:
            ax[1].plot(Data[i, 8], color = 'green', linewidth = 1)  
        
        if Chosen[i, 8] < Chosen[i, 5]:
            ax[1].plot(Data[i, 8], color = 'red', linewidth = 1)  

        if Chosen[i, 8] == Chosen[i, 5]:
            ax[1].plot(Data[i, 8], color = 'black', linewidth = 1)    
            
    ax[1].grid() 
    ax[1].axhline(y = 0, color = 'black', linewidth = 0.5, linestyle = '--')
    

def td_rei(Data, high, low, where):
    
    # Adding a few columns
    Data = adder(Data, 4)
    
    # Calculating the High-Low range
    for i in range(len(Data)):
        
        Data[i, where] = (Data[i, high] - Data[i - 2, high]) + (Data[i, low] - Data[i - 2, low]) 
        
    # Defining the first condition
    for i in range(len(Data)):
        
        if Data[i, high] < Data[i - 5, low] or Data[i - 2, high] < Data[i - 7, 3]:
            Data[i, where] = 0
            
    # Defining the second condition
    for i in range(len(Data)):
        
        if Data[i, low] > Data[i - 5, high] or Data[i - 2, low] > Data[i - 7, 3]:
            Data[i, where] = 0
            
    for i in range(len(Data)):   

        if Data[i, where] != 0:
            
            Data[i, where] = Data[i, high] - Data[i, low]
            
    # Calculating the sum of the High-Low range
    for i in range(len(Data)):    
        
        Data[i, where + 1] = Data[i - 5 + 1:i + 1, where].sum()     
       
    # Determining the Absolute range
    for i in range(len(Data)):    
        try:
            
            Data[i, where + 2] = (max(Data[i - 5 + 1:i + 1, 1]) - min(Data[i - 5 + 1:i + 1, 2]))    
            
        except ValueError:
            pass
    
    # Calculating the TD Range Expansion Index
    for i in range(len(Data)):    
        
        Data[i, where + 3] = (Data[i, where + 1] / Data[i, where + 2]) * 100
    
    # Cleaning
    Data = deleter(Data, where, 3)
    
    Data = jump(Data, 5)
    
    return Data

def fibonacci_timing_pattern(Data, count, step, step_two, step_three, close, buy, sell):
   
    # Adding a few columns
    Data = adder(Data, 10)
    
    # Bullish Fibonacci Timing Pattern
    counter = -1
    for i in range(len(Data)):    
        if Data[i, close] < Data[i - step, close] and \
           Data[i, close] < Data[i - step_two, close] and \
           Data[i, close] < Data[i - step_three, close]:
               
            Data[i, buy] = counter
            counter += -1 
            
            if counter == -count - 1:
                counter = 0
            else:
                continue   
            
        elif Data[i, close] >= Data[i - step, close]:
            counter = -1 
            Data[i, buy] = 0 
        
    # Bearish Fibonacci Timing Pattern
    counter = 1 
    
    for i in range(len(Data)):
        if Data[i, close] > Data[i - step, close] and \
           Data[i, close] > Data[i - step_two, close] and \
           Data[i, close] > Data[i - step_three, close]: 
               
            Data[i, sell] = counter 
            counter += 1        
            if counter == count + 1: 
                counter = 0            
            else:
                continue   
            
        elif Data[i, close] <= Data[i - step, close]: 
            counter = 1 
            Data[i, sell] = 0 
      
    return Data

def plot_signal_equity_curve(Data, equity_curve, what_bull, what_bear, window = 250):
    
    fig, ax = plt.subplots(2, figsize = (10, 5))

    Chosen = Data[-window:, ]
    
    for i in range(len(Chosen)):
        
        ax[0].vlines(x = i, ymin = Chosen[i, 2], ymax = Chosen[i, 1], color = 'black', linewidth = 1)  
        
        if Chosen[i, 3] > Chosen[i, 0]:
            color_chosen = 'grey'
            ax[0].vlines(x = i, ymin = Chosen[i, 0], ymax = Chosen[i, 3], color = color_chosen, linewidth = 2)  

        if Chosen[i, 3] < Chosen[i, 0]:
            color_chosen = 'black'
            ax[0].vlines(x = i, ymin = Chosen[i, 3], ymax = Chosen[i, 0], color = color_chosen, linewidth = 2)  
            
        if Chosen[i, 3] == Chosen[i, 0]:
            color_chosen = 'black'
            ax[0].vlines(x = i, ymin = Chosen[i, 3], ymax = Chosen[i, 0], color = color_chosen, linewidth = 2)  
            
    for i in range(len(Chosen)):
        
        if Chosen[i, what_bull] == 1:
            
            x = i
            y = Chosen[i, 3]
        
            ax[0].annotate(' ', xy = (x, y), 
                        arrowprops = dict(width = 9, headlength = 11, headwidth = 11, facecolor = 'green', color = 'green'))
        
        elif Chosen[i, what_bear] == -1:
            
            x = i
            y = Chosen[i, 3]
        
            ax[0].annotate(' ', xy = (x, y), 
                        arrowprops = dict(width = 9, headlength = -11, headwidth = -11, facecolor = 'red', color = 'red'))
                         
    ax[0].grid()  
    
    ax[1].plot(equity_curve[-window:, 3])
    ax[1].grid()

        
def detrended_price_oscillator(Data, lookback, what, where):
    
    # Calculating the Simple Moving Average
    Data = ma(Data, lookback, what, where)
    
    # Defining the Detrended Lookback Period
    detrended_period = (lookback / 2) + 1
    detrended_period = int(detrended_period)
    
    # Calculating the Detrended Price Oscillator
    for i in range(len(Data)):  
        
        Data[i, where + 1] = Data[i, where] - Data[i - detrended_period, what]
        
    # Cleaning up
    Data = deleter(Data, where, 1)
    Data = jump(Data, lookback)
    
    return Data

def correlation_wise_reversal_indicator(Data, lookback, close, where, threshold = 0.75):
    
    # Adding a few columns
    Data = adder(Data, 8)
    
    # Average of current close minus the previous period 
    for i in range(len(Data)):
        
        Data[i, where] = Data[i, close] - Data[i - 1, close]

    # Average of current close minus n then 
    for i in range(len(Data)):
        
        Data[i, where + 1] = Data[i, close] - Data[i - 2, close]

    # Average of current close minus the close 2 periods ago 
    for i in range(len(Data)):
        
        Data[i, where + 2] = Data[i, close] - Data[i - 3, close]

    # Average of current close minus the close 3 periods ago
    for i in range(len(Data)):
        
        Data[i, where + 3] = Data[i, close] - Data[i - 4, close]

    # Average of current close minus close 4 periods ago
    for i in range(len(Data)):
        
        Data[i, where + 4] = Data[i, close] - Data[i - 5, close]

    # Average of current close minus close 5 periods ago 
    for i in range(len(Data)):
        
        Data[i, where + 5] = Data[i, close] - Data[i - 6, close]

    # Calculating the average mean-reversion
    Data[:, where + 6] = (Data[:, where] + Data[:, where + 1] + Data[:, where + 2] + Data[:, where + 3] + Data[:, where + 4] + Data[:, where + 5]) / 6
    
    # Cleaning
    Data = deleter(Data, where, 6)
    
    # Adjusting for correlation
    Data = rolling_correlation(Data, close, where, lookback, where + 1)
    
    for i in range(len(Data)):
        
        if Data[i, where + 1] > threshold:
                
            Data[i, where] = Data[i, where]
        
        elif Data[i, where + 1] < threshold:
         
            Data[i, where] = 0
            
    # Cleaning
    Data = deleter(Data, where + 1, 1)
    
    return Data

def chande_trendscore(Data, close, where):
    
    # Adding a column
    Data = adder(Data, 2)
    
    # Calculating the TrendScore
    for i in range(len(Data)):
        
        if Data[i, close] < Data[i - 1, close]:
            
            Data[i, where] = -1
            
        elif Data[i, close] > Data[i - 1, close]:
            
            Data[i, where] = 1
            
    # Cumulative Score
    for i in range(len(Data)):

        Data[i, where + 1] = Data[i - 20 + 1:i + 1, where].sum()     
        
    # Cleaning
    Data = deleter(Data, where, 1)
        
    return Data

def time_spent_above_below_mean(Data, lookback, close, where):
    
    # Adding the required columns
    Data = adder(Data, 4)
    
    # Calculating the moving average
    Data = ma(Data, lookback, close, where)
        
    # Time Spent Above the Mean    
    for i in range(len(Data)):
        
        Data[0, where + 1] = 1
        
        if Data[i, close] > Data[i, where]:
            Data[i, where + 1] = Data[i - 1, where + 1] + 1
            
        else:
            Data[i, where + 1] = 0

    # Time Spent Below the Mean   
    for i in range(len(Data)):
        
        Data[0, where + 2] = -1
        
        if Data[i, close] < Data[i, where]:
            Data[i, where + 2] = Data[i - 1, where + 2] - 1
            
        else:
            Data[i, where + 2] = 0
            
    # Time Spent Below/Above Mean
    Data[:, where + 3] = Data[:, where + 1] + Data[:, where + 2]
    
    # Cleaning
    Data = deleter(Data, where, 3)
            
    return Data

def super_rsi(Data, lookback, close, where, amortization_factor, genre = 'Smoothed'):
    
    # Adding a few columns
    Data = adder(Data, 9)
    
    # Calculating Differences
    for i in range(len(Data)):
        
        Data[i, where] = Data[i, close] - Data[i - 1, close]
    
    Data = jump(Data, 1)
    
    # Calculating the Up and Down absolute values
    for i in range(len(Data)):
        
        if Data[i, where] > 0:
            
            Data[i, where + 1] = Data[i, where]
            
        elif Data[i, where] < 0:
            
            Data[i, where + 2] = abs(Data[i, where])
            
    # Incorporating Amortization as a filter      
    for i in range(1, len(Data)):
              
        if Data[i - 1, where + 1] > 0:
            
            Data[i, where + 3] = Data[i - 1, where + 1] * amortization_factor
                    
        if Data[i - 1, where + 2] > 0:
            
            Data[i, where + 4] = Data[i - 1, where + 2] * amortization_factor
                  
    # Calculating the Smoothed Moving Average on Up and Down absolute values    
    if genre == 'Smoothed':
        
        lookback = (lookback * 2) - 1 # From exponential to smoothed
        
        Data = ema(Data, 2, lookback, where + 3, where + 5)
        Data = ema(Data, 2, lookback, where + 4, where + 6)
    
    if genre == 'Simple':
        
        Data = ma(Data, lookback, where + 3, where + 5)
        Data = ma(Data, lookback, where + 4, where + 6)
    
    # Calculating the Relative Strength
    Data[:, where + 7] = Data[:, where + 5] / Data[:, where + 6]
    
    # Calculate the Relative Strength Index
    Data[:, where + 8] = (100 - (100 / (1 + Data[:, where + 7])))

    # Cleaning
    Data = deleter(Data, where, 8)
    Data = jump(Data, lookback)

    return Data

def td_waldo_8(Data, high, low, close, buy, sell):

    # Adding a few columns
    Data = adder(Data, 10)
    
    for i in range(len(Data)):
        
        # Short-term Bottom
        if Data[i, 3] < Data[i - 1, 2] and \
           Data[i, 3] < Data[i - 2, 2] and \
           Data[i, 3] < Data[i - 3, 2] and \
           Data[i, 3] < Data[i - 4, 2] and \
           Data[i, 3] < Data[i - 5, 2] and \
           Data[i, 3] < Data[i - 6, 2] and \
           Data[i, 3] < Data[i - 7, 2] and \
           Data[i, 3] > Data[i - 12, 3]:
               
               Data[i, buy] = 1
        
        # Short-term Top
        if Data[i, 3] > Data[i - 1, 1] and \
           Data[i, 3] > Data[i - 2, 1] and \
           Data[i, 3] > Data[i - 3, 1] and \
           Data[i, 3] > Data[i - 4, 1] and \
           Data[i, 3] > Data[i - 5, 1] and \
           Data[i, 3] > Data[i - 6, 1] and \
           Data[i, 3] > Data[i - 7, 1] and \
           Data[i, 3] < Data[i - 12, 3]:
               
               Data[i, sell] = -1

    return Data

def relative_smoothing_index(Data, lookback, close, where, width = 1):
      
    lookback = (lookback * 2) - 1 # From exponential to smoothed
    
    # Adding a few columns
    Data = adder(Data, 8)
    
    # Calculating Differences
    for i in range(len(Data)):
        
        Data[i, where] = Data[i, 1] - Data[i - width, 1]
        Data[i, where + 1] = Data[i, 2] - Data[i - width, 2]
     
    # Calculating the Up and Down absolute values | Highs
    for i in range(len(Data)):
        
        if Data[i, where] > 0:
            
            Data[i, where + 2] = Data[i, where]
            
        elif Data[i, where] < 0:
            
            Data[i, where + 3] = abs(Data[i, where + 1])
            
    # Calculating the Smoothed Moving Average on Up and Down absolute values    
    Data = ema(Data, 2, lookback, where + 2, where + 4)
    Data = ema(Data, 2, lookback, where + 3, where + 5)

    # Calculating the Relative Strength | Highs
    Data[:, where + 6] = Data[:, where + 4] / Data[:, where + 5]    
    
    # Calculate the Relative Strength Index
    Data[:, where + 7] = (100 - (100 / (1 + Data[:, where + 6])))

    # Cleaning
    Data = deleter(Data, where, 7)
    Data = jump(Data, lookback)

    return Data

def ulcer_index(Data, lookback, close, where):
    
    # Adding the necessary columns
    Data = adder(Data, 1)
    
    # Percentage Down Calculation
    for i in range(len(Data)):
        try:
            
            Data[i, where] = (Data[i, 3] / max(Data[i - lookback + 1:i + 1, 1])) * lookback
        
        except:
            pass
        
    # Squared Average
    Data[:, where] = Data[:, where] / lookback
    
    # Ulcer Index
    Data[:, where] = np.sqrt(Data[:, where])
    
    return Data

def psychological_levels_scanner(Data, trend, signal, buy, sell):
    
    # Adding buy and sell columns
    Data = adder(Data, 15)
    
    # Rounding for ease of use
    Data = rounding(Data, 4)
    
    # Scanning for Psychological Levels
    for i in range(len(Data)):
        
        if  Data[i, 3] == 0.6000 or Data[i, 3] == 0.6100 or Data[i, 3] == 0.6200 or Data[i, 3] == 0.6300 or \
            Data[i, 3] == 0.6400 or Data[i, 3] == 0.6500 or Data[i, 3] == 0.6600 or Data[i, 3] == 0.6700 or \
            Data[i, 3] == 0.6800 or Data[i, 3] == 0.6900 or Data[i, 3] == 0.7000 or Data[i, 3] == 0.7100 or \
            Data[i, 3] == 0.7200 or Data[i, 3] == 0.7300 or Data[i, 3] == 0.7400 or Data[i, 3] == 0.7500 or \
            Data[i, 3] == 0.7600 or Data[i, 3] == 0.7700 or Data[i, 3] == 0.7800 or Data[i, 3] == 0.7900 or \
            Data[i, 3] == 0.8000 or Data[i, 3] == 0.8100 or Data[i, 3] == 0.8200 or Data[i, 3] == 0.8300 or \
            Data[i, 3] == 0.8400 or Data[i, 3] == 0.8500 or Data[i, 3] == 0.8600 or Data[i, 3] == 0.8700 or \
            Data[i, 3] == 0.8800 or Data[i, 3] == 0.8900 or Data[i, 3] == 0.9000 or Data[i, 3] == 0.9100 or \
            Data[i, 3] == 0.9200 or Data[i, 3] == 0.9300 or Data[i, 3] == 0.9400 or Data[i, 3] == 0.9500 or \
            Data[i, 3] == 0.9600 or Data[i, 3] == 0.9700 or Data[i, 3] == 0.9800 or Data[i, 3] == 0.9900 or \
            Data[i, 3] == 1.0000 or Data[i, 3] == 1.0100 or Data[i, 3] == 1.0200 or Data[i, 3] == 1.0300 or \
            Data[i, 3] == 1.0400 or Data[i, 3] == 1.0500 or Data[i, 3] == 1.0600 or Data[i, 3] == 1.0700 or \
            Data[i, 3] == 1.0800 or Data[i, 3] == 1.0900 or Data[i, 3] == 1.1000 or Data[i, 3] == 1.1100 or \
            Data[i, 3] == 1.1200 or Data[i, 3] == 1.1300 or Data[i, 3] == 1.1400 or Data[i, 3] == 1.1500 or \
            Data[i, 3] == 1.1600 or Data[i, 3] == 1.1700 or Data[i, 3] == 1.1800 or Data[i, 3] == 1.1900 or \
            Data[i, 3] == 1.2000 or Data[i, 3] == 1.2100 or Data[i, 3] == 1.2300 or Data[i, 3] == 1.2400 or \
            Data[i, 3] == 1.2500 or Data[i, 3] == 1.2600 or Data[i, 3] == 1.2700 or Data[i, 3] == 1.2800 or \
            Data[i, 3] == 1.2900 or Data[i, 3] == 1.3000 or Data[i, 3] == 1.3100 or Data[i, 3] == 1.3200 or \
            Data[i, 3] == 1.3300 or Data[i, 3] == 1.3400 or Data[i, 3] == 1.3500 or Data[i, 3] == 1.3600 or \
            Data[i, 3] == 1.3700 or Data[i, 3] == 1.3800 or Data[i, 3] == 1.3900 or Data[i, 3] == 1.4000 or \
            Data[i, 3] == 1.4100 or Data[i, 3] == 1.4200 or Data[i, 3] == 1.4300 or Data[i, 3] == 1.4400 or \
            Data[i, 3] == 1.4500 or Data[i, 3] == 1.4600 or Data[i, 3] == 1.4700 or Data[i, 3] == 1.4800 or \
            Data[i, 3] == 1.4900 or Data[i, 3] == 1.5000 or Data[i, 3] == 1.5100 or Data[i, 3] == 1.5200 or \
            Data[i, 3] == 1.5300 or Data[i, 3] == 1.5400 or Data[i, 3] == 1.5500 or Data[i, 3] == 1.5600 or \
            Data[i, 3] == 1.5700 or Data[i, 3] == 1.5800 or Data[i, 3] == 1.5900 or Data[i, 3] == 1.6000 or \
            Data[i, 3] == 1.6100 or Data[i, 3] == 1.6200 or Data[i, 3] == 1.6300 or Data[i, 3] == 1.6400 or \
            Data[i, 3] == 1.6500 or Data[i, 3] == 1.6600 or Data[i, 3] == 1.6700 or Data[i, 3] == 1.6800 or \
            Data[i, 3] == 1.6900 or Data[i, 3] == 1.7000 or Data[i, 3] == 1.7100 or Data[i, 3] == 1.7200 or \
            Data[i, 3] == 1.7300 or Data[i, 3] == 1.7400 or Data[i, 3] == 1.7500 or Data[i, 3] == 1.7600 or \
            Data[i, 3] == 1.7700 or Data[i, 3] == 1.7800 or Data[i, 3] == 1.7900 or Data[i, 3] == 1.8000:
                
                Data[i, signal] = 1

    return Data
 
def augmented_rsi(Data, lookback, high, low, close, where, width = 1, genre = 'Smoothed'):
    
    Data = adder(Data, 5)
    
    # Calculating Differences
    for i in range(len(Data)):
        
        Data[i, where] = Data[i, high] - Data[i - width, high]

    for i in range(len(Data)):
        
        Data[i, where + 1] = Data[i, low] - Data[i - width, low]
        
    for i in range(len(Data)):
        
        Data[i, where + 2] = Data[i, close] - Data[i - width, close]
        
    # Calculating the Up and Down absolute values
    for i in range(len(Data)):
        
        if Data[i, where + 2] > 0:
            
            Data[i, where + 3] = abs(Data[i, where])
            
        elif Data[i, where + 2] < 0:
            
            Data[i, where + 4] = abs(Data[i, where + 1])
              
    # Cleaning
    Data = deleter(Data, where, 3)
    Data = adder(Data, 2)
            
    # Calculating the Smoothed Moving Average on Up and Down absolute values    
    if genre == 'Smoothed':
        
        lookback = (lookback * 2) - 1 # From exponential to smoothed
            
        Data = ema(Data, 2, lookback, where, where + 2)
        Data = ema(Data, 2, lookback, where + 1, where + 3)
    
    if genre == 'Simple':
        Data = ma(Data, lookback, where, where + 2)
        Data = ma(Data, lookback, where + 1, where + 3)
    
    # Calculating the Relative Strength
    Data[:, where + 4] = Data[:, where + 2] / Data[:, where + 3]
    
    # Calculate the Relative Strength Index
    Data[:, where + 5] = (100 - (100 / (1 + Data[:, where + 4])))

    # Cleaning
    Data = deleter(Data, where, 5)
    Data = jump(Data, lookback)

    return Data

def td_waldo_6(Data, high, low, close, buy, sell):

    # Adding a few columns
    Data = adder(Data, 10)
    
    for i in range(len(Data)):
        
        # Short-term Bottom
        if Data[i, 2] < Data[i - 1, 2] and \
           Data[i, 2] < Data[i - 2, 2] and \
           Data[i, 2] < Data[i - 3, 2] and \
           Data[i, 2] < Data[i - 4, 2] and \
           Data[i, 2] < Data[i - 5, 2] and \
           Data[i, 2] < Data[i - 6, 2] and \
           Data[i, 2] < Data[i - 7, 2] and \
           Data[i, 2] < Data[i - 8, 2] and \
           abs(Data[i, 3] - Data[i, 2]) - abs(Data[i - 1, 3] - Data[i - 1, 2]) > 0:
               
               Data[i, buy] = 1
        
        # Short-term Top
        if Data[i, 1] > Data[i - 1, 1] and \
           Data[i, 1] > Data[i - 2, 1] and \
           Data[i, 1] > Data[i - 3, 1] and \
           Data[i, 1] > Data[i - 4, 1] and \
           Data[i, 1] > Data[i - 5, 1] and \
           Data[i, 1] > Data[i - 6, 1] and \
           Data[i, 1] > Data[i - 7, 1] and \
           Data[i, 1] > Data[i - 8, 1] and \
           abs(Data[i, 3] - Data[i, 1]) - abs(Data[i - 1, 3] - Data[i - 1, 1]) > 0:
               
               Data[i, sell] = -1

    return Data


def td_waldo_5(Data, high, low, close, buy, sell):

    # Adding a few columns
    Data = adder(Data, 10)
    Data = rounding(Data, 4)
    
    for i in range(len(Data)):
        
        # Short-term Bottom
        if Data[i, 3] == Data[i - 1, 3] and Data[i - 1, 3] < Data[i - 2, 3]:
               
               Data[i, buy] = 1
        
        # Short-term Top
        if Data[i, 3] == Data[i - 1, 3] and Data[i - 1, 3] > Data[i - 2, 3]:
                              
               Data[i, sell] = -1

    return Data


def k_candlesticks(Data, opening, high, low, close, lookback, where):
    
    # Adding the necessary columns
    Data = adder(Data, 4)
    
    # Averaging the Open
    Data = ma(Data, lookback, opening, where)
     
    # Averaging the High
    Data = ma(Data, lookback, high, where + 1)
       
    # Averaging the Low
    Data = ma(Data, lookback, low, where + 2)
       
    # Averaging the Close
    Data = ma(Data, lookback, close, where + 3)
    
    return Data

def td_camouflage(Data):
    
    # Adding columns
    Data = adder(Data, 20)
    
    # True Low Calculation
    for i in range(len(Data)):
        
        Data[i, 5] = min(Data[i, 2], Data[i - 2, 2])
    
    # True High Calculation
    for i in range(len(Data)):
        
        Data[i, 6] = max(Data[i, 1], Data[i - 2, 1])    
    
    # Bullish signal
    for i in range(len(Data)):
        
        if Data[i, 3] < Data[i - 1, 3] and Data[i, 3] > Data[i, 0] and Data[i, 2] < Data[i - 2, 5]:
            
            Data[i, 7] = 1
    
    # Bearish signal
    for i in range(len(Data)):
        
        if Data[i, 3] > Data[i - 1, 3] and Data[i, 3] < Data[i, 0] and Data[i, 1] > Data[i - 2, 6]:
            
            Data[i, 8] = -1    
    
    # Cleaning
    Data = deleter(Data, 5, 1)
    
    return Data

def td_clopwin(Data):
    
    # Adding columns
    Data = adder(Data, 20) 
    
    # Bullish signal
    for i in range(len(Data)):
        
        if Data[i, 1] < Data[i - 1, 1] and Data[i, 2] > Data[i - 1, 2] and Data[i, 3] > Data[i - 2, 3]: 
            
            Data[i, 6] = 1
    
    # Bearish signal
    for i in range(len(Data)):
        
        if Data[i, 1] < Data[i - 1, 1] and Data[i, 2] > Data[i - 1, 2] and Data[i, 3] < Data[i - 2, 3]: 
            
            Data[i, 7] = -1    
    
    return Data

def extreme_duration(Data, indicator, upper_barrier, lower_barrier, where_upward_extreme, where_downward_extreme, net_col):
     
    # Adding columns
    Data = adder(Data, 20) 
       
    # Time Spent Overbought 
    for i in range(len(Data)):
      if Data[i, indicator] > upper_barrier:
            Data[i, where_upward_extreme] = Data[i - 1, where_upward_extreme] + 1
            
      else:
            a = 0
            Data[i, where_upward_extreme] = a
            
    # Time Spent Oversold   
    for i in range(len(Data)):
        
      if Data[i, indicator] < lower_barrier:
            Data[i, where_downward_extreme] = Data[i - 1, where_downward_extreme] + 1
            
      else:
            a = 0
            Data[i, where_downward_extreme] = a
        
    Data[:, net_col] = Data[:, where_upward_extreme] - Data[:, where_downward_extreme]
    
    Data = deleter(Data, 6, 2)

    
    return Data

def countdown_indicator(Data, lookback, ma_lookback, opening, high, low, close, where):
    
    # Adding columns
    Data = adder(Data, 20)
    
    # Calculating Upside Pressure
    for i in range(len(Data)):
        
        if Data[i, close] > Data[i, opening]:
            
            Data[i, where] = 1

        if Data[i, high] > Data[i - 1, high]:
            
            Data[i, where + 1] = 1
            
    Data[:, where + 2] = Data[:, where] + Data[:, where + 1]    
    Data = deleter(Data, where, 2)    
            
    # Calculating Downside Pressure
    for i in range(len(Data)):
        
        if Data[i, close] < Data[i, opening]:
            
            Data[i, where + 1] = 1

        if Data[i, low] < Data[i - 1, low]:
            
            Data[i, where + 2] = 1
            
    Data[:, where + 3] = Data[:, where + 1] + Data[:, where + 2]    
    Data = deleter(Data, where + 1, 2)     
    
    # Calculate Cumulative Upside Pressure
    for i in range(len(Data)):
        
        Data[i, where + 2] = Data[i - lookback + 1:i + 1, where].sum()
    
    # Calculate Cumulative Downside Pressure
    for i in range(len(Data)):
        
        Data[i, where + 3] = Data[i - lookback + 1:i + 1, where + 1].sum()
       
    # Calculate the Countdown Indicator
    Data[:, where + 4] = Data[:, where + 2] - Data[:, where + 3]    
    Data = ema(Data, 2, ma_lookback, where + 4, where + 5)
    
    Data = deleter(Data, where, 5)
    Data = jump(Data, lookback)
       
    return Data


def disparity_index(Data, lookback, close, where):
    
    # Adding a column
    Data = adder(Data, 2)
    
    # Calculating the moving average on closing prices
    Data = ma(Data, lookback, close, where)
    
    # Calculating the Disparity Index
    for i in range(len(Data)):
        
      Data[i, where + 1] = ((Data[i, close] / Data[i, where]) - 1) * 100
    
    # Cleaning
    Data = deleter(Data, where, 1)
    
    return Data


def z_score_indicator(Data, ma_lookback, std_lookback, close, where):
    
    # Adding Columns
    Data = adder(Data, 1)
    
    # Calculating the moving average
    Data = ma(Data, ma_lookback, close, where)
    
    # Calculating the standard deviation
    Data = volatility(Data, std_lookback, close, where + 1)
    
    # Calculating the Z-Score
    for i in range(len(Data)):
        
        Data[i, where + 2] = (Data[i, close] - Data[i, where]) / Data[i, where + 1]
        
    # Cleaning
    Data = deleter(Data, where, 2)
    
    return Data
    
def aroon(Data, period, close, where):
    
    # Adding Columns
    Data = adder(Data, 10)

    # Max Highs
    for i in range(len(Data)):
        
        try:
        
            Data[i, where] = max(Data[i - period + 1:i + 1, 1])
        
        except ValueError:
            
            pass
    # Max Lows
    for i in range(len(Data)):
        
        try:
       
            Data[i, where + 1] = min(Data[i - period + 1:i + 1, 2]) 
        
        except ValueError:
            
            pass
        
    # Where the High Equals the Highest High in the period
    for i in range(len(Data)):
       
        if Data[i, 1] == Data[i, where]:
            
            Data[i, where + 2] = 1
        
    # Where the Low Equals the Lowest Low in the period
    for i in range(len(Data)):
       
        if Data[i, 2] == Data[i, where + 1]:
            
            Data[i, where + 3] = 1

    # Jumping Rows
    Data = jump(Data, period)

    # Calculating Aroon Up
    for i in range(len(Data)):
        
        try:
        
            try:
                
                x = max(Data[i - period:i, 1])
                y = np.where(Data[i - period:i, 1] == x)
                y = np.array(y)
                distance = period - y
            
                Data[i - 1, where + 4] = 100 *((period - distance) / period)


            except ValueError:
                
                pass
            
        except IndexError:
            
            pass


    # Calculating Aroon Down
    for i in range(len(Data)):
        
        try:
        
            try:
                
                x = min(Data[i - period:i, 2])
                y = np.where(Data[i - period:i, 2] == x)
                y = np.array(y)
                distance = period - y
            
                Data[i - 1, where + 5] = 100 *((period - distance) / period)


            except ValueError:
                
                pass
            
        except IndexError:
            
            pass
    
    # Cleaning
    Data = deleter(Data, 5, 4)
    
    return Data    










