
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

frame_MIN1 = mt5.TIMEFRAME_M1
frame_M5   = mt5.TIMEFRAME_M5
frame_M10  = mt5.TIMEFRAME_M10
frame_M15  = mt5.TIMEFRAME_M15
frame_M30  = mt5.TIMEFRAME_M30
frame_H1   = mt5.TIMEFRAME_H1
frame_H2   = mt5.TIMEFRAME_H2
frame_H3   = mt5.TIMEFRAME_H3
frame_H4   = mt5.TIMEFRAME_H4
frame_H6   = mt5.TIMEFRAME_H6
frame_D1   = mt5.TIMEFRAME_D1
frame_W1   = mt5.TIMEFRAME_W1
frame_M1   = mt5.TIMEFRAME_MN1

now = datetime.datetime.now()

def asset_list(asset_set):
   
    if asset_set == 'FX':
        
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
    
    if horizon == 'MN1':
        data = get_quotes(frame_MIN1, 2021, 7, 1, asset = assets[asset])
        data = data.iloc[:, 1:5].values
        data = data.round(decimals = 5)    
    
    if horizon == 'M5':
        data = get_quotes(frame_M5, 2021, 6, 1, asset = assets[asset])
        data = data.iloc[:, 1:5].values
        data = data.round(decimals = 5)

    if horizon == 'M10':
        data = get_quotes(frame_M10, 2020, 8, 1, asset = assets[asset])
        data = data.iloc[:, 1:5].values
        data = data.round(decimals = 5)
        
    if horizon == 'M15':
        data = get_quotes(frame_M15, 2019, 1, 1, asset = assets[asset])
        data = data.iloc[:, 1:5].values
        data = data.round(decimals = 5)
        
    if horizon == 'M30':
        data = get_quotes(frame_M30, 2016, 8, 1, asset = assets[asset])
        data = data.iloc[:, 1:5].values
        data = data.round(decimals = 5)        
        
    if horizon == 'H1':
        data = get_quotes(frame_H1, 2020, 1, 1, asset = assets[asset])
        data = data.iloc[:, 1:5].values
        data = data.round(decimals = 5)        
        
    if horizon == 'H2':
        data = get_quotes(frame_H2, 2010, 1, 1, asset = assets[asset])
        data = data.iloc[:, 1:5].values
        data = data.round(decimals = 5)        
        
    if horizon == 'H3':
        data = get_quotes(frame_H3, 2000, 1, 1, asset = assets[asset])
        data = data.iloc[:, 1:5].values
        data = data.round(decimals = 5)        
        
    if horizon == 'H4':
        data = get_quotes(frame_H4, 2000, 1, 1, asset = assets[asset])
        data = data.iloc[:, 1:5].values
        data = data.round(decimals = 5)        
        
    if horizon == 'H6':
        data = get_quotes(frame_H6, 2000, 1, 1, asset = assets[asset])
        data = data.iloc[:, 1:5].values
        data = data.round(decimals = 5)            
        
    if horizon == 'D1':
        data = get_quotes(frame_D1, 2000, 1, 1, asset = assets[asset])
        data = data.iloc[:, 1:5].values
        data = data.round(decimals = 5)        
           
    if horizon == 'W1':
        data = get_quotes(frame_W1, 2000, 1, 1, asset = assets[asset])
        data = data.iloc[:, 1:5].values
        data = data.round(decimals = 5)        
         
    if horizon == 'M1':
        data = get_quotes(frame_M1, 2000, 1, 1, asset = assets[asset])
        data = data.iloc[:, 1:5].values
        data = data.round(decimals = 5)        

    return data 

def get_quotes(time_frame, year = 2005, month = 1, day = 1, asset = "EURUSD"):
        
    # Establish connection to MetaTrader 5 
    if not mt5.initialize():
        print("initialize() failed, error code =", mt5.last_error())
        quit()
    
    timezone = pytz.timezone("Europe/Paris")
    
    utc_from = datetime.datetime(year, month, day, tzinfo = timezone)
    utc_to = datetime.datetime(now.year, now.month, now.day + 1, tzinfo = timezone)
    
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
            
def risk_management(Data, stop, target, atr_col):  
    
    # Buy Orders
    for i in range(len(Data)):
        
        try:
            
            if Data[i, 6] == 1:
                
                for a in range(i + 1, i + 1000):
                    
                    if Data[a, 1] >=  Data[i, 3] + (Data[i, atr_col] * target):
                        
                        Data[a, 8] = Data[i, atr_col] * target
                        
                        break
                    
                    elif Data[a, 2] <= Data[i, 3] - (Data[i, atr_col] * stop):
                    
                        Data[a, 8] = - Data[i, atr_col] * stop
                        
                        break
                    
                    elif Data[a, 6] == 1 or Data[a, 7] == -1:
                        
                        Data[a, 8] = Data[a, 3] - Data[i, 3]
                        
                        break
                    
                    else:
                        
                        continue                
                    
            else:
                
                continue
            
        except IndexError:
            
            pass
                    
    # Sell Orders
    for i in range(len(Data)):
        
        try:
            
            if Data[i, 7] == -1:
                
                for a in range(i + 1, i + 1000):
                    
                    if Data[a, 2] <= Data[i, 3] - (Data[i, atr_col] * target):
                        
                        Data[a, 9] = (Data[i, atr_col] * target)
                        
                        break
                    
                    elif Data[a, 1] >= Data[i, 3] + (Data[i, atr_col] * stop):
                        
                        Data[a, 9] = - Data[i, atr_col] * stop
                        
                        break    
                    
                    elif Data[a, 6] == 1 or Data[a, 7] == -1:
                        
                        Data[a, 9] = Data[i, 3] - Data[a, 3]
                        
                        break   
                     
                    else:
                        
                        continue
                    
            else:
                continue
            
        except IndexError:
            
            pass
    
    # Combining Returns and netting the results
    for i in range(len(Data)):
        
        try:    
            
            if Data[i, 8] != 0:
                
                Data[i, 10] = (Data[i, 8] - expected_cost) * lot
                
            if Data[i, 9] != 0:
                
                Data[i, 10] = (Data[i, 9] - expected_cost) * lot
                
        except IndexError:
            
            pass
        
    # Creating a portfolio balance array
    Data[:, 11] = investment 
    
    # Adding returns to the balance    
    for i in range(len(Data)):
    
        Data[i, 11] = Data[i - 1, 11] + (Data[i, 10])
   
    return Data

def position_sizing(Data, window):
    
    # Preparing the Data
    net_result = Data[Data[:, 10] != 0, 10]
    
    net_result = np.reshape(net_result, (-1, 1))
    
    net_result = adder(net_result, 1)
    
    net_result[:, 1] = investment
    
    for i in range(len(net_result)):
        
        net_result[i, 1] = net_result[i, 0] + net_result[i - 1, 1]
        
    net_result = adder(net_result, 15)     

    # Sizing using the Hit Ratio     
    for i in range(len(Data)):
        
        try:
                
            # Rolling Number of Trades
            if net_result[i, 0] > 0:
                
                net_result[i, 2] = 1
                
        except IndexError:
            
            pass
        
    # Rolling Number of Profitable Trades
    for i in range(len(net_result)):
        
        net_result[i, 3] = np.sum(net_result[i - window + 1:i + 1, 2])
    
    net_result = deleter(net_result, 2, 1)
    
    # Rolling Hit ratio
    net_result[:, 3] = net_result[:, 2] / window
    
    net_result = deleter(net_result, 2, 1)    

    # Calculating Rolling Average Gain & Average Loss
    for i in range(len(net_result)):
        
        # Average Gain
        if net_result[i, 0] > 0:
            
            net_result[i, 3] = net_result[i, 0]
        
    net_result = ema(net_result, 2, window, 3, 4)
    net_result = deleter(net_result, 3, 1)

    # Calculating Rolling Average Gain & Average Loss
    for i in range(len(net_result)):
        
        # Average Loss
        if net_result[i, 0] < 0:
            
            net_result[i, 4] = abs(net_result[i, 0])
        
    net_result = ema(net_result, 2, window, 4, 5)
    net_result = deleter(net_result, 4, 1)

    # Rolling Expectancy
    for i in range(len(net_result)):
         
        net_result[i, 5] = (net_result[i, 2] * net_result[i, 3]) - ((1 - net_result[i, 2]) * net_result[i, 4])
                   
    # Adjusting Position Size - Hit Ratio
    for i in range(len(net_result)):
        
        if net_result[i - 1, 2] < 0.60:
            
            net_result[i, 6] = net_result[i, 0] * 0.50
        
        if net_result[i - 1, 2] >= 0.60 and net_result[i - 1, 2] < 0.80:
            
            net_result[i, 6] = net_result[i, 0]       

        if net_result[i - 1, 2] >= 0.80:
            
            net_result[i, 6] = net_result[i, 0] * 1.25
            
    # Adjusting Position Size - Expectancy
    for i in range(len(net_result)):
        
        if net_result[i - 1, 5] < 50:
            
            net_result[i, 7] = net_result[i, 0] * 0.50
        
        if net_result[i - 1, 5] >= 50:
            
            net_result[i, 7] = net_result[i, 0] * 1.25          
        
    # Adjusting Position Size - Portfolio Size
    for i in range(len(net_result)):
        

        
        net_result[i, 8] = (net_result[i - 1, 8] / 100000) * net_result[i, 0]

    # Cleaning
    net_result = deleter(net_result, 2, 4)
    
    # Comparing to Equal Sizing
    
    for i in range(len(net_result)):
        
        try:
            net_result[0, 5] = net_result[0, 1]
            
            net_result[i, 5] = net_result[i - 1, 5] + (net_result[i, 2]) 
            
        except IndexError:
        
            pass
    
    for i in range(len(Data)):
        try:
            net_result[0, 6] = net_result[0, 1]
    
            net_result[i, 6] = net_result[i - 1, 6] + (net_result[i, 3])
        except IndexError:
        
            pass        
    for i in range(len(Data)):
        try:
            net_result[0, 7] = net_result[0, 1]
    
            net_result[i, 7] = net_result[i - 1, 7] + (net_result[i, 0] * net_result[i - 1, 7] / 100000)
        except IndexError:
        
            pass     
    
    plt.plot(net_result[:, 1], label = 'Equal Sizing', color = 'black', linewidth = 1)
    plt.plot(net_result[:, 5], label = 'Hit Ratio Sizing', linewidth = 1)
    plt.plot(net_result[:, 6], label = 'Expectancy Sizing', linewidth = 1)
    plt.plot(net_result[:, 7], label = 'Percentage of Portfolio Sizing', linewidth = 1)
    plt.grid()
    plt.legend()
    plt.axhline(y = investment, color = 'black')

    return net_result


def performance(Data, signal_quality_period, name, time_period = 11):
    
    # Net Profit
    net_profit = round(Data[-1, 11] - investment, 0)
    
    # Net Return
    net_return = round((Data[-1, 11] / investment - 1) * 100, 2)
    
    # Profit Factor    
    total_net_profits = Data[Data[:, 10] > 0, 10]
    total_net_losses  = Data[Data[:, 10] < 0, 10] 
    total_net_losses  = abs(total_net_losses)
    profit_factor     = round(np.sum(total_net_profits) / np.sum(total_net_losses), 2)

    # Hit Ratio    
    hit_ratio         = len(total_net_profits) / (len(total_net_losses) + len(total_net_profits))
    hit_ratio         = round(hit_ratio, 2) * 100
    
    # Risk Reward Ratio
    theoretical_risk_reward = target / stop
    average_gain            = total_net_profits.mean()
    average_loss            = total_net_losses.mean()
    realized_risk_reward    = average_gain / average_loss
    
    # Risk Gap
    risk_gap                = round(realized_risk_reward - theoretical_risk_reward, 2)
    
    # Breakeven Rate
    breakeven_hit_ratio     = 1 / (1 + realized_risk_reward)
    breakeven_hit_ratio     = round(breakeven_hit_ratio, 2) * 100
    
    # Accuracy margin
    accuracy_margin         = hit_ratio - breakeven_hit_ratio
    
    # Signal Quality
    for i in range(len(Data)):
        
        try: 
            
            if Data[i, 6] == 1:
                
                Data[i + signal_quality_period, 15] = Data[i + signal_quality_period, 3] - Data[i, 3]
    
            elif Data[i, 7] == -1:
                
                Data[i + signal_quality_period, 16] = Data[i, 3] - Data[i + signal_quality_period, 3]  
                
        except IndexError:
            
            pass
        
    positives_signal_quality = Data[Data[:, 15] > 0, 15]
    negatives_signal_quality = Data[Data[:, 16] < 0, 16]   

    signal_quality = len(positives_signal_quality) / (len(negatives_signal_quality) + len(positives_signal_quality))     
    signal_quality = round(signal_quality, 2) * 100 
    
    # Signal Gap
    signal_gap = hit_ratio - signal_quality
    
    # Expectancy
    expectancy    = (average_gain * (hit_ratio / 100)) - ((1 - (hit_ratio / 100)) * average_loss)
    expectancy    = round(expectancy, 2)
    
    # Largest Win and Largest Loss
    largest_win  = round(max(total_net_profits), 2)
    largest_loss = round(max(total_net_losses), 2)
    
    # Total number of Trades
    trades = len(total_net_losses) + len(total_net_profits)
    
    # Frequency of Trades
    frequency = trades / time_period
    
    # Fees Paid in $
    fees_paid = trades * 0.5  
    
    # Bias
    total_long_trades  = len(Data[Data[:, 8] > 0, 8]) + len(Data[Data[:, 8] < 0, 8])
    total_short_trades = len(Data[Data[:, 9] > 0, 9]) + len(Data[Data[:, 9] < 0, 9])
    
    bias = round(total_long_trades / total_short_trades, 2)
    
    # Percentage of Winning Long/Short Trades
    total_net_profits_longs  = Data[Data[:, 8] > 0, 8]
    total_net_profits_shorts = Data[Data[:, 9] > 0, 9]
    
    percentage_win_long  = len(total_net_profits_longs)  / len(total_net_profits)
    percentage_win_short = len(total_net_profits_shorts) / len(total_net_profits)
    
    # Average Trade Duration
    for i in range(len(Data)):
        
        if Data[i, 12] != 0:
            
            for a in range(len(Data)):
                
                if Data[a, 12] != 0:
                    
                    Data[a, 22] = Data[a, 12] - Data[i, 12]
                    
                    break
                
                else:
                
                    continue
        else:
            
            continue
        
    # Completion Ratio (Times Stopped over Total Closes)

    # Active to Passive Performance Metric
    passive_performance     = round((Data[-1, 3] / Data[0, 3] - 1) * 100, 2)
    active_to_passive_metric = net_return - passive_performance   
    
    # Maximum Drawdown
    dried_equity_curve_model = dried_equity_curve(Data)
    xs = dried_equity_curve_model[:, 1]
    i  = np.argmax(np.maximum.accumulate(xs) - xs) # end of the period
    j  = np.argmax(xs[:i]) # start of period

    maximum_drawdown = (dried_equity_curve_model[i, 1] / dried_equity_curve_model[j, 1] - 1) * 100

    # plt.plot([trough, peak], [segregated_equity_curve[int(peak)], segregated_equity_curve[int(trough)]], 'o', color = 'Red', markersize = 5)      
    
    print('-----------Performance-----------', name)    
    print('Hit Ratio         = ', hit_ratio, '%')
    print('Total Return      = ', net_return, '%')    
    print('Expectancy        = ', '$', expectancy)
    print('Profit factor     = ', profit_factor) 
    print('Signal Quality    = ', signal_quality, '%')
    print('Signal Gap        = ', signal_gap, '%')
    print('Break-Even Ratio  = ', breakeven_hit_ratio, '%')
    print('Accuracy Margin   = ', round(accuracy_margin, 2), '%')
    
    print('') 
    
    print('Maximum Drawdown  = ', round(maximum_drawdown, 2), '%')
    print('Theoretical RR    = ', theoretical_risk_reward)
    print('Realized RR       = ', round(realized_risk_reward, 3))
    print('Risk Gap          = ', risk_gap)
    
    print('')  
    
    print('Net profit        = ', '$', net_profit)
    print('Periodic Return   = ', round(net_return / 11, 2), '%')
    print('Average Gain      = ', '$', round(average_gain, 2))
    print('Average Loss      = ', '$', round(average_loss, 2))
    print('Largest Gain      = ', '$', largest_win)
    print('Largest Loss      = ', '$', largest_loss)    
    print('% Long Trades     = ', round(total_long_trades / trades, 3) * 100, '%')
    print('% Short Trades    = ', round(total_short_trades / trades, 3) * 100, '%')
    print('% Win Longs       = ', round(percentage_win_long, 2) * 100, '%')
    print('% Win Shorts      = ', round(percentage_win_short, 2) * 100, '%')
    print('Bias              = ', bias)
    print('Active to Passive = ', round(active_to_passive_metric, 2), '%')
    
    print('')
    
    print('Minimum Balance   = ', '$', min(dried_equity_curve_model[:, 1]))
    print('Maximum Balance   = ', '$', max(dried_equity_curve_model[:, 1])) 
    print('Trades            = ', trades)    
    print('Fees Paid         = ', '$', fees_paid)   
    print('Yearly Frequency  = ', int(frequency))
    print('Monthly Frequency = ', int(frequency / 12))
    print('Weekly Frequency  = ', int(frequency / 12 / 4))    
    
    return Data



def dried_equity_curve(Data):
    
    dried_equity_curve = Data[Data[:, 10] != 0, 10]
    
    dried_equity_curve = np.reshape(dried_equity_curve, (-1, 1))
    
    dried_equity_curve = adder(dried_equity_curve, 1)
    
    dried_equity_curve[:, 1] = investment
    
    for i in range(len(dried_equity_curve)):
        
        dried_equity_curve[i, 1] = dried_equity_curve[i, 0] + dried_equity_curve[i - 1, 1]

    return dried_equity_curve


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
    
def td_flip(Data, td, step, what, where_long, where_short):
        
    Data = adder(Data, 10)
    
    # Timing buy signal
    counter = -1 

    for i in range(len(Data)):    
        if Data[i, what] < Data[i - step, what]:
            Data[i, where_long] = counter
            counter += -1       
            if counter == -td - 1:
                counter = 0
            else:
                continue        
        elif Data[i, what] >= Data[i - step, what]:
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
        if Data[i, what] > Data[i - step, what]: 
            Data[i, where_short] = counter 
            counter += 1        
            if counter == td + 1: 
                counter = 0            
            else:
                continue        
        elif Data[i, what] <= Data[i - step, what]: 
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

def countdown(Data, where_td_buy, where_td_sell, onwhat, td, count, step, buy, sell):
    
    counter = -1
    Data = adder(Data, 1)
    
    for i in range(len(Data)):    
        
        try:
            if Data[i, where_td_buy] == -td:
                
                    Data = adder(Data, 1)
 
                    for j in range(i, i + 500):  
                        if Data[j, onwhat] < Data[j - step, 2] and Data[j, where_td_sell] != td: 
                            
                            Data[j, -1] = counter 
                            
                            if counter == -8:
                                eight = j
                            
                            counter += -1 
                    
                            if counter == -count - 1 and Data[j, onwhat] < Data[eight, onwhat]: 
                                   
                                counter = 0
                                Data[j, buy] = 1
                                Data = deleter(Data, -1, 1)
                                break
                        
                            else:
                                continue
                    
                        elif Data[j, onwhat] > Data[j - step, 2] and Data[j, where_td_sell] != td:
                            
                            if Data[j, 2] > Data[i - 8, 1]:
                                counter = -1
                                Data = deleter(Data, -1, 1)                                
                                break
                            else:
                                continue
              
                            continue
                    
                        elif Data[j, where_td_sell] == td:
                            counter = -1 
                            Data = deleter(Data, -1, 1)                            
                            break
                            
            else:
                continue
        except IndexError:
            pass

    if i == len(Data) - 1 and Data[i, -1] != Data[i, buy]:
        Data = deleter(Data, -1, 1)        

    counter = 1
    Data = adder(Data, 1)
    
    for i in range(len(Data)):    
        
        try:
            if Data[i, where_td_sell] == td:
                
                    Data = adder(Data, 1)
 
                    for j in range(i, i + 500):  
                        if Data[j, onwhat] > Data[j - step, 1] and Data[j, where_td_buy] != -td: 
                            
                            Data[j, -1] = counter 
                            
                            if counter == 8:
                                eight = j
                                
                            counter += 1 
                    
                            if counter == count + 1 and Data[j, onwhat] > Data[eight, onwhat]: 
                                   
                                counter = 0
                                Data[j, sell] = -1
                                Data = deleter(Data, -1, 1)
                                break
                        
                            else:
                                continue
                    
                        elif Data[j, onwhat] < Data[j - step, 1] and Data[j, where_td_buy] != -td:
                            
                            if Data[j, 1] < Data[i - 8, 2]:
                                counter = 1
                                Data = deleter(Data, -1, 1)                                
                                break
                            else:
                                continue
              
                            continue
                    
                        elif Data[j, where_td_buy] == -td:
                            counter = 1 
                            Data = deleter(Data, -1, 1)                            
                            break
                            
            else:
                continue
        except IndexError:
            pass
    
    if i == len(Data) - 1 and Data[i, -1] != Data[i, sell]:
        Data = deleter(Data, -1, 1)  
    
    return Data

def combo(Data, where_td_buy, where_td_sell, onwhat, td, combo, step, buy, sell):
    
    counter = -1
    Data = adder(Data, 1)
    
    for i in range(len(Data)):    
        
        try:
            if Data[i, where_td_buy] == -td:
                
                    Data = adder(Data, 1)
 
                    for j in range(i - 8, i + 500):  
                        if Data[j, onwhat] < Data[j - step, 2] and Data[j, 2] < Data[j - 1, 2] \
                           and Data[j, where_td_sell] != td and Data[j, onwhat] < Data[j - 1, onwhat]: 
                            
                            Data[j, -1] = counter 
                            
                            if counter == -8:
                                eight = j
                            
                            counter += -1 
                    
                            if counter == -combo - 1 and Data[j, onwhat] < Data[eight, onwhat]: 
                                   
                                counter = 0
                                Data[j, buy] = 1
                                Data = deleter(Data, -1, 1)
                                break
                        
                            else:
                                continue
                    
                        elif Data[j, onwhat] > Data[j - step, 2] and Data[j, where_td_sell] != td:
                            
                            if Data[j, 2] > Data[i - 8, 1]:
                                counter = -1
                                Data = deleter(Data, -1, 1)                                
                                break
                            else:
                                continue
              
                            continue
                    
                        elif Data[j, where_td_sell] == td:
                            counter = -1 
                            Data = deleter(Data, -1, 1)                            
                            break
                            
            else:
                continue
        except IndexError:
            pass

    if i == len(Data) - 1 and Data[i, -1] != Data[i, buy]:
        Data = deleter(Data, -1, 1)        

    counter = 1
    Data = adder(Data, 1)
    
    for i in range(len(Data)):    
        
        try:
            if Data[i, where_td_sell] == td:
                
                    Data = adder(Data, 1)
 
                    for j in range(i - 8, i + 500):  
                        if Data[j, onwhat] > Data[j - step, 1] and Data[j, 1] > Data[j - 1, 1] \
                           and Data[j, where_td_buy] != -td and Data[j, onwhat] > Data[j - 1, onwhat]: 
                            
                            Data[j, -1] = counter 
                            
                            if counter == 8:
                                eight = j
                                
                            counter += 1 
                    
                            if counter == combo + 1 and Data[j, onwhat] > Data[eight, onwhat]: 
                                   
                                counter = 0
                                Data[j, sell] = -1
                                Data = deleter(Data, -1, 1)
                                break
                        
                            else:
                                continue
                    
                        elif Data[j, onwhat] < Data[j - step, 1] and Data[j, where_td_buy] != -td:
                            
                            if Data[j, 1] < Data[i - 8, 2]:
                                counter = 1
                                Data = deleter(Data, -1, 1)                                
                                break
                            else:
                                continue
              
                            continue
                    
                        elif Data[j, where_td_buy] == -td:
                            counter = 1 
                            Data = deleter(Data, -1, 1)                            
                            break
                            
            else:
                continue
        except IndexError:
            pass
    
    if i == len(Data) - 1 and Data[i, -1] != Data[i, sell]:
        Data = deleter(Data, -1, 1)  
    
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

def astral(Data, td, step, step_two, what, high, low, where_long, where_short):
   
    # Timing buy signal
    counter = -1 

    for i in range(len(Data)):    
        if Data[i, what] < Data[i - step, what] and Data[i, low] < Data[i - step_two, low]:
            Data[i, where_long] = counter
            counter += -1       
            if counter == -td - 1:
                counter = 0
            else:
                continue        
        elif Data[i, what] >= Data[i - step, what]:
            counter = -1 
            Data[i, where_long] = 0 
        
    # Timing sell signal       
    counter = 1 
    
    for i in range(len(Data)):
        if Data[i, what] > Data[i - step, what] and Data[i, high] > Data[i - step_two, high]: 
            Data[i, where_short] = counter 
            counter += 1        
            if counter == td + 1: 
                counter = 0            
            else:
                continue        
        elif Data[i, what] <= Data[i - step, what]: 
            counter = 1 
            Data[i, where_short] = 0 
      
    return Data

def astral_technical(Data, td, step, step_two, what, where_long, where_short):
   
    # Timing buy signal
    counter = -1 

    for i in range(len(Data)):    
        if Data[i, what] < Data[i - step, what] and Data[i, what] < Data[i - step_two, what]:
            Data[i, where_long] = counter
            counter += -1       
            if counter == -td - 1:
                counter = 0
            else:
                continue        
        elif Data[i, what] >= Data[i - step, what]:
            counter = -1 
            Data[i, where_long] = 0 
        
    # Timing sell signal       
    counter = 1 
    
    for i in range(len(Data)):
        if Data[i, what] > Data[i - step, what] and Data[i, what] > Data[i - step_two, what]: 
            Data[i, where_short] = counter 
            counter += 1        
            if counter == td + 1: 
                counter = 0            
            else:
                continue        
        elif Data[i, what] <= Data[i - step, what]: 
            counter = 1 
            Data[i, where_short] = 0 
      
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

def democratic_indicator(Data, beginning_ma, final_ma, step, what, where):
    
    for i in range(beginning_ma, final_ma, step):
        Data = adder(Data, 1)
        Data = ma(Data, i, what, where)
        where = where + 1
        
    Data = jump(Data, i)
    
    Data = rounding(Data, 4)
    Data = adder(Data, 1)
    
    for i in range(len(Data)):
        
        transposed = np.transpose(Data[i, 4:43])
        transposed = list(transposed)
        mode_value = np.array(stats.multimode(transposed))
        
        if len(mode_value) > 1:
            mode_value = 0
        
        Data[i, 44] = np.array(mode_value)
           
    for i in range(len(Data)):
        
        if Data[i, -1] == 0:
            Data[i, -1] = Data[i - 1, -1]
      
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
            
def smoothed_ma(Data, lookback, what, where):

    # First value is a simple SMA
    Data = ma(Data, lookback, what, where)
    
    # Calculating first EMA
    Data[lookback - 1, where + 1] = Data[lookback + 1 - lookback:lookback + 1, 3].sum()

    # Calculating the rest of EMA
    for i in range(lookback + 2, len(Data)):
            try:
                Data[i, where] = (Data[i, what] * alpha) + (Data[i - 1, where] * beta)
        
            except IndexError:
                pass

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
        
def objective_strength_index(Data, start_lookback, end_lookback, what, where):

    for i in range(start_lookback, end_lookback):

        Data = rsi(Data, i, what, 0)        

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
 
   
def indicator_plot_double(Data, opening, high, low, close, second_panel, window = 250):

    fig, ax = plt.subplots(2, figsize = (10, 5))

    Chosen = Data[-window:, ]
    
    for i in range(len(Chosen)):
        
        ax[0].vlines(x = i, ymin = Chosen[i, low], ymax = Chosen[i, high], color = 'black', linewidth = 1)  
        
        if Chosen[i, close] > Chosen[i, opening]:
            color_chosen = 'green'
            ax[0].vlines(x = i, ymin = Chosen[i, opening], ymax = Chosen[i, close], color = color_chosen, linewidth = 2)  

        if Chosen[i, close] < Chosen[i, opening]:
            color_chosen = 'red'
            ax[0].vlines(x = i, ymin = Chosen[i, close], ymax = Chosen[i, opening], color = color_chosen, linewidth = 2)  
            
        if Chosen[i, close] == Chosen[i, opening]:
            color_chosen = 'black'
            ax[0].vlines(x = i, ymin = Chosen[i, close], ymax = Chosen[i, opening], color = color_chosen, linewidth = 2)  
   
    ax[0].grid() 
     
    ax[1].plot(Data[-window:, second_panel], color = 'royalblue', linewidth = 1)
    ax[1].grid()

def indicator_plot_triple_trouble(Data, first, second, choice, name = '', name_ind = '', window = 250):

    fig, ax = plt.subplots(2, figsize = (10, 5))

    Chosen = Data[-window:, ]
    
    for i in range(len(Chosen)):
        
        ax[0].vlines(x = i, ymin = Chosen[i, 2], ymax = Chosen[i, 1], color = 'black', linewidth = 1)  
        
        if Chosen[i, 3] > Chosen[i, 0]:
            color_chosen = 'green'
            ax[0].vlines(x = i, ymin = Chosen[i, 0], ymax = Chosen[i, 3], color = color_chosen, linewidth = 2)  

        if Chosen[i, 3] < Chosen[i, 0]:
            color_chosen = 'red'
            ax[0].vlines(x = i, ymin = Chosen[i, 3], ymax = Chosen[i, 0], color = color_chosen, linewidth = 2)  
            
        if Chosen[i, 3] == Chosen[i, 0]:
            color_chosen = 'black'
            ax[0].vlines(x = i, ymin = Chosen[i, 3], ymax = Chosen[i, 0], color = color_chosen, linewidth = 2)  
   
    ax[0].grid() 
    ax[0].plot(Data[-window:, choice])
     
    ax[1].plot(Data[-window:, second], color = 'black', label = name_ind, linewidth = 1)
    ax[1].grid()
    ax[1].legend()


def indicator_plot_double_trouble(Data, first, second, third, name = '', name_ind = '', name_ind_two = '', window = 250):

    fig, ax = plt.subplots(2, figsize = (10, 5))

    Chosen = Data[-window:, ]
    
    for i in range(len(Chosen)):
        
        ax[0].vlines(x = i, ymin = Chosen[i, 2], ymax = Chosen[i, 1], color = 'black', linewidth = 1)  
        ax[0].vlines(x = i, ymin = Chosen[i, 2], ymax = Chosen[i, 1], color = 'black', linewidth = 1)
   
    ax[0].grid() 
     
    ax[1].plot(Data[-window:, second], color = 'blue', label = name_ind)
    ax[1].plot(Data[-window:, third], color = 'brown', label = name_ind_two)
    ax[1].grid()
    ax[1].legend()
    
def indicator_plot_triple(Data, first, second, third, name = '', name_ind = '', name_ind_two = '', window = 500):

    fig, ax = plt.subplots(3, figsize = (10, 5))

    Chosen = Data[-window:, ]
    
    for i in range(len(Chosen)):
        
        ax[0].vlines(x = i, ymin = Chosen[i, 2], ymax = Chosen[i, 1], color = 'black', linewidth = 1)  
        ax[0].vlines(x = i, ymin = Chosen[i, 2], ymax = Chosen[i, 1], color = 'black', linewidth = 1)
   
    ax[0].grid() 
     
    ax[1].plot(Data[-window:, second], color = 'blue', label = name_ind)
    ax[1].grid()
    ax[1].legend()
    
    ax[2].plot(Data[-window:, third], color = 'brown', label = name_ind_two)
    ax[2].grid()
    ax[2].legend()
    
def deflated_rsi(Data, lookback, what, where):
    
    Data = rsi(Data, lookback, what, 0)
    
    Data = adder(Data, 20)
    
    Data = volatility(Data, lookback, what, where)
    Data[:, where] = 1 + (Data[:, where] * 1000)
    
    for i in range(len(Data)):
        
        if Data[i - 1, 5] < Data[i, 5]:
            Data[i, 6] = Data[i, 5] / Data[i, where]
            
        if Data[i - 1, 5] > Data[i, 5]:
            Data[i, 6] = Data[i, 5] * Data[i, where]
    
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
           
def variable_moving_average(Data, what, vol_lookback, where):
    
    Data = volatility(Data, vol_lookback, what, where)
    Data = normalizer(Data, vol_lookback, what, where + 1)
    
    for i in range(len(Data)):
        
        if Data[i, where + 1] >= 0 and Data[i, where + 1] <= 20 :
            Data[i, where + 2] = 100
        if Data[i, where + 1] > 20 and Data[i, where + 1] <= 40 :
            Data[i, where + 2] = 95            
        if Data[i, where + 1] > 40 and Data[i, where + 1] <= 60 :
            Data[i, where + 2] = 90
        if Data[i, where + 1] > 60 and Data[i, where + 1] <= 80 :
            Data[i, where + 2] = 85
        if Data[i, where + 1] > 80 and Data[i, where + 1] <= 100 :
            Data[i, where + 2] = 80              
            
    for i in range(len(Data)):
        variable_lookback = int(Data[i, where + 2])
        Data[i, where + 3] = (Data[i - variable_lookback + 1:i + 1, what].mean())
    
    Data = jump(Data, int(max(Data[:, where + 2])))
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

def trinity(Data, start, where_first_pattern_long, where_first_pattern_short, final_pattern, 
            final_step, wherelong, whereshort, buy, sell):
    
    counter = -1
    for i in range(len(Data)):   
       
        if Data[i, where_first_pattern_long] == -start:
        
            for j in range(i + 1, len(Data)): 
                
                if Data[j, 3] < Data[j - final_step, 3]: 
                    
                    Data[j, wherelong] = counter
                    counter += -1        
                    if counter == -final_pattern - 1:
                        counter = 0
                        Data[j, buy] = 1
                        break
                    else:
                        continue
                    
                elif Data[j, 3] >= Data[j - final_step, 3]: 
                    counter = -1 
                    Data[j, wherelong] = 0          
                    break

    counter = 1
    for i in range(len(Data)):   
       
        if Data[i, where_first_pattern_short] == start:
        
            for j in range(i + 1, len(Data)): 
                
                if Data[j, 3] > Data[j - final_step, 3]: 
                    
                    Data[j, whereshort] = counter
                    counter += 1        
                    if counter == final_pattern + 1:
                        Data[j, sell] = - 1                        
                        counter = 0
                        break
                    else:
                        continue
                    
                elif Data[j, 3] <= Data[j - final_step, 3]: 
                    counter = 1 
                    Data[j, whereshort] = 0          
                    break

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
                    Data[a - 1, where + 3] = (Data[i - 1, r] - Data[a - 1, indicator])
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

def zig_zag(Data, threshold, what, where):
    
    # Looping to find swing highs and lows
    for i in range(len(Data)):
       
        for a in range(len(Data)):
            
            if (Data[a, what] - Data[i, what]) > threshold or (Data[a, what] - Data[i, what]) < -threshold:
                Data[i, where] = 1

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

    Data = pure_pupil(Data, lookback, high, low, where + 1)
    
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

def hull_rsi(Data, close, where):

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

def random_walk_index(Data, lookback, atr_column, high, low, where):
    
    # Finding Lowest Low in previous n periods
    for i in range(len(Data)):
        try:
            
            Data[i, where] = min(Data[i - lookback + 1:i + 1, 2])
            
        except ValueError:
            pass
        
    # Finding Highest High in previous n periods
    for i in range(len(Data)):
        try:
    
            Data[i, where + 1] = max(Data[i - lookback + 1:i + 1, 1]) 
            
        except ValueError:
            pass        
    # RWI
    for i in range(len(Data)):
        
        Data[i, where + 2] = (Data[i, high] - Data[i, where]) / (Data[i, atr_column] * np.sqrt(lookback))
    
    
    # RWI Low
    for i in range(len(Data)):
        
        Data[i, where + 3] = (Data[i, where + 1] - Data[i - lookback, low]) / (Data[i, atr_column] * np.sqrt(lookback))
    
    Data = deleter(Data, where, 2)
    
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

def hull_hlc_data(Data, lookback, high, low, close, where):
    
    # Calculating the Hull Stochastic on Highs
    Data = hull_moving_average(Data, high, lookback, where)
    
    # Calculating the Hull Stochastic on Lows
    Data = hull_moving_average(Data, low, lookback, where + 1)
    
    # Calculating the Hull Stochastic on Closing Prices
    Data = hull_moving_average(Data, close, lookback, where + 2)
    
    # Adding an extra column for the Hull Stochastic
    Data = adder(Data, 1)
    
    return Data
   
def hull_stochastic(Data, lookback):
    
    # Calculating Hull Stochastic
    for i in range(len(Data)):
            
        try:
            Data[i, 8] = (Data[i, 7] - min(Data[i - lookback + 1:i + 1, 6])) / (max(Data[i - lookback + 1:i + 1, 5]) - min(Data[i - lookback + 1:i + 1, 6]))
        
        except ValueError:
            pass
    
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
    
def indicator_plot_triple(Data, first, second, third, name = '', name_ind_one = '', name_ind_two = '', window = 250):

    fig, ax = plt.subplots(3, figsize = (10, 5))

    Chosen = Data[-window:, ]
    
    for i in range(len(Chosen)):
        
        ax[0].vlines(x = i, ymin = Chosen[i, 2], ymax = Chosen[i, 1], color = 'black', linewidth = 1)  
        ax[0].vlines(x = i, ymin = Chosen[i, 2], ymax = Chosen[i, 1], color = 'black', linewidth = 1)
   
    ax[0].grid() 
    
    for i in range(len(Chosen)):
        
        if Chosen[i, 5] > Chosen[i - 1, 5]:
            ax[1].vlines(x = i, ymin = 0, ymax = Chosen[i, 5], color = 'green', linewidth = 1)  
        
        if Chosen[i, 5] < Chosen[i - 1, 5]:
            ax[1].vlines(x = i, ymin = Chosen[i, 5], ymax = 0, color = 'red', linewidth = 1)  
            
    ax[1].grid() 
    ax[1].axhline(y = 0, color = 'black', linewidth = 0.5, linestyle = '--')
    
    for i in range(len(Chosen)):
        
        if Chosen[i, 6] > Chosen[i - 1, 6]:
            ax[2].vlines(x = i, ymin = 0, ymax = Chosen[i, 5], color = 'grey', linewidth = 1)  
        
        if Chosen[i, 6] < Chosen[i - 1, 6]:
            ax[2].vlines(x = i, ymin = Chosen[i, 5], ymax = 0, color = 'black', linewidth = 1)  
            
    ax[2].grid() 
    ax[2].axhline(y = 0, color = 'black', linewidth = 0.5, linestyle = '--')
    
def indicator_plotito(Data, first, second, third, name = '', name_ind_one = '', name_ind_two = '', window = 250):

    fig, ax = plt.subplots(2, figsize = (10, 5))

    Chosen = Data[-window:, ]
    
    for i in range(len(Chosen)):
        
        ax[0].vlines(x = i, ymin = Chosen[i, 2], ymax = Chosen[i, 1], color = 'black', linewidth = 1)  
        ax[0].vlines(x = i, ymin = Chosen[i, 2], ymax = Chosen[i, 1], color = 'black', linewidth = 1)
   
    ax[0].grid() 
    
    ax[0].plot(Data[-window:, second], color = 'purple')
    
            
    ax[1].plot(Data[-window:, third])
    ax[1].grid() 
    ax[1].axhline(y = 0, color = 'black', linewidth = 0.5, linestyle = '--')    
    
def adaptive_macd(Data, what, long_ama, short_ama, signal_ama, where):
    
    Data = kama(Data, 3, 5, long_ama)
    Data = kama(Data, 3, 6, short_ama)
    
    Data[:, 7] = Data[:, 6] - Data[:, 5]

    Data = jump(Data, long_ama)
    Data = kama(Data, 7, 8, signal_ama)
    
    Data = deleter(Data, where, 2)   
    Data = jump(Data, signal_ama)
    
    return Data

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
    
def holding_fixed_period(Data, buy, sell, buy_return, sell_return, period):

    for i in range(len(Data)):
        try:
            if Data[i, buy] == 1: 
                Data[i + period, buy_return] = Data[i + period, 3] - Data[i, 3]
                Data[i + period, 10] = Data[i + period, buy_return]              
            elif Data[i, sell] == -1:        
                Data[i + period, sell_return] = Data[i, 3] - Data[i + period, 3]
                Data[i + period, 11] = Data[i + period, sell_return]              
                
        except IndexError:
            pass
        
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
 
def augmented_rsi(Data, lookback, close, where, width = 1, genre = 'Smoothed'):
    
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



# Application
def application(Data, close, distance, market):

    # Relative Strength Index Section
    rsi_lookback_five        = 2
    rsi_lookback_eight       = 5
    rsi_lookback_thirteen    = 13
    rsi_lookback_twenty_one  = 21
    rsi_lookback_thirty_four = 34
    
    upper_barrier_rsi_five        = 90
    upper_barrier_rsi_eight       = 85
    upper_barrier_rsi_thirteen    = 75
    upper_barrier_rsi_twenty_one  = 75
    upper_barrier_rsi_thirty_four = 70
    
    lower_barrier_rsi_five        = 10
    lower_barrier_rsi_eight       = 15
    lower_barrier_rsi_thirteen    = 25
    lower_barrier_rsi_twenty_one  = 25
    lower_barrier_rsi_thirty_four = 30
    
    # Stochastic Oscillator Section
    stochastic_lookback_five        = 2
    stochastic_lookback_eight       = 5
    stochastic_lookback_thirteen    = 13
    stochastic_lookback_twenty_one  = 21
    stochastic_lookback_thirty_four = 34
    
    upper_barrier_stochastic_five        = 90
    upper_barrier_stochastic_eight       = 85
    upper_barrier_stochastic_thirteen    = 75
    upper_barrier_stochastic_twenty_one  = 75
    upper_barrier_stochastic_thirty_four = 70
    
    lower_barrier_stochastic_five        = 10
    lower_barrier_stochastic_eight       = 15
    lower_barrier_stochastic_thirteen    = 25
    lower_barrier_stochastic_twenty_one  = 25
    lower_barrier_stochastic_thirty_four = 30
    
    # Demarker
    demarker_lookback_five        = 2
    demarker_lookback_eight       = 5
    demarker_lookback_thirteen    = 13
    demarker_lookback_twenty_one  = 21
    demarker_lookback_thirty_four = 34
    
    upper_barrier_demarker_five        = 0.90
    upper_barrier_demarker_eight       = 0.85
    upper_barrier_demarker_thirteen    = 0.75
    upper_barrier_demarker_twenty_one  = 0.75
    upper_barrier_demarker_thirty_four = 0.70
    
    lower_barrier_demarker_five        = 0.10
    lower_barrier_demarker_eight       = 0.15
    lower_barrier_demarker_thirteen    = 0.25
    lower_barrier_demarker_twenty_one  = 0.25
    lower_barrier_demarker_thirty_four = 0.30
    
    # Fisher Transform
    fisher_lookback_five        = 2
    fisher_lookback_eight       = 5
    fisher_lookback_thirteen    = 13
    fisher_lookback_twenty_one  = 21
    fisher_lookback_thirty_four = 34
    
    upper_barrier_fisher_five        =  3.618
    upper_barrier_fisher_eight       =  3.618
    upper_barrier_fisher_thirteen    =  2.618
    upper_barrier_fisher_twenty_one  =  2.618
    upper_barrier_fisher_thirty_four =  2.618
    
    lower_barrier_fisher_five        = -3.618
    lower_barrier_fisher_eight       = -3.618
    lower_barrier_fisher_thirteen    = -2.618
    lower_barrier_fisher_twenty_one  = -2.618
    lower_barrier_fisher_thirty_four = -2.618
    
    # Bollinger Bands Section
    bollinger_lookback_twenty = 20
    bollinger_lookback_thirty = 30
    bollinger_lookback_sixty  = 40
    
    standard_distance = 2.5
    
    # Market Timing
    td_flip_pattern = 9
    td_step         = 4
    
    td_alternative_pattern = 8
    td_alternative_step = 3
    
    fibonacci_pattern    = 8
    fibonacci_step_one   = 5
    fibonacci_step_two   = 3
    fibonacci_step_three = 2
    
    # Moving Average Proximity
    simple_moving_average_thirty_four            = 100
    simple_moving_average_fifty_five             = 200
    simple_moving_average_eighty_nine            = 300
    simple_moving_average_one_hundred            = 400
    simple_moving_average_one_hundred_forty_four = 500
    simple_moving_average_two_hundred            = 600
    
    exponential_moving_average_thirty_four            = 100
    exponential_moving_average_fifty_five             = 200
    exponential_moving_average_eighty_nine            = 300
    exponential_moving_average_one_hundred            = 400
    exponential_moving_average_one_hundred_forty_four = 500
    exponential_moving_average_two_hundred            = 600
    
    smoothed_moving_average_thirty_four            = 100
    smoothed_moving_average_fifty_five             = 200
    smoothed_moving_average_eighty_nine            = 300
    smoothed_moving_average_one_hundred            = 400
    smoothed_moving_average_one_hundred_forty_four = 500
    smoothed_moving_average_two_hundred            = 600

    # First Target
    atr_lookback = 14
    
    # Average True Range
    Data = atr(Data, atr_lookback, 1, 2, 3, 4)
    
    # Relative Strength Index
    Data = rsi(Data, rsi_lookback_five, close, 5)
    Data = rsi(Data, rsi_lookback_eight, close, 6)
    Data = rsi(Data, rsi_lookback_thirteen, close, 7)
    Data = rsi(Data, rsi_lookback_twenty_one, close, 8)
    Data = rsi(Data, rsi_lookback_thirty_four, close, 9)
    
    # Signal generation on the Relative Strength Index
    for i in range(len(Data)):
        
        if Data[i,   5] < lower_barrier_rsi_five:
            Data[i,  5] = -1
        elif Data[i, 5] > upper_barrier_rsi_five:
            Data[i,  5] = 1
        else:
            Data[i,  5] = 0
            
        if Data[i,   6] < lower_barrier_rsi_eight:
            Data[i,  6] = -1
        elif Data[i, 6] > upper_barrier_rsi_eight:
            Data[i,  6] = 1  
        else:
            Data[i,  6] = 0  
            
        if Data[i,   7] < lower_barrier_rsi_thirteen:
            Data[i,  7] = -1
        elif Data[i, 7] > upper_barrier_rsi_thirteen:
            Data[i,  7] = 1  
        else:
            Data[i,  7] = 0  
            
        if Data[i,   8] < lower_barrier_rsi_twenty_one:
            Data[i,  8] = -1
        elif Data[i, 8] > upper_barrier_rsi_twenty_one:
            Data[i,  8] = 1  
        else:
            Data[i,  8] = 0   
            
        if Data[i,   9] < lower_barrier_rsi_thirty_four:
            Data[i,  9] = -1
        elif Data[i, 9] > upper_barrier_rsi_thirty_four:
            Data[i,  9] = 1  
        else:
            Data[i,  9] = 0            
    '''
    '''
    
    # Stochastic Oscillator
    Data = stochastic(Data, stochastic_lookback_five, close, 10)
    Data = stochastic(Data, stochastic_lookback_eight, close, 11)
    Data = stochastic(Data, stochastic_lookback_thirteen, close, 12)
    Data = stochastic(Data, stochastic_lookback_twenty_one, close, 13)
    Data = stochastic(Data, stochastic_lookback_thirty_four, close, 14)    

    # Signal generation on the Stochastic Oscillator
    for i in range(len(Data)):
        
        if Data[i,   10] < lower_barrier_stochastic_five:
            Data[i,  10] = -1
        elif Data[i, 10] > upper_barrier_stochastic_five:
            Data[i,  10] = 1
        else:
            Data[i,  10] = 0  
            
        if Data[i,   11] < lower_barrier_stochastic_eight:
            Data[i,  11] = -1
        elif Data[i, 11] > upper_barrier_stochastic_eight:
            Data[i,  11] = 1  
        else:
            Data[i,  11] = 0
            
        if Data[i,   12] < lower_barrier_stochastic_thirteen:
            Data[i,  12] = -1
        elif Data[i, 12] > upper_barrier_stochastic_thirteen:
            Data[i,  12] = 1  
        else:
            Data[i,  12] = 0 
            
        if Data[i,   13] < lower_barrier_stochastic_twenty_one:
            Data[i,  13] = -1
        elif Data[i, 13] > upper_barrier_stochastic_twenty_one:
            Data[i,  13] = 1  
        else:
            Data[i,  13] = 0  
            
        if Data[i,   14] < lower_barrier_stochastic_thirty_four:
            Data[i,  14] = -1
        elif Data[i, 14] > upper_barrier_stochastic_thirty_four:
            Data[i,  14] = 1     
        else:
            Data[i,  14] = 0 
            
    
    '''
    '''
    
    # Demarker
    Data = demarker(Data, demarker_lookback_five, 1, 2, 15)
    Data = demarker(Data, demarker_lookback_eight, 1, 2, 16)
    Data = demarker(Data, demarker_lookback_thirteen, 1, 2, 17)
    Data = demarker(Data, demarker_lookback_twenty_one, 1, 2, 18)
    Data = demarker(Data, demarker_lookback_thirty_four, 1, 2, 19)

    # Signal generation on the Demarker
    for i in range(len(Data)):
        
        if Data[i,   15] < lower_barrier_demarker_five:
            Data[i,  15] = -1
        elif Data[i, 15] > upper_barrier_demarker_five:
            Data[i,  15] = 1
        else:
            Data[i,  15] = 0   
            
        if Data[i,   16] < lower_barrier_demarker_eight:
            Data[i,  16] = -1
        elif Data[i, 16] > upper_barrier_demarker_eight:
            Data[i,  16] = 1  
        else:
            Data[i,  16] = 0 
            
        if Data[i,   17] < lower_barrier_demarker_thirteen:
            Data[i,  17] = -1
        elif Data[i, 17] > upper_barrier_demarker_thirteen:
            Data[i,  17] = 1  
        else:
            Data[i,  17] = 0 
            
            
        if Data[i,   18] < lower_barrier_demarker_twenty_one:
            Data[i,  18] = -1
        elif Data[i, 18] > upper_barrier_demarker_twenty_one:
            Data[i,  18] = 1  
        else:
            Data[i,  18] = 0 
            
            
        if Data[i,   19] < lower_barrier_demarker_thirty_four:
            Data[i,  19] = -1
        elif Data[i, 19] > upper_barrier_demarker_thirty_four:
            Data[i,  19] = 1     
        else:
            Data[i,  19] = 0 
            
    
    '''
    '''
    
    # Fisher Transform    
    Data = fisher_transform(Data, fisher_lookback_five, 3, 20)
    Data = fisher_transform(Data, fisher_lookback_eight, 3, 21)
    Data = fisher_transform(Data, fisher_lookback_thirteen, 3, 22)
    Data = fisher_transform(Data, fisher_lookback_twenty_one, 3, 23)
    Data = fisher_transform(Data, fisher_lookback_thirty_four, 3, 24)

    # Signal generation on the Fisher Transform
    for i in range(len(Data)):
        
        if Data[i,   20] < lower_barrier_fisher_five:
            Data[i,  20] = -1
        elif Data[i, 20] > upper_barrier_fisher_five:
            Data[i,  20] = 1
        else:
            Data[i,  20] = 0 
                        
        if Data[i,   21] < lower_barrier_fisher_eight:
            Data[i,  21] = -1
        elif Data[i, 21] > upper_barrier_fisher_eight:
            Data[i,  21] = 1  
        else:
            Data[i,  21] = 0 
                        
        if Data[i,   22] < lower_barrier_fisher_thirteen:
            Data[i,  22] = -1
        elif Data[i, 22] > upper_barrier_fisher_thirteen:
            Data[i,  22] = 1  
        else:
            Data[i,  22] = 0 
            
        if Data[i,   23] < lower_barrier_fisher_twenty_one:
            Data[i,  23] = -1
        elif Data[i, 23] > upper_barrier_fisher_twenty_one:
            Data[i,  23] = 1  
        else:
            Data[i,  23] = 0 
                        
        if Data[i,   24] < lower_barrier_fisher_thirty_four:
            Data[i,  24] = -1
        elif Data[i, 24] > upper_barrier_fisher_thirty_four:
            Data[i,  24] = 1    
        else:
            Data[i,  24] = 0          

    '''
    '''

    # Bollinger Bands
    Data = BollingerBands(Data, bollinger_lookback_twenty, standard_distance, close, 25)
    
    # Signal generation on the First Bollinger
    for i in range(len(Data)):
        
        if Data[i,    3] < Data[i, 26]:
            Data[i,  26] = -1
        elif Data[i,  3] > Data[i, 26]:
            Data[i,  26] = 0
 
    for i in range(len(Data)):
        
        if Data[i,    3] > Data[i, 25]:
            Data[i,  25] = 1
        elif Data[i,  3] < Data[i, 25]:
            Data[i,  25] = 0
             
    Data = BollingerBands(Data, bollinger_lookback_thirty, standard_distance, close, 27)
    
    # Signal generation on the Second Bollinger
    for i in range(len(Data)):
        
        if Data[i,    3] < Data[i, 28]:
            Data[i,  28] = -1
        elif Data[i,  3] > Data[i, 28]:
            Data[i,  28] = 0
 
    for i in range(len(Data)):
        
        if Data[i,    3] > Data[i, 27]:
            Data[i,  27] = 1
        elif Data[i,  3] < Data[i, 27]:
            Data[i,  27] = 0    
    
    Data = BollingerBands(Data, bollinger_lookback_sixty,  standard_distance, close, 29)
    
    # Signal generation on the Third Bollinger
    for i in range(len(Data)):
        
        if Data[i,    3] < Data[i, 30]:
            Data[i,  30] = -1
        elif Data[i,  3] > Data[i, 30]:
            Data[i,  30] = 0
 
    for i in range(len(Data)):
        
        if Data[i,    3] > Data[i, 29]:
            Data[i,  29] = 1
        elif Data[i,  3] < Data[i, 29]:
            Data[i,  29] = 0      

    '''
    '''

    # TD Flip
    Data = td_flip(Data, td_flip_pattern, td_step, close, 31, 32)
    
    for i in range(len(Data)):
        
        if Data[i, 31] == -9:
            
            Data[i, 31] = -1
        else:
            Data[i, 31] = 0
            
    for i in range(len(Data)):
        
        if Data[i, 32] == 9:
            
            Data[i, 32] = 1
        else:
            Data[i, 32] = 0                        
       
    Data = td_flip(Data, td_alternative_pattern, td_alternative_step, close, 33, 34)

    for i in range(len(Data)):
        
        if Data[i, 33] == -8:
            
            Data[i, 33] = -1
        else:
            Data[i, 33] = 0
            
    for i in range(len(Data)):
        
        if Data[i, 34] == 8:
            
            Data[i, 34] = 1
        else:
            Data[i, 34] = 0  
            
    # Fibonacci Timing Pattern
    Data = fibonacci_timing_pattern(Data, fibonacci_pattern, 
                                    fibonacci_step_one, 
                                    fibonacci_step_two, 
                                    fibonacci_step_three, close, 35, 36)
    
    for i in range(len(Data)):
        
        if Data[i, 35] == -8:
            
            Data[i, 35] = -1
        else:
            Data[i, 35] = 0
            
    for i in range(len(Data)):
        
        if Data[i, 36] == 8:
            
            Data[i, 36] = 1
        else:
            Data[i, 36] = 0      

    '''
    '''

    # Parabolic SAR
    mirror = Data[:, 0:4]
    mirror = pd.DataFrame(mirror)
    mirror.columns = ['open','high','low','close']
    Parabolic = sar(mirror, 0.02, 0.2)
    Parabolic = np.array(Parabolic)
    Parabolic = np.reshape(Parabolic, (-1, 1))
    Data = np.concatenate((Data, Parabolic), axis = 1) 
    Data = adder(Data, 1)
    
    for i in range(len(Data)):
        
        if Data[i, 3] > Data[i, 37] and Data[i - 1, 3] < Data[i - 1, 37]:
            
            Data[i, 38] = -1

        if Data[i, 3] < Data[i, 37] and Data[i - 1, 3] > Data[i - 1, 37]:
            
            Data[i, 38] = 1
        
    Data = deleter(Data, 37, 1)
                      
    '''
    '''
    
    # Simple Moving Averages
    Data = ma(Data, simple_moving_average_thirty_four, close, 38)
    
    Data = adder(Data, 1)
    for i in range(len(Data)):
        
        if Data[i, 3] > Data[i, 38] and (Data[i, 3] - Data[i, 38]) < distance and Data[i - 8, 3] > Data[i - 5, 38]:
            
            Data[i, 39] = -1
            
        elif Data[i, 3] < Data[i, 38] and (Data[i, 38] - Data[i, 3]) < distance and Data[i - 8, 3] < Data[i - 5, 38]:
            
            Data[i, 39] = 1
            
        else:
            
            Data[i, 39] = 0
    Data = deleter(Data, 38, 1)
        
    Data = ma(Data, simple_moving_average_fifty_five, close, 39)
    
    Data = adder(Data, 1)
    for i in range(len(Data)):
        
        if Data[i, 3] > Data[i, 39] and (Data[i, 3] - Data[i, 39]) < distance and Data[i - 8, 3] > Data[i - 5, 39]:
            
            Data[i, 40] = -1
            
        elif Data[i, 3] < Data[i, 39] and (Data[i, 39] - Data[i, 3]) < distance and Data[i - 8, 3] < Data[i - 5, 39]:
            
            Data[i, 40] = 1
            
        else:
            
            Data[i, 40] = 0
    Data = deleter(Data, 39, 1)
            
    Data = ma(Data, simple_moving_average_eighty_nine, close, 40)
    
    Data = adder(Data, 1)
    for i in range(len(Data)):
        
        if Data[i, 3] > Data[i, 40] and (Data[i, 3] - Data[i, 40]) < distance and Data[i - 8, 3] > Data[i - 5, 40]:
            
            Data[i, 41] = -1
            
        elif Data[i, 3] < Data[i, 40] and (Data[i, 40] - Data[i, 3]) < distance and Data[i - 8, 3] < Data[i - 5, 40]:
            
            Data[i, 41] = 1
            
        else:
            
            Data[i, 41] = 0
    Data = deleter(Data, 40, 1)
    
    Data = ma(Data, simple_moving_average_one_hundred, close, 41)

    Data = adder(Data, 1)
    for i in range(len(Data)):
        
        if Data[i, 3] > Data[i, 41] and (Data[i, 3] - Data[i, 41]) < distance and Data[i - 8, 3] > Data[i - 5, 41]:
            
            Data[i, 42] = -1
            
        elif Data[i, 3] < Data[i, 41] and (Data[i, 41] - Data[i, 3]) < distance and Data[i - 8, 3] < Data[i - 5, 41]:
            
            Data[i, 42] = 1
            
        else:
            
            Data[i, 42] = 0
    Data = deleter(Data, 41, 1)
              
    Data = ma(Data, simple_moving_average_one_hundred_forty_four, close, 42)

    Data = adder(Data, 1)
    for i in range(len(Data)):
        
        if Data[i, 3] > Data[i, 42] and (Data[i, 3] - Data[i, 42]) < distance and Data[i - 8, 3] > Data[i - 5, 42]:
            
            Data[i, 43] = -1
            
        elif Data[i, 3] < Data[i, 42] and (Data[i, 42] - Data[i, 3]) < distance and Data[i - 8, 3] < Data[i - 5, 42]:
            
            Data[i, 43] = 1
            
        else:
            
            Data[i, 43] = 0
    Data = deleter(Data, 42, 1)              
    
    Data = ma(Data, simple_moving_average_two_hundred, close, 43)

    Data = adder(Data, 1)
    for i in range(len(Data)):
        
        if Data[i, 3] > Data[i, 43] and (Data[i, 3] - Data[i, 43]) < distance and Data[i - 8, 3] > Data[i - 5, 43]:
            
            Data[i, 44] = -1
            
        elif Data[i, 3] < Data[i, 43] and (Data[i, 43] - Data[i, 3]) < distance and Data[i - 8, 3] < Data[i - 5, 43]:
            
            Data[i, 44] = 1
            
        else:
            
            Data[i, 44] = 0
    Data = deleter(Data, 43, 1)
    
    # Exponential Moving Averages
    Data = ema(Data, 2, exponential_moving_average_thirty_four, close, 44)

    Data = adder(Data, 1)
    for i in range(len(Data)):
        
        if Data[i, 3] > Data[i, 44] and (Data[i, 3] - Data[i, 44]) < distance and Data[i - 8, 3] > Data[i - 5, 44]:
            
            Data[i, 45] = -1
            
        elif Data[i, 3] < Data[i, 44] and (Data[i, 44] - Data[i, 3]) < distance and Data[i - 8, 3] < Data[i - 5, 44]:
            
            Data[i, 45] = 1
            
        else:
            
            Data[i, 45] = 0
    Data = deleter(Data, 44, 1)
              
    Data = ema(Data, 2, exponential_moving_average_fifty_five, close, 45)

    Data = adder(Data, 1)
    for i in range(len(Data)):
        
        if Data[i, 3] > Data[i, 45] and (Data[i, 3] - Data[i, 45]) < distance and Data[i - 8, 3] > Data[i - 5, 45]:
            
            Data[i, 46] = -1
            
        elif Data[i, 3] < Data[i, 45] and (Data[i, 45] - Data[i, 3]) < distance and Data[i - 8, 3] < Data[i - 5, 45]:
            
            Data[i, 46] = 1
            
        else:
            
            Data[i, 46] = 0
    Data = deleter(Data, 45, 1)
              
    Data = ema(Data, 2, exponential_moving_average_eighty_nine, close, 46)

    Data = adder(Data, 1)
    for i in range(len(Data)):
        
        if Data[i, 3] > Data[i, 46] and (Data[i, 3] - Data[i, 46]) < distance and Data[i - 8, 3] > Data[i - 5, 46]:
            
            Data[i, 47] = -1
            
        elif Data[i, 3] < Data[i, 46] and (Data[i, 46] - Data[i, 3]) < distance and Data[i - 8, 3] < Data[i - 5, 46]:
            
            Data[i, 47] = 1
            
        else:
            
            Data[i, 47] = 0
    Data = deleter(Data, 46, 1)        
    
    Data = ema(Data, 2, exponential_moving_average_one_hundred, close, 47)

    Data = adder(Data, 1)
    for i in range(len(Data)):
        
        if Data[i, 3] > Data[i, 47] and (Data[i, 3] - Data[i, 47]) < distance and Data[i - 8, 3] > Data[i - 5, 47]:
            
            Data[i, 48] = -1
            
        elif Data[i, 3] < Data[i, 47] and (Data[i, 47] - Data[i, 3]) < distance and Data[i - 8, 3] < Data[i - 5, 47]:
            
            Data[i, 48] = 1
            
        else:
            
            Data[i, 48] = 0
    Data = deleter(Data, 47, 1)             
    
    Data = ema(Data, 2, exponential_moving_average_one_hundred_forty_four, close, 48)

    Data = adder(Data, 1)
    for i in range(len(Data)):
        
        if Data[i, 3] > Data[i, 48] and (Data[i, 3] - Data[i, 48]) < distance and Data[i - 8, 3] > Data[i - 5, 48]:
            
            Data[i, 49] = -1
            
        elif Data[i, 3] < Data[i, 48] and (Data[i, 48] - Data[i, 3]) < distance and Data[i - 8, 3] < Data[i - 5, 48]:
            
            Data[i, 49] = 1
            
        else:
            
            Data[i, 49] = 0
    Data = deleter(Data, 48, 1)
    
    Data = ema(Data, 2, exponential_moving_average_two_hundred, close, 49)

    Data = adder(Data, 1)
    for i in range(len(Data)):
        
        if Data[i, 3] > Data[i, 49] and (Data[i, 3] - Data[i, 49]) < distance and Data[i - 8, 3] > Data[i - 5, 49]:
            
            Data[i, 50] = -1
            
        elif Data[i, 3] < Data[i, 49] and (Data[i, 49] - Data[i, 3]) < distance and Data[i - 8, 3] < Data[i - 5, 49]:
            
            Data[i, 50] = 1
            
        else:
            
            Data[i, 50] = 0
    Data = deleter(Data, 49, 1)
              
    # Smoothed Moving Averages
    Data = ema(Data, 2, (exponential_moving_average_thirty_four * 2) - 1, close, 50)

    Data = adder(Data, 1)
    for i in range(len(Data)):
        
        if Data[i, 3] > Data[i, 50] and (Data[i, 3] - Data[i, 50]) < distance and Data[i - 8, 3] > Data[i - 5, 50]:
            
            Data[i, 51] = -1
            
        elif Data[i, 3] < Data[i, 50] and (Data[i, 50] - Data[i, 3]) < distance and Data[i - 8, 3] < Data[i - 5, 50]:
            
            Data[i, 51] = 1
            
        else:
            
            Data[i, 51] = 0
    Data = deleter(Data, 50, 1)          
    
    Data = ema(Data, 2, (exponential_moving_average_fifty_five * 2) - 1, close, 51)

    Data = adder(Data, 1)
    for i in range(len(Data)):
        
        if Data[i, 3] > Data[i, 51] and (Data[i, 3] - Data[i, 51]) < distance and Data[i - 8, 3] > Data[i - 5, 51]:
            
            Data[i, 52] = -1
            
        elif Data[i, 3] < Data[i, 51] and (Data[i, 51] - Data[i, 3]) < distance and Data[i - 8, 3] < Data[i - 5, 51]:
            
            Data[i, 52] = 1
            
        else:
            
            Data[i, 52] = 0
    Data = deleter(Data, 51, 1)
    
    Data = ema(Data, 2, (exponential_moving_average_eighty_nine * 2) - 1, close, 52)

    Data = adder(Data, 1)
    for i in range(len(Data)):
        
        if Data[i, 3] > Data[i, 52] and (Data[i, 3] - Data[i, 52]) < distance and Data[i - 8, 3] > Data[i - 5, 52]:
            
            Data[i, 53] = -1
            
        elif Data[i, 3] < Data[i, 52] and (Data[i, 52] - Data[i, 3]) < distance and Data[i - 8, 3] < Data[i - 5, 52]:
            
            Data[i, 53] = 1
            
        else:
            
            Data[i, 53] = 0
    Data = deleter(Data, 52, 1)
    
    Data = ema(Data, 2, (exponential_moving_average_one_hundred * 2) - 1, close, 53)

    Data = adder(Data, 1)
    for i in range(len(Data)):
        
        if Data[i, 3] > Data[i, 53] and (Data[i, 3] - Data[i, 53]) < distance and Data[i - 8, 3] > Data[i - 5, 53]:
            
            Data[i, 54] = -1
            
        elif Data[i, 3] < Data[i, 53] and (Data[i, 53] - Data[i, 3]) < distance and Data[i - 8, 3] < Data[i - 5, 53]:
            
            Data[i, 54] = 1
            
        else:
            
            Data[i, 54] = 0
    Data = deleter(Data, 53, 1)
    
    Data = ema(Data, 2, (exponential_moving_average_one_hundred_forty_four * 2) - 1, close, 54)

    Data = adder(Data, 1)
    for i in range(len(Data)):
        
        if Data[i, 3] > Data[i, 54] and (Data[i, 3] - Data[i, 54]) < distance and Data[i - 8, 3] > Data[i - 5, 54]:
            
            Data[i, 55] = -1
            
        elif Data[i, 3] < Data[i, 54] and (Data[i, 54] - Data[i, 3]) < distance and Data[i - 8, 3] < Data[i - 5, 54]:
            
            Data[i, 55] = 1
            
        else:
            
            Data[i, 55] = 0
    Data = deleter(Data, 54, 1)
    
    Data = ema(Data, 2, (exponential_moving_average_two_hundred * 2) - 1, close, 55) 
         
    Data = adder(Data, 1)
    for i in range(len(Data)):
        
        if Data[i, 3] > Data[i, 55] and (Data[i, 3] - Data[i, 55]) < distance and Data[i - 8, 3] > Data[i - 5, 55]:
            
            Data[i, 56] = -1
            
        elif Data[i, 3] < Data[i, 55] and (Data[i, 55] - Data[i, 3]) < distance and Data[i - 8, 3] < Data[i - 5, 55]:
            
            Data[i, 56] = 1
            
        else:
            
            Data[i, 56] = 0
    Data = deleter(Data, 55, 1)  
    
    Data = adder(Data, 1)
    
    for i in range(len(Data)):
        
        Data[i, 56] = Data[i, 5:56].sum()

    Data = deleter(Data, 5, 51)
    
    # Scaling
    for i in range(len(Data)):
        
        Data[i, 5] = (Data[i, 5] + 44) / 88
        Data[i, 5] = Data[i, 5] * 100
        
    if Data[-1, 5] <= 35:
        print(market, 'is oversold')
        
    if Data[-1, 5] >= 65:
        print(market, 'is overbought')    
        
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

def heikin_ashi_indicator(Data, start, where):
    
    Data = adder(Data, 1)
    
    for i in range(len(Data)):
        
        if (Data[i, start + 3] > Data[i, start]):
            
            Data[i, where] = 1
            
        elif (Data[i, start + 3] < Data[i, start]):

            Data[i, where] = -1
            
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

def multiple_plotting():
    
    dried_equity_curve_1 = dried_equity_curve(Asset1)
    dried_equity_curve_2 = dried_equity_curve(Asset2)
    dried_equity_curve_3 = dried_equity_curve(Asset3)
    dried_equity_curve_4 = dried_equity_curve(Asset4)
    dried_equity_curve_5 = dried_equity_curve(Asset5)
    dried_equity_curve_6 = dried_equity_curve(Asset6)
    dried_equity_curve_7 = dried_equity_curve(Asset7)
    dried_equity_curve_8 = dried_equity_curve(Asset8)
    dried_equity_curve_9 = dried_equity_curve(Asset9)
    dried_equity_curve_10 = dried_equity_curve(Asset10)
    
    plt.plot(dried_equity_curve_1[:, 1], linewidth = 1, label = assets[0])
    plt.plot(dried_equity_curve_2[:, 1], linewidth = 1, label = assets[1])
    plt.plot(dried_equity_curve_3[:, 1], linewidth = 1, label = assets[2])
    plt.plot(dried_equity_curve_4[:, 1], linewidth = 1, label = assets[3])
    plt.plot(dried_equity_curve_5[:, 1], linewidth = 1, label = assets[4])
    plt.plot(dried_equity_curve_6[:, 1], linewidth = 1, label = assets[5])
    plt.plot(dried_equity_curve_7[:, 1], linewidth = 1, label = assets[6])
    plt.plot(dried_equity_curve_8[:, 1], linewidth = 1, label = assets[7])
    plt.plot(dried_equity_curve_9[:, 1], linewidth = 1, label = assets[8])
    plt.plot(dried_equity_curve_10[:, 1], linewidth = 1, label = assets[9])
    
    plt.grid()
    plt.legend()
    plt.axhline(y = investment, color = 'black', linewidth = 1, linestyle = '--')
    plt.title('Strategy Result', fontsize = 20)
    
def global_portfolio_performance(window):
    
    unified_equity_curves_1 = Asset1[-window:, 10]
    unified_equity_curves_2 = Asset2[-window:, 10]
    unified_equity_curves_3 = Asset3[-window:, 10]
    unified_equity_curves_4 = Asset4[-window:, 10]
    unified_equity_curves_5 = Asset5[-window:, 10]
    unified_equity_curves_6 = Asset6[-window:, 10]
    unified_equity_curves_7 = Asset7[-window:, 10]
    unified_equity_curves_8 = Asset8[-window:, 10]
    unified_equity_curves_9 = Asset9[-window:, 10]
    unified_equity_curves_10 = Asset10[-window:, 10]
    
    unified_equity_curves_1 = np.reshape(unified_equity_curves_1, (-1, 1))
    unified_equity_curves_2 = np.reshape(unified_equity_curves_2, (-1, 1))
    unified_equity_curves_3 = np.reshape(unified_equity_curves_3, (-1, 1))
    unified_equity_curves_4 = np.reshape(unified_equity_curves_4, (-1, 1))
    unified_equity_curves_5 = np.reshape(unified_equity_curves_5, (-1, 1))
    unified_equity_curves_6 = np.reshape(unified_equity_curves_6, (-1, 1))
    unified_equity_curves_7 = np.reshape(unified_equity_curves_7, (-1, 1))
    unified_equity_curves_8 = np.reshape(unified_equity_curves_8, (-1, 1))
    unified_equity_curves_9 = np.reshape(unified_equity_curves_9, (-1, 1))
    unified_equity_curves_10 = np.reshape(unified_equity_curves_10, (-1, 1))
    
    global_portfolio = np.concatenate((unified_equity_curves_1, unified_equity_curves_2, unified_equity_curves_3,
                                       unified_equity_curves_4, unified_equity_curves_5, unified_equity_curves_6,
                                       unified_equity_curves_7, unified_equity_curves_8, unified_equity_curves_9,
                                       unified_equity_curves_10), axis = 1)
    
    global_portfolio = adder(global_portfolio, 1)
    
    for i in range(len(global_portfolio)):
        
        global_portfolio[i, -1] = np.sum(global_portfolio[i, 0:10])
    
    global_portfolio = global_portfolio[:, -1]
        
    global_portfolio = np.reshape(global_portfolio, (-1, 1))
        
    global_portfolio = adder(global_portfolio, 1)
        
    global_portfolio[:, 1] = investment
            
    for i in range(len(global_portfolio)):
            
        global_portfolio[i, 1] = global_portfolio[i, 0] + global_portfolio[i - 1, 1]
    
    plt.plot(global_portfolio[:, 1], linewidth = 1)
    plt.grid()
    plt.axhline(y = investment, color = 'black', linewidth = 1, linestyle = '--')
    plt.title('Global Portfolio', fontsize = 20)
        
    # Global Net Return
    net_return = round((global_portfolio[-1, 1] / investment - 1) * 100, 2)
    
    # Global Profit Factor    
    total_net_profits = global_portfolio[global_portfolio[:, 0] > 0, 0]
    total_net_losses  = global_portfolio[global_portfolio[:, 0] < 0, 0] 
    total_net_losses  = abs(total_net_losses)
    profit_factor     = round(np.sum(total_net_profits) / np.sum(total_net_losses), 2)

    # Global Hit Ratio
    hit_ratio         = len(total_net_profits) / (len(total_net_losses) + len(total_net_profits))
    hit_ratio         = round(hit_ratio, 2) * 100
    
    # Global Expectancy
    average_gain  = total_net_profits.mean()
    average_loss  = total_net_losses.mean()    
    expectancy    = (average_gain * (hit_ratio / 100)) - ((1 - (hit_ratio / 100)) * average_loss)
    expectancy    = round(expectancy, 2)
    
    # Global Maximum Loss
    max_loss = min(global_portfolio[:, 0])

    # Global Maximum Gain
    max_gain = max(global_portfolio[:, 0])
    
    # Global Maximum Drawdown
    xs = global_portfolio[:, 1]
    i  = np.argmax(np.maximum.accumulate(xs) - xs) # end of the period
    j  = np.argmax(xs[:i]) # start of period

    maximum_drawdown = (global_portfolio[i, 1] / global_portfolio[j, 1] - 1) * 100

    print('-----------Global Portfolio Performance-----------')    
    print('Global Hit Ratio         = ', hit_ratio, '%')
    print('Global Total Return      = ', net_return, '%')    
    print('Global Expectancy        = ', '$', expectancy)
    print('Global Profit factor     = ', profit_factor) 
    print('Global Maximum Drawdown  = ', round(maximum_drawdown, 2), '%')
    print('Maximum Loss             = ', '$', round(max_loss, 2))
    print('Maximum Gain             = ', '$', round(max_gain, 2))
    
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











