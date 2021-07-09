
import datetime
import pytz
import pandas            as pd
import MetaTrader5       as mt5
import matplotlib.pyplot as plt
import numpy             as np
from scipy.stats                 import pearsonr
from scipy.stats                 import spearmanr
from scipy.ndimage.interpolation import shift
import statistics as stats
from scipy import stats

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
    
    if asset_set == 1:
        assets = ['EURUSD', 'USDCHF', 'GBPUSD', 'EURNZD', 'GBPCHF',
                  'USDCAD', 'EURCAD', 'EURGBP', 'EURCHF', 'AUDCAD']
        
    elif asset_set == 2:
        assets = ['EURNZD', 'NZDCHF', 'NZDCAD', 'EURAUD','AUDNZD', 
                  'GBPCAD', 'AUDCHF', 'GBPAUD', 'GBPCHF', 'GBPNZD']
        
    elif asset_set == 3:
        assets = ['BTCUSD', 'ETHUSD', 'XRPUSD', 'ETCUSD','LTCUSD', 
                  'MBTUSD', 'XMRUSD', 'ZECUSD', 'EOSUSD', 'EMCUSD']
        
    elif asset_set == 4:
        assets = ['XAUUSD', 'XAGUSD', 'XPTUSD', 'XPDUSD']    
        
    return assets

def mass_import(asset, horizon):
    
    if horizon == 'MN1':
        data = get_quotes(frame_MIN1, 2021, 5, 1, asset = assets[asset])
        data = data.iloc[:, 1:5].values
        data = data.round(decimals = 5)    
    
    if horizon == 'M5':
        data = get_quotes(frame_M5, 2021, 1, 1, asset = assets[asset])
        data = data.iloc[:, 1:5].values
        data = data.round(decimals = 5)

    if horizon == 'M10':
        data = get_quotes(frame_M10, 2020, 6, 1, asset = assets[asset])
        data = data.iloc[:, 1:5].values
        data = data.round(decimals = 5)
        
    if horizon == 'M15':
        data = get_quotes(frame_M15, 2020, 1, 1, asset = assets[asset])
        data = data.iloc[:, 1:5].values
        data = data.round(decimals = 5)
        
    if horizon == 'M30':
        data = get_quotes(frame_M30, 2016, 1, 1, asset = assets[asset])
        data = data.iloc[:, 1:5].values
        data = data.round(decimals = 5)        
        
    if horizon == 'H1':
        data = get_quotes(frame_H1, 2010, 5, 1, asset = assets[asset])
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

assets           = asset_list(1) 
horizon          = 'H1'
test             = mass_import(0, horizon)
