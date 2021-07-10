
# Base parameters

assets           = asset_list(1) 
window           = 1000

# Trading parameters   
horizon          = 'H1'

# Risk parameters
stop             = 10
target           = 0.5
risk_lookback    = 2

# Indicator / Strategy parameters
lookback         = 14
upper_barrier    = 70
lower_barrier    = 30

# Mass imports 
Asset1 = mass_import(0, horizon)
Asset2 = mass_import(1, horizon)
Asset3 = mass_import(2, horizon)
Asset4 = mass_import(3, horizon)
Asset5 = mass_import(4, horizon)
Asset6 = mass_import(5, horizon)
Asset7 = mass_import(6, horizon)
Asset8 = mass_import(7, horizon)
Asset9 = mass_import(8, horizon)
Asset10 = mass_import(9, horizon)

def signal(Data, what, buy, sell):
    
    Data = adder(Data, 10)
    Data = rounding(Data, 5)
    
    for i in range(len(Data)):
            
        if Data[i, what] < lower_barrier and Data[i - 1, buy] == 0 and \
            Data[i - 2, buy] == 0 and Data[i - 3, buy] == 0 and Data[i - 4, buy] == 0:
            Data[i, buy] = 1
            
        elif Data[i, what] > upper_barrier and Data[i - 1, sell] == 0 and \
            Data[i - 2, sell] == 0 and Data[i - 3, sell] == 0 and Data[i - 4, sell] == 0:
            Data[i, sell] = -1    
            
    return Data      

##############################################################################   1

Asset1 = atr(Asset1, risk_lookback, 1, 2, 3, 4)
Asset1 = rsi(Asset1, lookback, 3, 5, genre = 'Smoothed')
Asset1 = signal(Asset1, 5, 6, 7)

holding(Asset1, 6, 7, 8, 9)
holding_fix(Asset1)
risk_management_atr(Asset1, stop, target, 4)
Asset1_eq = equity_curve(Asset1, expected_cost, lot, investment)
performance(Asset1_eq, Asset1, assets[0])

if sigchart == True:
    signal_chart_ohlc_color(Asset1, assets[0], 3, 6, 7, window = 500)
    indicator_plot_double(Asset1, 3, 5, name = assets[0], name_ind = '', window = 500)    
    plt.axhline(y = 1.2175, color = 'black', linewidth = 1, linestyle = '--')
    plt.axhline(y = lower_barrier, color = 'black', linewidth = 1, linestyle = '--')    
    plot_signal_equity_curve(Asset1, Asset1_eq, 6, 7, window = 200) 

##############################################################################   2

Asset2 = atr(Asset2, risk_lookback, 1, 2, 3, 4)
Asset2 = rsi(Asset2, lookback, 3, 5)
Asset2 = signal(Asset2, 5, 6, 7)

holding(Asset2, 6, 7, 8, 9)
holding_fix(Asset2)
risk_management_atr(Asset2, stop, target, 4)
Asset2_eq = equity_curve(Asset2, expected_cost, lot, investment)
performance(Asset2_eq, Asset2, assets[1])

if sigchart == True:
    signal_chart_ohlc_color(Asset2, assets[1], 3, 6, 7, window = 500)
    indicator_plot_double(Asset2, 3, 5, name = assets[1], name_ind = '', window = 250)    
    plt.axhline(y = 0.8960, color = 'black', linewidth = 1, linestyle = '--')
    plt.axhline(y = lower_barrier, color = 'black', linewidth = 1, linestyle = '--')    
    plot_signal_equity_curve(Asset2, Asset2_eq, 6, 7, window = 200)  

##############################################################################   3

Asset3 = atr(Asset3, risk_lookback, 1, 2, 3, 4)
Asset3 = rsi(Asset3, lookback, 3, 5)
Asset3 = signal(Asset3, 5, 6, 7)

holding(Asset3, 6, 7, 8, 9)
holding_fix(Asset3)
risk_management_atr(Asset3, stop, target, 4)
Asset3_eq = equity_curve(Asset3, expected_cost, lot, investment)
performance(Asset3_eq, Asset3, assets[2])

if sigchart == True:
    signal_chart_ohlc_color(Asset3, assets[2], 3, 6, 7, window = 250)
    indicator_plot_double(Asset3, 3, 5, name = assets[2], name_ind = '', window = 250)    
    plt.axhline(y = upper_barrier, color = 'black', linewidth = 1, linestyle = '--')
    plt.axhline(y = lower_barrier, color = 'black', linewidth = 1, linestyle = '--')    
    plot_signal_equity_curve(Asset3, Asset3_eq, 6, 7, window = 200) 
    
##############################################################################   4

Asset4 = atr(Asset4, risk_lookback, 1, 2, 3, 4)
Asset4 = rsi(Asset4, lookback, 3, 5)
Asset4 = signal(Asset4, 5, 6, 7)

holding(Asset4, 6, 7, 8, 9)
holding_fix(Asset4)
risk_management_atr(Asset4, stop, target, 4)
Asset4_eq = equity_curve(Asset4, expected_cost, lot, investment)
performance(Asset4_eq, Asset4, assets[3])

if sigchart == True:
    signal_chart_ohlc_color(Asset4, assets[3], 3, 6, 7, window = 250)
    indicator_plot_double(Asset4, 3, 5, name = assets[3], name_ind = '', window = 250)    
    plt.axhline(y = upper_barrier, color = 'black', linewidth = 1, linestyle = '--')
    plt.axhline(y = lower_barrier, color = 'black', linewidth = 1, linestyle = '--')    
    plot_signal_equity_curve(Asset4, Asset4_eq, 6, 7, window = 200) 

##############################################################################   5

Asset5 = atr(Asset5, risk_lookback, 1, 2, 3, 4)
Asset5 = rsi(Asset5, lookback, 3, 5)
Asset5 = signal(Asset5, 5, 6, 7)

holding(Asset5, 6, 7, 8, 9)
holding_fix(Asset5)
risk_management_atr(Asset5, stop, target, 4)
Asset5_eq = equity_curve(Asset5, expected_cost, lot, investment)
performance(Asset5_eq, Asset5, assets[4])

if sigchart == True:
    signal_chart_ohlc_color(Asset5, assets[4], 3, 6, 7, window = 250)
    indicator_plot_double(Asset5, 3, 5, name = assets[4], name_ind = '', window = 250)    
    plt.axhline(y = upper_barrier, color = 'black', linewidth = 1, linestyle = '--')
    plt.axhline(y = lower_barrier, color = 'black', linewidth = 1, linestyle = '--')    
    plot_signal_equity_curve(Asset5, Asset5_eq, 6, 7, window = 200) 

##############################################################################   6

Asset6 = atr(Asset6, risk_lookback, 1, 2, 3, 4)
Asset6 = rsi(Asset6, lookback, 3, 5)
Asset6 = signal(Asset6, 5, 6, 7)

holding(Asset6, 6, 7, 8, 9)
holding_fix(Asset6)
risk_management_atr(Asset6, stop, target, 4)
Asset6_eq = equity_curve(Asset6, expected_cost, lot, investment)
performance(Asset6_eq, Asset6, assets[5])

if sigchart == True:
    signal_chart_ohlc_color(Asset6, assets[5], 3, 6, 7, window = 250)
    indicator_plot_double(Asset6, 3, 5, name = assets[5], name_ind = '', window = 250)    
    plt.axhline(y = upper_barrier, color = 'black', linewidth = 1, linestyle = '--')
    plt.axhline(y = lower_barrier, color = 'black', linewidth = 1, linestyle = '--')    
    plot_signal_equity_curve(Asset6, Asset6_eq, 6, 7, window = 200) 

##############################################################################   7

Asset7 = atr(Asset7, risk_lookback, 1, 2, 3, 4)
Asset7 = rsi(Asset7, lookback, 3, 5)
Asset7 = signal(Asset7, 5, 6, 7)

holding(Asset7, 6, 7, 8, 9)
holding_fix(Asset7)
risk_management_atr(Asset7, stop, target, 4)
Asset7_eq = equity_curve(Asset7, expected_cost, lot, investment)
performance(Asset7_eq, Asset7, assets[6])

if sigchart == True:
    signal_chart_ohlc_color(Asset7, assets[6], 3, 6, 7, window = 250)
    indicator_plot_double(Asset7, 3, 5, name = assets[6], name_ind = '', window = 250)    
    plt.axhline(y = upper_barrier, color = 'black', linewidth = 1, linestyle = '--')
    plt.axhline(y = lower_barrier, color = 'black', linewidth = 1, linestyle = '--')    
    plot_signal_equity_curve(Asset7, Asset7_eq, 6, 7, window = 200) 

##############################################################################   8

Asset8 = atr(Asset8, risk_lookback, 1, 2, 3, 4)
Asset8 = rsi(Asset8, lookback, 3, 5)
Asset8 = signal(Asset8, 5, 6, 7)

holding(Asset8, 6, 7, 8, 9)
holding_fix(Asset8)
risk_management_atr(Asset8, stop, target, 4)
Asset8_eq = equity_curve(Asset8, expected_cost, lot, investment)
performance(Asset8_eq, Asset8, assets[7])

if sigchart == True:
    signal_chart_ohlc_color(Asset8, assets[7], 3, 6, 7, window = 250)
    indicator_plot_double(Asset8, 3, 5, name = assets[7], name_ind = '', window = 250)    
    plt.axhline(y = upper_barrier, color = 'black', linewidth = 1, linestyle = '--')
    plt.axhline(y = lower_barrier, color = 'black', linewidth = 1, linestyle = '--')    
    plot_signal_equity_curve(Asset8, Asset8_eq, 6, 7, window = 200) 

##############################################################################   9

Asset9 = atr(Asset9, risk_lookback, 1, 2, 3, 4)
Asset9 = rsi(Asset9, lookback, 3, 5)
Asset9 = signal(Asset9, 5, 6, 7)

holding(Asset9, 6, 7, 8, 9)
holding_fix(Asset9)
risk_management_atr(Asset9, stop, target, 4)
Asset9_eq = equity_curve(Asset9, expected_cost, lot, investment)
performance(Asset9_eq, Asset9, assets[8])

if sigchart == True:
    signal_chart_ohlc_color(Asset9, assets[8], 3, 6, 7, window = 250)
    indicator_plot_double(Asset9, 3, 5, name = assets[8], name_ind = '', window = 250)    
    plt.axhline(y = upper_barrier, color = 'black', linewidth = 1, linestyle = '--')
    plt.axhline(y = lower_barrier, color = 'black', linewidth = 1, linestyle = '--')    
    plot_signal_equity_curve(Asset9, Asset9_eq, 6, 7, window = 200) 

##############################################################################   10

Asset10 = atr(Asset10, risk_lookback, 1, 2, 3, 4)
Asset10 = rsi(Asset10, lookback, 3, 5)
Asset10 = signal(Asset10, 5, 6, 7)

holding(Asset10, 6, 7, 8, 9)
holding_fix(Asset10)
risk_management_atr(Asset10, stop, target, 4)
Asset10_eq = equity_curve(Asset10, expected_cost, lot, investment)
performance(Asset10_eq, Asset10, assets[9])

if sigchart == True:
    signal_chart_ohlc_color(Asset10, assets[9], 3, 6, 7, window = 250)
    indicator_plot_double(Asset10, 3, 5, name = assets[9], name_ind = '', window = 250)    
    plt.axhline(y = upper_barrier, color = 'black', linewidth = 1, linestyle = '--')
    plt.axhline(y = lower_barrier, color = 'black', linewidth = 1, linestyle = '--')    
    plot_signal_equity_curve(Asset10, Asset10_eq, 6, 7, window = 200) 

##############################################################################

plt.plot(Asset1_eq[:, 3], linewidth = 1, label = assets[0])
plt.plot(Asset2_eq[:, 3], linewidth = 1, label = assets[1])
plt.plot(Asset3_eq[:, 3], linewidth = 1, label = assets[2])
plt.plot(Asset4_eq[:, 3], linewidth = 1, label = assets[3])
plt.plot(Asset5_eq[:, 3], linewidth = 1, label = assets[4])
plt.plot(Asset6_eq[:, 3], linewidth = 1, label = assets[5])
plt.plot(Asset7_eq[:, 3], linewidth = 1, label = assets[6])
plt.plot(Asset8_eq[:, 3], linewidth = 1, label = assets[7])
plt.plot(Asset9_eq[:, 3], linewidth = 1, label = assets[8])
plt.plot(Asset10_eq[:, 3], linewidth = 1, label = assets[9])
plt.grid()
plt.legend()
plt.axhline(y = investment, color = 'black', linewidth = 1)
plt.title('Relative Strength Strategy', fontsize = 20)