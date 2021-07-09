def holding(Data, buy, sell, buy_return, sell_return):

    for i in range(len(Data)):
        try:
            if Data[i, buy] == 1: 
                for a in range(i + 1, i + 1000):                        
                    if Data[a, buy] != 0 or Data[a, sell] != 0:
                        Data[a, buy_return] = (Data[a, 3] - Data[i, 3])

                        break                        
                    else:
                        continue
                
            elif Data[i, sell] == -1:        
                for a in range(i + 1, i + 1000):                        
                    if Data[a, buy] != 0 or Data[a, sell] != 0:
                        Data[a, sell_return] = (Data[i, 3] - Data[a, 3])
                      
                        break                                        
                    else:
                        continue                                         
        except IndexError:
            pass
        
def equity_curve(Data, pnl, expected_cost, lot, investment):
    
    # Charting portfolio evolution  
    indexer = Data[:, pnl:pnl + 2]    
    
    # Creating a combined array for long and short returns
    z = np.zeros((len(Data), 1), dtype = float)
    indexer = np.append(indexer, z, axis = 1)
    
    # Combining Returns
    for i in range(len(indexer)):
        try:    
            if indexer[i, 0] != 0:
                indexer[i, 2] = indexer[i, 0] - (expected_cost / lot)
                
            if indexer[i, 1] != 0:
                indexer[i, 2] = indexer[i, 1] - (expected_cost / lot)
        except IndexError:
            pass
        
    # Switching to monetary values
    indexer[:, 2] = indexer[:, 2] * lot
    
    # Creating a portfolio balance array
    indexer = np.append(indexer, z, axis = 1)
    indexer[:, 3] = investment 
    
    # Adding returns to the balance    
    for i in range(len(indexer)):
    
        indexer[i, 3] = indexer[i - 1, 3] + (indexer[i, 2])
    
    indexer = np.array(indexer)
    
    return np.array(indexer)
   
def performance(indexer, pnl, Data, name):
    
    # Profitability index
    indexer = np.delete(indexer, 0, axis = 1)
    indexer = np.delete(indexer, 0, axis = 1)
    
    profits = []
    losses  = []
    np.count_nonzero(Data[:, 7])
    np.count_nonzero(Data[:, 8])
    
    for i in range(len(indexer)):
        
        if indexer[i, 0] > 0:
            value    = indexer[i, 0]
            profits  = np.append(profits, value)
            
        if indexer[i, 0] < 0:
            value    = indexer[i, 0]
            losses   = np.append(losses, value)
    
    # Hit ratio calculation
    hit_ratio = round((len(profits) / (len(profits) + len(losses))) * 100, 2)
    
    realized_risk_reward = round(abs(profits.mean() / losses.mean()), 2)
    
    # Expected and total profits / losses
    expected_profits = np.mean(profits)
    expected_losses  = np.abs(np.mean(losses))
    total_profits    = round(np.sum(profits), 3)
    total_losses     = round(np.abs(np.sum(losses)), 3)
    
    # Expectancy
    expectancy    = round((expected_profits * (hit_ratio / 100)) \
                       - (expected_losses * (1 - (hit_ratio / 100))), 2)
        
    # Largest Win and Largest Loss
    largest_win = round(max(profits), 2)
    largest_loss = round(min(losses), 2)

    # Total Return
    indexer = Data[:, pnl:pnl + 2]    
    
    # Creating a combined array for long and short returns
    z = np.zeros((len(Data), 1), dtype = float)
    indexer = np.append(indexer, z, axis = 1)
    
    # Combining Returns
    for i in range(len(indexer)):
        try:    
            if indexer[i, 0] != 0:
                indexer[i, 2] = indexer[i, 0] - (expected_cost / lot)
                
            if indexer[i, 1] != 0:
                indexer[i, 2] = indexer[i, 1] - (expected_cost / lot)
        except IndexError:
            pass
        
    # Switching to monetary values
    indexer[:, 2] = indexer[:, 2] * lot
    
    # Creating a portfolio balance array
    indexer = np.append(indexer, z, axis = 1)
    indexer[:, 3] = investment 
    
    # Adding returns to the balance    
    for i in range(len(indexer)):
    
        indexer[i, 3] = indexer[i - 1, 3] + (indexer[i, 2])
    
    indexer = np.array(indexer)
    
    total_return = (indexer[-1, 3] / indexer[0, 3]) - 1
    total_return = total_return * 100
    
    
    print('-----------Performance-----------', name)
    
    print('Hit ratio       = ', hit_ratio, '%')
    
    print('Net profit      = ', '$', round(indexer[-1, 3] - indexer[0, 3], 2))
    
    print('Expectancy      = ', '$', expectancy, 'per trade')
 
    print('Profit factor   = ' , round(total_profits / total_losses, 2)) 
    
    print('Total Return    = ', round(total_return, 2), '%')
    
    print('')    
    
    print('Average Gain    = ', '$', round((expected_profits), 2), 'per trade')

    print('Average Loss    = ', '$', round((expected_losses * -1), 2), 'per trade')

    print('Largest Gain    = ', '$', largest_win)

    print('Largest Loss    = ', '$', largest_loss)    
    
    print('')
    
    print('Realized RR     = ', realized_risk_reward)
    
    print('Minimum         =', '$', round(min(indexer[:, 3]), 2))
    
    print('Maximum         =', '$', round(max(indexer[:, 3]), 2))
    
    print('Trades          =', len(profits) + len(losses))        