def adder(Data, times):
    
    for i in range(1, times + 1):
    
        z = np.zeros((len(Data), 1), dtype = float)
        Data = np.append(Data, z, axis = 1)

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