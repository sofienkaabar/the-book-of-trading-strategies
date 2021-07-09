def ohlc_plot(Data, window, name):
      
    Chosen = Data[-window:, ]
    
    for i in range(len(Chosen)):
        
        plt.vlines(x = i, ymin = Chosen[i, 2], ymax = Chosen[i, 1], color = 'black', linewidth = 1)  
        
        if Chosen[i, 3] > Chosen[i, 0]:
            color_chosen = 'blue'
            plt.vlines(x = i, ymin = Chosen[i, 0], ymax = Chosen[i, 3], color = color_chosen, linewidth = 3)  

        if Chosen[i, 3] < Chosen[i, 0]:
            color_chosen = 'brown'
            plt.vlines(x = i, ymin = Chosen[i, 3], ymax = Chosen[i, 0], color = color_chosen, linewidth = 3)  
            
        if Chosen[i, 3] == Chosen[i, 0]:
            color_chosen = 'black'
            plt.vlines(x = i, ymin = Chosen[i, 3], ymax = Chosen[i, 0], color = color_chosen, linewidth = 3)  
            
    plt.grid()
    plt.title(name)