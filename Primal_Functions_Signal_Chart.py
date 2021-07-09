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

def signal_chart_ohlc_color(Data, name, onwhat, what_bull, what_bear, window = 1000):   
     
    Plottable = Data[-window:, ]
    
    fig, ax = plt.subplots(figsize = (10, 5))
    
    ohlc_plot(Data, window, '')    

    for i in range(len(Plottable)):
        
        if Plottable[i, what_bull] == 1:
            
            x = i
            y = Plottable[i, onwhat]
        
            ax.annotate(' ', xy = (x, y), 
                        arrowprops = dict(width = 9, headlength = 11, headwidth = 11, facecolor = 'green', color = 'green'))
        
        elif Plottable[i, what_bear] == -1:
            
            x = i
            y = Plottable[i, onwhat]
        
            ax.annotate(' ', xy = (x, y), 
                        arrowprops = dict(width = 9, headlength = -11, headwidth = -11, facecolor = 'red', color = 'red'))
                     
    ax.set_facecolor((0.95, 0.95, 0.95)) 
    plt.legend()    
