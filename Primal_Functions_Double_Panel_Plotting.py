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
     
    ax[1].plot(Data[-window:, second_panel], color = 'black', linewidth = 1)
    ax[1].grid()
    ax[1].legend()