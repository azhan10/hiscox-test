
"""

The module includes the statistics to generate the data.

"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns


from pathlib import Path

np.random.seed(1889)


class Analytics:
    
    
    """
    
    The constructor contains the global variables and creates the finalised the dataframe
    
    Some variables are placed to make it easier to code the logic (e.g. min max sclaing)
    
    Hyperparameters:
        - data -> the data in dataframe object
    
    
    """
    def __init__(self, data):
        self.data = data
        
    """
    
    Produces pairwise relationships between a column and save to a file
    
    Hyperparameters:
        - column -> the column you wish to pairwise with
    
    Returns true if sucessfully produced. Otherwise, returns false
    
    Azhan's comments:
        - I can return it as a file but keep in mind, it's file size. I tried making the size 10, it takes a long time
    
    """
    def display_pairwise_relationships(self, column) -> bool:
        
        try:
            sns.pairplot(self.data, hue=column)
            plt.title(f"Pair plot for {column}")
            plt.savefig(self.data_path() + f"pair plot for {column}.pdf")
            plt.show()
        except:
            return False
        
        return True
    
    
    """
    
    Produces the correlation between columns
    
    Hyperparameters:
        - exclude_columns -> the column to not include in the correlation
    
    Returns the correlation scores between columns in dataframe
    
    """
    def get_correlation(self, exclude_columns) -> pd.DataFrame:
        return self.data.drop(exclude_columns, axis=1, errors='ignore').corr()
    
    
    """
    
    Produces the heatmap between columns with a given correlation
    
    Hyperparameters:
        - correlation -> the correlation matrix in dataframe object
        
    Returns true if heatmap produced successfully. Otherwise, returns false
    
    
    """
    def display_heatmap(self, correlation) -> bool:
        try:
            upper_triangular = np.triu(np.ones_like(correlation, dtype=bool))
            
            colour_map = sns.diverging_palette(230, 20, as_cmap=True)
            
            sns.heatmap(
                correlation,
                mask=upper_triangular,
                cmap=colour_map,
                vmax=0.3,
                center=0,
                square=True,
                linewidths=0.5,
                cbar_kws={"shrink": 0.5},
            )
            plt.title(f"Heatmap correlation")
            plt.savefig(self.data_path() + f"heatmap correlation.pdf")
            plt.show()
        except:
            return False
        
        return True
    
    
    """
    
    Produces the boxplots and histograms with a selection list of columns
    
    Hyperparameters:
        - columns -> the list of columns to display plots
    
    Returns true if the plots are produced successfully. Otherwise, returns false
    
    """
    def display_boxplots_and_histogram(self, columns) -> bool:
        
        try:
            for column in columns:
                fig, ax = plt.subplots(1, 2, figsize=(6, 4))
                sns.boxplot(data=self.data, y=column, orient="v", ax=ax[0])
                sns.histplot(self.data, x=column, kde=True, ax=ax[1])
                plt.suptitle(f"Boxplot and Histogram for {column}")
                plt.savefig(self.data_path() + f"boxplot and histogram for {column}.pdf")
                plt.show()
        except:
            return False
        
        return True 
    
    
    """
    
    Returns the data file path
    
    """
    def data_path(self) -> str:
        return str(Path(__file__).parent.parent) + "/data/"
        
        
    
    
        
        
        
        