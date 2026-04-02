import os
import polars as pl
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import pyarrow

class AnalyticsPipe():
    def __init__(self, resultsPath: str, outName: str):
        self.resultsPath = resultsPath
        self.outName = outName
    def run(self):
        self.createFrames()
        self.createAverages()
        self.train_combined.to_csv(self.outName + "train.csv")
        self.val_combined.to_csv(self.outName + "val.csv")
        
        #self.plotTraining()
        #self.plotValidation()

    def createFrames(self):
        self.train_combined = []
        self.val_combined = []
    
        for file in os.listdir(self.resultsPath):
            file_path = os.path.join(self.resultsPath, file)

            if "train" in file.lower():  # Check if "train" is in the filename
                #self.train_combined = 
                self.train_combined.append(pl.read_csv(file_path))

            elif "val" in file.lower():  # Check if "val" is in the filename
                #self.train_combined = 
                self.val_combined.append(pl.read_csv(file_path))

        # Combine DataFrames for "train" and "val"
        self.train_combined = pl.concat(self.train_combined) # if self.train_combined else None
        self.val_combined = pl.concat(self.val_combined) # if self.train_combined else None

    def createAverages(self):

        self.train_combined = self.train_combined.group_by('steps', 'epoch').agg([
            pl.mean('accuracy').alias('Avg Accuracy'), # Sugar for pl.col('a').mean()
            pl.mean('loss').alias('Avg Loss'),
            pl.mean('layer_0_dimensionality').alias('Avg Layer 0 Dim'),
            pl.mean('layer_1_dimensionality').alias('Avg Layer 1 Dim')]).sort(by='steps')
        
        self.val_combined = self.val_combined.group_by('steps', 'epoch').agg([
            pl.mean('accuracy'),
            pl.mean('loss'),
            pl.mean('layer_0_dimensionality'),
            pl.mean('layer_1_dimensionality')]).sort(by='steps')
        
        self.train_combined = self.train_combined.with_columns([
            self.train_combined[col].rolling_mean(window_size=4, center=True).alias(f"{col}_moving_avg") 
            for col in self.train_combined.columns])

        self.val_combined = self.val_combined.with_columns([
            self.val_combined[col].rolling_mean(window_size=4, center=True).alias(f"{col}_moving_avg") 
            for col in self.val_combined.columns])

        self.train_combined = self.train_combined.to_pandas()
        self.val_combined = self.val_combined.to_pandas()

    def plotTraining(self):
    # Convert to pandas for plotting

        # Extract the 'steps' column and the rolling average columns
        step_col = "steps"
        metric_cols = [col for col in self.train_combined.columns if col not in ["epoch", "steps"]]

        # Set up the figure size
        sns.set_style("whitegrid")

        for col in metric_cols:
            fig, axes = plt.subplots(1, 2, figsize=(20, 12))

            #self.train_combined[col] = gaussian_filter1d(self.train_combined[col], sigma=1)

            # Regular Scale
            sns.lineplot(data=self.train_combined, x=step_col, y=col, ax=axes[0])
            axes[0].set_title(f"{col} vs {step_col} (Regular Scale)")
            axes[0].set_xlabel(step_col)
            axes[0].set_ylabel(col)

            # Log Scale
            sns.lineplot(data=self.train_combined, x=step_col, y=col, ax=axes[1])
            axes[1].set_xscale("log")
            axes[1].set_title(f"{col} vs {step_col} (Log Scale)")
            axes[1].set_xlabel(step_col)
            axes[1].set_ylabel(col)

            plt.tight_layout()
            plt.show()

    def plotValidation(self):
    # Convert to pandas for plotting
        
        # Extract the 'steps' column and the rolling average columns
        step_col = "steps"
        metric_cols = [col for col in self.val_combined.cols if col not in ["epoch", "steps"]]

        # Set up the figure size
        sns.set_style("whitegrid")

        for col in metric_cols:
            fig, axes = plt.subplots(1, 2, figsize=(18, 5))
            plt.gca().set_xlim(left=0, right=10000)

            #self.train_combined[col] = gaussian_filter1d(self.train_combined[col], sigma=1)
            #smoothed_data = gaussian_filter1d(self.val_combined[col].to_numpy(), sigma=20)
            #data=self.val_combined
            # Regular Scale
            sns.lineplot(data=self.val_combined, x=self.val_combined[step_col], y=col, ax=axes[0])
            axes[0].set_title(f"{col} vs {step_col} (Regular Scale)")
            axes[0].set_xlabel(step_col)
            axes[0].set_ylabel(col)

            # Log Scale
            sns.lineplot(data=self.val_combined, x=self.val_combined[step_col], y=col, ax=axes[1])
            axes[1].set_xscale("log")
            axes[1].set_title(f"{col} vs {step_col} (Log Scale)")
            axes[1].set_xlabel(step_col)
            axes[1].set_ylabel(col)

            plt.tight_layout()
            plt.show()

'''            # Cumulative Sum
            sns.lineplot(data=self.train_combined.with_columns(self.train_combined[col].cumsum().alias(f"Cumulative {col}")), 
                         x=step_col, y=f"Cumulative {col}", ax=axes[2])
            axes[2].set_title(f"Cumulative {col} vs {step_col}")
            axes[2].set_xlabel(step_col)
            axes[2].set_ylabel(f"Cumulative {col}")'''
           

        #self.train_combined.write_csv(self.resultsPath + '/trainingCombined')
        #self.train_combined.with_columns(self.train_combined["accuracy"].rolling_mean(window_size=3, min_periods=2).alias("acc_moving_avg"))

        #pl.DataFrame().with_columns(self.val_combined[col].rolling_mean)

        #pl.DataFrame().write_csv()
    
