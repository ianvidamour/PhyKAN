from .Experiment import Experiment
import os
import tqdm
import shutil
import torch
import numpy as np

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class   ExperimentWrapper():
    def __init__(self, configPath: str, configFile: str, repeats: int):
        
        self.repeats = repeats
        
        self.configFile = configFile
        self.configPath = configPath
        
        self.outputFolder = "Results/" + configFile
        self.outputFolder = self.ensure_unique_folder()
        
        #self.experiment = Experiment(self.configPath, self.configFile, self.outputFolder)

    def ensure_unique_folder(self):
        """Ensures a unique folder name by appending an integer if necessary."""
        folder = self.outputFolder
        count = 1
        while os.path.exists(folder):
            folder = f"{self.outputFolder}_{count}"
            count += 1  
        os.makedirs(folder)
        
        shutil.copy(self.configPath + self.configFile + '.yaml', folder + "/" + self.configFile + '.yaml')
        
        return folder
    
    def run(self):
        """Call function to run experiment class for x repeats"""
        print(f"Experiment started!")
        for i in tqdm.trange(self.repeats):
            set_seed(i)
            e = Experiment(self.configPath, self.configFile, self.outputFolder, seed=i)
            e.run(i)
        print(f"Completed... :ß")
