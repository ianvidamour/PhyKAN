import os
from Experiment import *

exp4 = ExperimentWrapper( './Configs/', 'PhyKanOneLayer', 1)
exp4.run()

#exp = ExperimentWrapper( './Configs/', 'MLPOneLayer', 1)
#exp.run()

#exp1 = ExperimentWrapper( './Configs/', 'MLPTwoLayer', 2)
#exp1.run()

#exp2 = ExperimentWrapper( './Configs/', 'KanOneLayer', 1)
#exp2.run()

#exp3 = ExperimentWrapper( './Configs/', 'KanTwoLayer', 1)
#exp3.run()

#exp4 = ExperimentWrapper( './Configs/', 'PhyKanOneLayer', 3)
#exp4.run()
#curdir = os.getcwd()
#configdir = curdir + "/Configs/PKOneLayer/"
#configs = os.listdir(configdir)
#
#for dir in configs:
#    dir = dir.replace(".yaml", "")
#    ExperimentWrapper(configdir, dir, 10).run()

#ExperimentWrapper(configdir, "/PhyKanOneLayer1FL0", 1).run()

#a = AnalyticsPipe('./Results/MLPOneLayer_5')
#a.run()

#print('Running Kan Experiment 1!')
#exp1.trainingLoop()
#print('Running Kan Experiment 2!')
#exp2.trainingLoop()

#print('Running MLP Experiment 1!')
#exp3.trainingLoop()
#print('Running MLP Experiment 1!')
#exp4.trainingLoop()
