from Experiment import *

#exp4 = ExperimentWrapper( './Configs/', 'PhyKanOneLayer', 1)
#exp4.run()

exp = ExperimentWrapper( './Configs/', 'MLPOneLayer', 1)
exp.run()

exp1 = ExperimentWrapper( './Configs/', 'MLPTwoLayer', 1)
exp1.run()

exp2 = ExperimentWrapper( './Configs/', 'KanOneLayer', 1)
exp2.run()

exp3 = ExperimentWrapper( './Configs/', 'KanTwoLayer', 1)
exp3.run()

exp4 = ExperimentWrapper( './Configs/', 'PhyKanOneLayer', 1)
exp4.run()

exp5 = ExperimentWrapper( './Configs/', 'PhyKanTwoLayer', 1)
exp5.run()
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
