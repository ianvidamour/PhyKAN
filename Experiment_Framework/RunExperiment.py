from Experiment import *

#exp1 = Experiment( './Configs/', 'MLPOneLayerNoisy', 'MLPOneLayerNoisyTrainingOutput.csv', 'MLPOneLayerNoisyOutput.csv' )

#exp1 = Experiment( './Configs/', 'KanOneLayerRefined', 'KANTrainingOutput1LayerRefined.csv', 'KANValidationOutput1LayerRefined.csv' )

#exp1 = Experiment( './Configs/', 'KanOneLayer', 'KANTrainingOutput1Layer.csv', 'KANValidationOutput1Layer.csv' )
#exp2 = Experiment( './Configs/', 'KanTwoLayer', 'KANTrainingOutput2Layer.csv', 'KANValidationOutput2Layer.csv' )

#exp3 = Experiment( './Configs/', 'MLPOneLayer', 'MLPTrainingOutput1Layer.csv', 'PLEASEVALIDATE.csv' )

exp4 = ExperimentWrapper( './Configs/', 'PhyKanOneLayer', 1)
exp4.run()

#exp5 = ExperimentWrapper( './Configs/', 'MLPOneLayer', 1)
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
