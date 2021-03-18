import pandas as pd
GLOP_S = pd.read_csv('results_GLOP_S.csv')
GLOP_TIME_S  = '[22:23<00:00, 122.10s/it]'
GLOP_SCORE_S = GLOP_S['score'].mean()
print('GLOP:',GLOP_TIME_S,GLOP_SCORE_S )
SCIP_S= pd.read_csv('results_SCIP_S.csv')
SCIP_TIME_S  = '[1:30:19<00:00, 492.65s/it]'
SCIP_SCORE_S = SCIP_S['score'].mean()
print('SCIP:',SCIP_TIME_S,SCIP_SCORE_S )
CLP_S = pd.read_csv('results_CLP_S.csv')
CLP_TIME_S  = '[22:57<00:00, 125.26s/it]'
CLP_SCORE_S = CLP_S['score'].mean()
print('CLP:',CLP_TIME_S,CLP_SCORE_S)
CBC_S = pd.read_csv('results_CBC_S.csv')
CBC_TIME_S  = '[29:45<00:00, 162.29s/it]'
CBC_SCORE_S = CBC_S['score'].mean()
print('CBC:',CBC_TIME_S,CBC_SCORE_S )
GLPK_S = pd.read_csv('results_GLPK_S.csv')
GLPK_TIME_S  = '[2:32:46<00:00, 833.30s/it]'
GLPK_SCORE_S = GLPK_S['score'].mean()
print('GLPK:',GLPK_TIME_S,GLPK_SCORE_S )