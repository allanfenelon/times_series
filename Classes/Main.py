from Auxiliar import InputOutputProcess
from Preprocessing import *
from Modeling import *

iops = InputOutputProcess(root='D:/OneDrive - Universidade Federal do Cariri - UFCA/Drive/UFCA/Ensino/CRAN R_aulas/RClasses/',
                          caseLabel="titanic",
                          dataFolderNm="data/",
                          resultsFolderNm="results/",
                          dataDelimiter=',')
iops.readData(phase='train', isToView=True)
# iops.train_val_data.head()
pp = Preprocessing(iops=iops,
                   targetsNms='Survived',
                   qlVarsNms=['Pclass', 'Sex', 'Embarked'],
                   qtVarsNms=['Fare', 'Age', 'Parch', 'SibSp'], 
                   targetsAreBinary = True)
pp.tv_split()
pp.transform_tv_x()
# print(pp.m)
modeling = Modeling(pp)
modeling.compute_model(label='mTree0',
                       modelObj=DecisionTreeRegressor(random_state=0))
modeling.compute_model(label='mTree1',
                       modelObj=DecisionTreeRegressor(random_state=0, max_depth=5))
modeling.compute_model(label='mTree2',
                       modelObj=DecisionTreeRegressor(random_state=0, max_depth=3))
