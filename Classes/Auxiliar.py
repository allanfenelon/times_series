# import numpy as np # linear algebra
import pandas as pd  # train processing, CSV file I/O (e.g. pd.read_csv)
from time import gmtime, strftime
# Data access


class InputOutputProcess:
    def __init__(self, root, caseLabel, dataFolderNm, resultsFolderNm, dataDelimiter):
        self.caseLabel = caseLabel
        self.dataPath = root+dataFolderNm
        self.resultsPath = root+resultsFolderNm
        self.dataDelimiter = dataDelimiter
        self.train_val_data = self.getData(phase='train', isToView=False)

    def getData(self, phase='train', isToView=False):
        ret = pd.read_csv(
            self.dataPath+self.caseLabel+'/'+phase+'.csv', delimiter=self.dataDelimiter)
        if isToView:
            print(self.ret.head())
        return ret


class KaggleSubmission:
    def __init__(self, modelingObj, idVarNm):
        self.modelingObj = modelingObj
        self.idVarNm = idVarNm
        self.test_data = modelingObj.pp.iops.getData(phase='test')
        # print(self.test_data.head())
        vars = modelingObj.pp.qlVarsNms + modelingObj.pp.qtVarsNms
        self.t_test_data = modelingObj.pp.ql_qt_pipeline.fit_transform(
            self.test_data[vars])
        # print(self.t_test_data)

    def save_Kaggle_submission_file(self):
        index = 0
        for model in self.modelingObj.models:
            y_pred = []
            try:
                y_pred = [round(y_hat)
                          for y_hat in model.predict(self.t_test_data)]
            except:
                try:
                    y_pred = [round(y_hat[0])
                              for y_hat in model.predict(self.t_test_data)]
                except:
                    for index, row in self.t_test_data.iterrows():
                        g = 1
                        # y_pred.append(round(bbnPredict(row)))

            reg_submission = pd.DataFrame({self.idVarNm: self.test_data[self.idVarNm],
                                           self.modelingObj.pp.targetsNms: y_pred})
            #    reg_submission.head()
            timeStr = strftime("%Y%m%d_%H%M%S", gmtime())
            filePath = self.modelingObj.pp.iops.resultsPath+self.modelingObj.pp.iops.caseLabel
            filePath+='/' + timeStr + '_' + self.modelingObj.models_label[index] + '_' + type(model).__name__ +'_submission.csv'
            index += 1
            reg_submission.to_csv(filePath, index=False)
            print(timeStr + '_' + type(model).__name__ +'_submission.csv saved!!')
