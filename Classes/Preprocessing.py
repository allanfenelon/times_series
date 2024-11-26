import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler  # , MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split


class Preprocessing:
    def __init__(self, iops, targetsNms, qlVarsNms=None,
                 qtVarsNms=None, pTrain=.8, 
                 targetsAreNonNegative=False, targetsAreInteger=False, targetsAreBinary=False):  # , isToPrintShapes=False):
        self.iops = iops
        self.pTrain = pTrain
        self.n_m = iops.train_val_data.shape[0]
        self.n = round(pTrain*self.n_m)
        self.m = self.n_m - self.n
        self.targetsNms = targetsNms
        self.targetsAreNonNegative = targetsAreNonNegative
        self.targetsAreInteger = targetsAreInteger
        self.targetsAreBinary = targetsAreBinary
        self.qlVarsNms = qlVarsNms
        self.qtVarsNms = qtVarsNms
        self.tv_y = iops.train_val_data[targetsNms] #if (len(targetsNms)>1) else tvDf[[targetsNms]]
        self.tv_x = iops.train_val_data.drop(columns=targetsNms)

    def tv_split(self, featuresNms=None, targetsNms=None):
        if featuresNms is None:
            featuresNms = self.tv_x.columns
        if targetsNms is None:
            targetsNms = self.targetsNms
        tv_x = self.tv_x[featuresNms]
        tv_y = self.tv_y 
        if type(targetsNms).__name__!='str':
            tv_y = self.tv_y[targetsNms]
        self.t_x, self.v_x, self.t_y, self.v_y = train_test_split(
            tv_x, tv_y, test_size=(1-self.pTrain), random_state=0)
        # if isToPrintShapes:
        print('self.t_x.shape: ', self.t_x.shape)
        print('self.t_y.shape: ', self.t_y.shape)
        print('self.v_x.shape: ', self.v_x.shape)
        print('self.v_y.shape: ', self.v_y.shape)
        # return x_train, x_valid, y_train, y_valid
    # x_train, x_valid, y_train, y_valid = get_sklearn_train_test_split(isToPrintShapes=True)

    def transform_tv_x(self):
        types = self.t_x.dtypes
        if self.qlVarsNms is None:
            isQuali = (types == "object")
            self.qlVarsNms = isQuali[isQuali].index
        if self.qtVarsNms is None:
            isQuanti = (types != "object")
            self.qtVarsNms = isQuanti[isQuanti].index
            # isQuanti = isQuanti.drop(['Survived', 'PassengerId', 'Age'])
            # isQuali = isQuali.drop(['Name', 'Ticket', 'Cabin'])
        print(">>> self.qlVarsNms: ", self.qlVarsNms)
        print(">>> self.qtVarsNms: ", self.qtVarsNms)
        self.tv_ql_x = self.tv_x[self.qlVarsNms]
        self.tv_qt_x = self.tv_x[self.qtVarsNms]
        self.qt_pipeline = Pipeline([('imputer', SimpleImputer(strategy="median", add_indicator=True)),
                                     ('std_scaler', StandardScaler())])  # MinMaxScaler(feature_range  = (.4, .6))
        self.ql_pipeline = Pipeline([('ohe', OneHotEncoder(sparse=False)),
                                     ('std_scaler', StandardScaler())])  # MinMaxScaler(feature_range  = (.4, .6))
        self.ql_qt_pipeline = ColumnTransformer([("qt", self.qt_pipeline, self.qtVarsNms),
                                                 ("ql", self.ql_pipeline, self.qlVarsNms)])
        # print(self.ql_qt_pipeline)
        self.t_train_x = pd.DataFrame(self.ql_qt_pipeline.fit_transform(self.t_x),
                                 columns=self.ql_qt_pipeline.get_feature_names_out())
        self.t_val_x = pd.DataFrame(self.ql_qt_pipeline.transform(self.v_x),
                                 columns=self.ql_qt_pipeline.get_feature_names_out())
        # print('self.t_train_x.describe:\n', self.t_train_x.describe())
        # print('self.t_val_x.describe:\n', self.t_val_x.describe())

    def get_transformed_new_x(self, newX):
        return self.ql_qt_pipeline.transform(newX)
