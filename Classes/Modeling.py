import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_log_error, mean_squared_error, mean_absolute_error, r2_score, accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from tensorflow import keras
# !pip install scikit-optimize
from skopt import BayesSearchCV
from skopt.plots import plot_objective
from skopt.space import Real, Categorical, Integer
import tensorflow as tf
# import pickle
import joblib


class Modeling:
    def __init__(self, pp):
        self.pp = pp
        # models performance
        self.models = []
        self.models_label = []
        self.models_r2_t = []
        self.models_r2_v = []
        self.models_rmse_t = []
        self.models_rmse_v = []
        self.models_rmsle_t = []
        self.models_rmsle_v = []
        self.models_mae_t = []
        self.models_mae_v = []
   # Fitting

    def get_model(self, label, modelObj, optObj=None,
                  verbose=False, epochs=150, batch_size=32,
                  validation_split=.2, n_jobs=4, buildForce=False):
        formalism = type(modelObj).__name__
        fittingObj = optObj if optObj is not None else modelObj
        if formalism == 'XGBRegressor' or formalism == 'LGBMRegressor':
            eval_set = [(self.pp.v_x, self.pp.v_y)]
            fittingObj.fit(self.pp.t_train_x, self.pp.t_y, eval_set=eval_set, early_stopping_rounds=2,
                           eval_metric="rmsle", verbose=verbose)
        elif formalism == 'KerasRegressor':
            early_stopping_cb = keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=10, restore_best_weights=True)
            fittingObj.fit(self.pp.t_train_x, self.pp.t_y, epochs=epochs, batch_size=batch_size,
                           validation_split=validation_split, callbacks=[early_stopping_cb])
        else:
            try:
                fittingObj.fit(self.pp.t_train_x, self.pp.t_y)
            except Exception:
                try:
                    fittingObj.fit(self.pp.t_train_x, self.pp.t_y.values)
                except Exception:
                    try:
                        fittingObj.fit(self.pp.t_train_x,
                                       self.pp.t_y.values.ravel())
                    except Exception:
                        print('>>>>>> Trouble in ', formalism,
                              ' <<<<<<< \n', Exception)
                        fittingObj = None
                    # pass
        self.save_model(model=fittingObj, label=label)
        return (fittingObj)

    def get_opt_model(self, modelObj, label, parsDists,
                      verbose=False, epochs=150, batch_size=32,
                      validation_split=.2, n_jobs=4, buildForce=False):
        model = None if buildForce else self.get_saved_model(modelObj)
        if model is None:
            # log-uniform: understand as search over p = exp(x) by varying x
            opt = BayesSearchCV(modelObj, parsDists, n_iter=32,  cv=3, return_train_score=True,
                                scoring='neg_mean_squared_log_error', verbose=verbose, n_jobs=n_jobs)
            formalism = type(modelObj).__name__
            if formalism == 'XGBRegressor' or formalism == 'LGBRegressor':
                eval_set = [(self.pp.t_val_x, self.pp.v_y)]
                opt.fit(self.pp.t_train_x, self.pp.t_y, eval_set=eval_set,
                        verbose=verbose, early_stopping_rounds=2)
            elif formalism == 'KerasRegressor':
                early_stopping_cb = keras.callbacks.EarlyStopping(
                    monitor='val_loss', patience=10, restore_best_weights=True)
                opt.fit(self.pp.t_train_x, self.pp.t_y, epochs=epochs, batch_size=batch_size,
                        validation_split=validation_split, callbacks=[early_stopping_cb])
            else:
                try:
                    opt.fit(self.pp.t_train_x, self.pp.t_y)
                except Exception:
                    try:
                        opt.fit(self.pp.t_train_x, self.pp.t_y.values.ravel())
                    except Exception:
                        print('>>>>>> Trouble in ', formalism,
                              ' <<<<<<< \n', Exception)
                        return None
                        # pass
            # parNames = [name for name, value in parsDists.items()]
            # _ = plot_objective(opt.optimizer_results_[0],
            #                 dimensions=parNames,
            #                 n_minimum_search=int(1e8))
            # plt.show()
            print('best_params= ', opt.best_params_)
            print('best_score (RMSLE) = ', np.sqrt(-opt.best_score_))
            model = opt.best_estimator_
            self.save_model(model=model, label=label)
            # saveKaggleSubmissionFile(models=[model])
        return (model)

    # https://mljar.com/blog/save-load-random-forest/
    def save_model(self, model, folder='/', label=''):
        # for model in models:
        modelNm = type(model).__name__ + '_' + label
        modelPath = self.pp.iops.resultsPath + \
            self.pp.iops.caseLabel+folder+modelNm + '.joblib'
        # with open(modelPath, 'wb') as f:
        # pickle.dump(model, f)
        joblib.dump(model, modelPath, compress=3)
        print('>>> ', modelNm, ' saved!')
    # models loading

    def get_saved_model(self, emptModel, folder='/', label=''):
        savedModel = None
        modelNm = type(emptModel).__name__ + '_' + label
        # for model in emptModels:
        modelPath = self.pp.iops.resultsPath + \
            self.pp.iops.caseLabel+folder+modelNm + '.joblib'
        # with open(modelPath, 'rb') as f:
        try:
            savedModel = joblib.load(modelPath)
        except Exception:
            print('>>> TROUBLE IN getSavedModel: \n', Exception)
            pass
        return (savedModel)

    # models prediction
    # l_ts_train_validation = pd.read_csv("../data/storeSalesTS/l_ts_train_validation.csv")
    # l_ts_train_validation.set_index('date', inplace=True)
    # l_targetsNms = l_ts_train_validation.columns#[('l_' + targetNm) for targetNm in targetsNms]

    # zdl_scaler = joblib.load('../results/storeSalesTS/zdl_scaler.joblib')
    # n_zdl_scaler_repetition = round(zdl_y_train.shape[1]/len(zdl_scaler.mean_))
    # zdl_scaler_mean_ = (list(zdl_scaler.mean_)*n_zdl_scaler_repetition)
    # zdl_scaler_scale_ = (list(zdl_scaler.scale_)*n_zdl_scaler_repetition)

    # ly_minus_1 = sum(l_ts_train_validation[l_targetsNms][:n_zdl_scaler_repetition].values.tolist(), [])#[0]*(n_zdl_scaler_repetition*len(zdl_targetsNms))#
    # ly_n_1 = sum(l_ts_train_validation[l_targetsNms][(n-n_zdl_scaler_repetition):n].values.tolist(), [])#[0]*(n_zdl_scaler_repetition*len(zdl_targetsNms))#
    # ly_nm_1 = [l_ts_train_validation[l_targetsNms][(n+m-1):(n+m)].values.tolist()[0]]#[0]*(n_zdl_scaler_repetition*len(zdl_targetsNms))#
    # ly_nm_1 += [[0]*(len(ly_minus_1) - len(ly_nm_1[0]))]#It must be enhanced
    # ly_nm_1 = sum(ly_nm_1, [])

    def get_prediction(self, model, x, phase='train'):
        y_pred = model.predict(x)  # zdl-based transformed TS forecasts
        if type(y_pred).__name__ == 'ndarray':
            if self.pp.targetsAreNonNegative:
                # if predictions must be non-negative
                y_pred = [max([0, y_ij]) for y_ij in y_pred]
            if self.pp.targetsAreInteger:
                # if predictions must be integer numbers
                y_pred = [round(y_ij) for y_ij in y_pred]
            if self.pp.targetsAreBinary:
                # if predictions must be binary
                y_pred = [0 if y_ij <= .5 else 1 for y_ij in y_pred]
        else:
            if self.pp.targetsAreNonNegative:
                for i in range(len(y_pred)):
                    # if predictions must be non-negative
                    y_pred[i] = [max([0, y_ij]) for y_ij in y_pred[i]]
            if self.pp.targetsAreInteger:
                for i in range(len(y_pred)):
                    # if predictions must be integer numbers
                    y_pred[i] = [round(y_ij) for y_ij in y_pred[i]]
            if self.pp.targetsAreBinary:
                for i in range(len(y_pred)):
                    # if predictions must be binary
                    y_pred[i] = [0 if y_ij <= .5 else 1 for y_ij in y_pred[i]]
        # y_pred = np.ndarray(shape=zdly_pred.shape, dtype=float)#forecasts of the original TS
        # ly_t_1 = None
        # if phase=='train':
        #     ly_t_1 = ly_minus_1
        # elif phase=='validation':
        #     ly_t_1 = ly_n_1
        # elif phase == 'test':
        #     ly_t_1 = ly_nm_1
        # try:
        #     # dly_t_1_pred = [0]*len(zdly_pred[0])
        #     for t in range(len(zdly_pred)):
        #         zdly_pred_t = zdly_pred[t]
        #         y_pred_t = np.ndarray(shape=zdly_pred_t.shape, dtype=float)
        #         for j in range(len(y_pred_t)):
        #             zdly_pred_tj = zdly_pred_t[j]
        #             m_j = zdl_scaler_mean_[j]
        #             s_j = zdl_scaler_scale_[j]
        #             dly_pred_tj = zdly_pred_tj*s_j + m_j
        #             # if j==1782 and t==1:
        #             #     g=1
        #             ly_pred_tj = dly_pred_tj + ly_t_1[j]
        #             ly_t_1[j] = ly_pred_tj
        #             y_pred_tj = np.expm1(ly_pred_tj)
        #             if np.isfinite(y_pred_tj):
        #                 y_pred_tj = max(0, y_pred_tj)
        #             else:
        #                 y_pred_tj = sys.float_info.max
        #             y_pred_t[j] = y_pred_tj
        #         y_pred[t] = y_pred_t
        #         # ly_t_1 = l_ts_train_validation[l_targetsNms].iloc[t+t0,:].to_list()
        #         # print('|t='+str(t)+' ok.. ', end='')
        #     print(' |'+phase+' ok.. ', end='')
        # except Exception:
        #     print('>>>> TROUBLE IN getSalePrediction', Exception)
        # # print(y_pred[:3])
        return y_pred

    # x_train_origin, x_valid_origin, y_train_origin, self.pp.v_y, n, m = get_usual_ts_train_validation_split(
    #     isToPrintShapes=True, targetsNms=zdl_lagTargetsNms, featuresNms=zdl_featuresNms,
    #     ts_train_validation = zdl_ts_train_validation)

    def print_performance_measures(self, model, label):
        # for model in models:
        y_train_predict = self.get_prediction(
            model, self.pp.t_train_x, phase='train')
        y_valid_predict = self.get_prediction(
            model, self.pp.t_val_x, phase='validation')
        # pd.DataFrame({'y_train_predict': y_train_predict, 'pp.t_y': self.pp.t_y})

        formalism = type(model).__name__
        self.models_label.append(formalism+'_'+label)
        self.models_r2_t.append(
            float(format(r2_score(y_true=self.pp.t_y, y_pred=y_train_predict), '.2g')))
        self.models_r2_v.append(
            float(format(r2_score(y_true=self.pp.v_y, y_pred=y_valid_predict),  '.2g')))
        self.models_rmse_t.append(float(format(
            mean_squared_error(y_true=self.pp.t_y, y_pred=y_train_predict, squared=False), '.2g')))
        self.models_rmse_v.append(float(format(
            mean_squared_error(y_true=self.pp.v_y, y_pred=y_valid_predict, squared=False), '.2g')))
        self.models_rmsle_t.append(float(format(
            mean_squared_log_error(y_true=self.pp.t_y, y_pred=y_train_predict, squared=False), '.2g')))
        self.models_rmsle_v.append(float(format(
            mean_squared_log_error(y_true=self.pp.v_y, y_pred=y_valid_predict, squared=False), '.2g')))
        self.models_mae_t.append(float(format(mean_absolute_error(
            y_true=self.pp.t_y, y_pred=y_train_predict), '.2g')))
        self.models_mae_v.append(float(format(mean_absolute_error(
            y_true=self.pp.v_y, y_pred=y_valid_predict), '.2g')))
        df = pd.DataFrame({'model': self.models_label,
                           'rmsle-t': self.models_rmsle_t,
                           'rmsle-v': self.models_rmsle_v,
                           'r2-t': self.models_r2_t,
                           'r2-v': self.models_r2_v,  # }  # ,
                           #    'rmse-t': models_rmse_t,
                           #    'rmse-v': models_rmse_v,
                           'mae-t': self.models_mae_t,
                           'mae-v': self.models_mae_v}
                          )
        print(" *********** Rank via RMSLE in the validation set *********** ")
        # .head(len(df))
        print(df.sort_values(
            by=["rmsle-v", 'r2-v'], ascending=True))

    def compute_model(self, label, modelObj=DecisionTreeRegressor(random_state=0, min_samples_leaf=.01),
                      optDists=None, verbose=False, epochs=150, batch_size=32,
                      validation_split=.2, n_jobs=4, buildForce=False):
        model = None
        if optDists is None:
            model = self.get_model(label = label, modelObj=modelObj, optObj=optDists,
                                   verbose=verbose, epochs=epochs, batch_size=batch_size,
                                   validation_split=validation_split, n_jobs=n_jobs, buildForce=buildForce)
        else:
            model = self.get_opt_model(label = label, modelObj=modelObj, parsDists=optDists,
                                       verbose=verbose, epochs=epochs, batch_size=batch_size,
                                       validation_split=validation_split, n_jobs=n_jobs, buildForce=buildForce)
        self.models.append(model)
        # PLOTING THE TREE
        # fig = plt.figure(figsize=(25, 20))
        # _ = tree.plot_tree(tree_reg, max_depth=1, feature_names=x_train.columns,
        #                    class_names=targetsNms, filled=True)
        # self.save_model(model=model, label=label)
        self.print_performance_measures(model=model, label=label)
        modelNm = type(model).__name__
        if (modelNm == 'DecisionTreeRegressor') or (modelNm == 'RandomForestRegressor') or (modelNm == 'XGBRegressor') or (modelNm == 'LGBRegressor'):
            df = pd.DataFrame(data=model.feature_importances_,
                              index=self.pp.ql_qt_pipeline.get_feature_names_out(),  # xgb_reg.feature_names_in_,
                              columns=['importance']).sort_values(by='importance', ascending=False)
            # print(modelNm, label, 'feature_importances_: \n', df)
