import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

from PreProcessing import PreProcessor
from FeatureExtraction import FeatureExtractor
from FeatureSelector import FeatureSelector
from AccuracyChecking import AccuracyChecker

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer


class Main:
    keys = ['city', 'year', 'weekofyear']

    all_features = ['city', 'year', 'weekofyear', 'week_start_date', 'ndvi_ne', 'ndvi_nw',
                    'ndvi_se', 'ndvi_sw', 'precipitation_amt_mm', 'reanalysis_air_temp_k',
                    'reanalysis_avg_temp_k', 'reanalysis_dew_point_temp_k',
                    'reanalysis_max_air_temp_k', 'reanalysis_min_air_temp_k',
                    'reanalysis_precip_amt_kg_per_m2',
                    'reanalysis_relative_humidity_percent', 'reanalysis_sat_precip_amt_mm',
                    'reanalysis_specific_humidity_g_per_kg', 'reanalysis_tdtr_k',
                    'station_avg_temp_c', 'station_diur_temp_rng_c', 'station_max_temp_c',
                    'station_min_temp_c', 'station_precip_mm']

    features = ['reanalysis_dew_point_temp_k', 'reanalysis_precip_amt_kg_per_m2',
                'reanalysis_specific_humidity_g_per_kg',
                'station_avg_temp_c', 'station_max_temp_c', 'station_min_temp_c']

    new_features = ['recent_mean_dew_point', 'recent_mean_spec_humid', 'recent_sum_precip']

    time_series_features = ['week_start_date']

    drop_features = list(set(all_features) - set(features) - set(keys) - set(time_series_features))
    impute_columns = ['reanalysis_avg_temp_c', 'reanalysis_max_air_temp_c',
                      'reanalysis_min_air_temp_c']

    def __init__(self):
        pass

    def processor(self):
        pre_processor = PreProcessor()
        feature_extractor = FeatureExtractor()
        feature_selector = FeatureSelector()
        accuracy_checker = AccuracyChecker()
        y_train, x_train_sj, y_train_sj, x_train_iq, y_train_iq, x_test_sj, x_test_iq = self.read_data()

        x_train_sj = pre_processor.impute_redundant_features(x_train_sj, self.impute_columns)
        x_train_iq = pre_processor.impute_redundant_features(x_train_iq, self.impute_columns)

        x_test_sj = pre_processor.impute_redundant_features(x_test_sj, self.impute_columns)
        x_test_iq = pre_processor.impute_redundant_features(x_test_iq, self.impute_columns)

        imputer_sj = Imputer(strategy='mean')
        x_train_sj = pre_processor.impute_missing_values(x_train_sj, self.features, imputer_sj)
        x_test_sj = pre_processor.impute_missing_values(x_test_sj, self.features, imputer_sj)

        imputer_iq = Imputer(strategy='mean')
        x_train_iq = pre_processor.impute_missing_values(x_train_iq, self.features, imputer_iq)
        x_test_iq = pre_processor.impute_missing_values(x_test_iq, self.features, imputer_iq)

        x_train_sj = feature_extractor.add_time_series_features(x_train_sj, window=100)
        x_train_iq = feature_extractor.add_time_series_features(x_train_iq, window=30)
        x_test_sj = feature_extractor.add_time_series_features(x_test_sj, window=100)
        x_test_iq = feature_extractor.add_time_series_features(x_test_iq, window=30)

        x_train_sj = feature_selector.drop_unnecessary_features(x_train_sj, self.drop_features, self.time_series_features)
        x_train_iq = feature_selector.drop_unnecessary_features(x_train_iq, self.drop_features, self.time_series_features)
        x_test_sj = feature_selector.drop_unnecessary_features(x_test_sj, self.drop_features, self.time_series_features)
        x_test_iq = feature_selector.drop_unnecessary_features(x_test_iq, self.drop_features, self.time_series_features)

        features_to_normalize = self.features + self.new_features

        x_train_sj[features_to_normalize] = x_train_sj[features_to_normalize].apply(pre_processor.normalize, axis=0)
        x_train_iq[features_to_normalize] = x_train_iq[features_to_normalize].apply(pre_processor.normalize, axis=0)
        x_test_sj[features_to_normalize] = x_test_sj[features_to_normalize].apply(pre_processor.normalize, axis=0)
        x_test_iq[features_to_normalize] = x_test_iq[features_to_normalize].apply(pre_processor.normalize, axis=0)

        x_train = pd.concat([x_train_sj, x_train_iq], axis=0)
        x_train.set_index('index', inplace=True)

        x_sj, y_sj = x_train.loc[x_train.city == 'sj', :], y_train.loc[x_train.city == 'sj', :]
        x_iq, y_iq = x_train.loc[x_train.city == 'iq', :], y_train.loc[x_train.city == 'iq', :]

        x_train_sj, x_cross_sj, y_train_sj, y_cross_sj = train_test_split(x_sj,
                                                                          y_sj,
                                                                          test_size=0.2,
                                                                          stratify=x_sj.weekofyear)

        x_train_iq, x_cross_iq, y_train_iq, y_cross_iq = train_test_split(x_iq,
                                                                          y_iq,
                                                                          test_size=0.2,
                                                                          stratify=x_iq.weekofyear)

        x_train_sj = feature_selector.select_features(x_train_sj, self.features, self.new_features)
        x_train_iq = feature_selector.select_features(x_train_iq, self.features, self.new_features)
        x_cross_sj = feature_selector.select_features(x_cross_sj, self.features, self.new_features)
        x_cross_iq = feature_selector.select_features(x_cross_iq, self.features, self.new_features)

        reg_sj = RandomForestRegressor(max_depth=None, n_estimators=700, random_state=67)
        accuracy_checker.cross_validate_out_of_sample(reg_sj, x_train_sj, y_train_sj.total_cases, x_cross_sj, y_cross_sj.total_cases)

        reg_iq = RandomForestRegressor(max_depth=7, n_estimators=700, random_state=67)
        accuracy_checker.cross_validate_out_of_sample(reg_iq, x_train_iq, y_train_iq.total_cases, x_cross_iq, y_cross_iq.total_cases)

        predict_sj = x_test_sj[self.keys].copy()
        predict_iq = x_test_iq[self.keys].copy()

        x_sj = feature_selector.select_features(x_sj, self.features, self.new_features)
        x_iq = feature_selector.select_features(x_iq, self.features, self.new_features)
        x_test_sj = feature_selector.select_features(x_test_sj, self.features, self.new_features)
        x_test_iq = feature_selector.select_features(x_test_iq, self.features, self.new_features)

        y_sj_pred, y_iq_pred = self.model_trainor(x_sj, y_sj, x_iq, y_iq, x_test_sj, x_test_iq)
        predict_sj['total_cases'] = y_sj_pred.round().astype(int)
        predict_iq['total_cases'] = y_iq_pred.round().astype(int)

        predict_df = pd.concat([predict_sj, predict_iq], axis=0)
        predict_df.loc[predict_df.total_cases < 0, 'total_cases'] = 0

        self.write_results(predict_df)

    def read_data(self):

        pd.set_option('display.max_columns', 100)

        mpl.rc(group='figure', figsize=(10, 8))
        plt.style.use('seaborn')

        x_train = pd.read_csv('Data/dengue_features_train.csv')
        x_train.week_start_date = pd.to_datetime(x_train.week_start_date)

        y_train = pd.read_csv('Data/dengue_labels_train.csv',
                              usecols=['total_cases'])

        x_test = pd.read_csv('Data/dengue_features_test.csv')
        x_test.week_start_date = pd.to_datetime(x_test.week_start_date)


        x_train = x_train.reset_index()
        x_test = x_test.reset_index()

        x_train_sj = x_train.loc[x_train.city == 'sj', :].copy()
        x_train_iq = x_train.loc[x_train.city == 'iq', :].copy()

        y_train_sj = y_train.loc[x_train.city == 'sj', :].copy()
        y_train_iq = y_train.loc[x_train.city == 'iq', :].copy()

        x_test_sj = x_test.loc[x_test.city == 'sj', :].copy()
        x_test_iq = x_test.loc[x_test.city == 'iq', :].copy()

        return y_train, x_train_sj, y_train_sj, x_train_iq, y_train_iq, x_test_sj, x_test_iq

    def model_trainor(self, x_sj, y_sj, x_iq, y_iq, x_test_sj, x_test_iq):

        reg_sj = GradientBoostingRegressor(learning_rate=0.1, max_depth=5, n_estimators=500, random_state=67)
        reg_sj.fit(x_sj, y_sj.total_cases)

        filename = 'sj.sav'
        joblib.dump(reg_sj, "Models/" + filename)

        reg_iq = GradientBoostingRegressor(learning_rate=0.1, max_depth=3, n_estimators=300, random_state=67)
        reg_iq.fit(x_iq, y_iq.total_cases)

        filename = 'iq.sav'
        joblib.dump(reg_iq, "Models/" + filename)

        # reg_sj = RandomForestRegressor(max_depth=None, n_estimators=500, random_state=67)
        # reg_sj.fit(x_sj, y_sj.total_cases)
        # reg_iq = RandomForestRegressor(max_depth=None, n_estimators=500, random_state=67)
        # reg_iq.fit(x_iq, y_iq.total_cases)

        y_sj_pred = reg_sj.predict(x_test_sj)
        y_iq_pred = reg_iq.predict(x_test_iq)

        return y_sj_pred, y_iq_pred

    def write_results(self, predict_df):
        submission_filename = 'Results/result_rf.csv'
        predict_df.to_csv(submission_filename, index=False)

        df1 = pd.read_csv('Data/submission_format.csv',
                          usecols=[0, 1, 2], header=0, names=['format_city', 'format_year', 'format_weekofyear'])

        df2 = pd.read_csv(submission_filename,
                          usecols=[0, 1, 2], header=0, names=['submit_city', 'submit_year', 'submit_weekofyear'])

        df = pd.merge(df1, df2, how='left',
                      left_on=['format_city', 'format_year', 'format_weekofyear'],
                      right_on=['submit_city', 'submit_year', 'submit_weekofyear'])


Main().processor()



