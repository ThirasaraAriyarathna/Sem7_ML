class FeatureExtractor:

    def __init__(self):
        pass

    def add_time_series_features(self, df, window):

        df.set_index('week_start_date', inplace=True)
        roll_df = df.rolling(window=window, min_periods=1)
        df['recent_mean_dew_point'] = roll_df.reanalysis_dew_point_temp_k.mean()
        df['recent_mean_spec_humid'] = roll_df.reanalysis_specific_humidity_g_per_kg.mean()
        df['recent_sum_precip'] = roll_df.reanalysis_precip_amt_kg_per_m2.sum()
        df.reset_index(inplace=True)
        return df