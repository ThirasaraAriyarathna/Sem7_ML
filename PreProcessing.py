class PreProcessor:

    def __init__(self):
        pass

    def impute_redundant_features(self, df, impute_columns):

        df['reanalysis_avg_temp_c'] = df.reanalysis_avg_temp_k - 273.15
        df.reanalysis_avg_temp_c -= (df.reanalysis_avg_temp_c - df.station_avg_temp_c).mean()
        df.loc[df.station_avg_temp_c.isnull(), 'station_avg_temp_c'] = df.reanalysis_avg_temp_c

        df['reanalysis_max_air_temp_c'] = df.reanalysis_max_air_temp_k - 273.15
        df.reanalysis_max_air_temp_c -= (df.reanalysis_max_air_temp_c - df.station_max_temp_c).mean()
        df.loc[df.station_max_temp_c.isnull(), 'station_max_temp_c'] = df.reanalysis_max_air_temp_c

        df['reanalysis_min_air_temp_c'] = df.reanalysis_min_air_temp_k - 273.15
        df.reanalysis_min_air_temp_c -= (df.reanalysis_min_air_temp_c - df.station_min_temp_c).mean()
        df.loc[df.station_min_temp_c.isnull(), 'station_min_temp_c'] = df.reanalysis_min_air_temp_c

        # Drop the temporary columns that we just added
        df.drop(impute_columns, axis=1, inplace=True)

        return df

    def impute_missing_values(self, df, features, imputer):
        imputer.fit(df[features])
        df[features] = imputer.transform(df[features])
        return df

    def normalize(self, feature):
        return (feature - feature.mean()) / feature.std()

