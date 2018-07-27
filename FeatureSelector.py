class FeatureSelector:

    def __init__(self):
        pass

    def drop_unnecessary_features(self, df, drop_features, time_series_features):
        df.drop(drop_features, axis=1, inplace=True)
        df.drop(time_series_features, axis=1, inplace=True)
        return df

    def select_features (self, df, features, new_features):
        return df[features + new_features + ['weekofyear']]
