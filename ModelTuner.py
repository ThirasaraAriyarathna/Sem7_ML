from AccuracyChecking import AccuracyChecker

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor


class ModelTuner:

    def __init__(self):
        self.accuracy_checker = AccuracyChecker()
        pass

    def random_forest_tuner(self, x_sj, y_sj, x_iq, y_iq):
        reg = RandomForestRegressor(random_state=67)

        param_grid = [
            {
              'n_estimators': [10, 30, 100, 300, 400, 500, 700],
              'max_depth': [3, 5, 7, None]
            }
        ]
        print("Random Forest")
        self.accuracy_checker.grid_search_cross_val(reg, x_sj, y_sj.total_cases, param_grid)
        self.accuracy_checker.grid_search_cross_val(reg, x_iq, y_iq.total_cases, param_grid)

    def gradient_boost_tuner(self, x_sj, y_sj, x_iq, y_iq):
        reg = GradientBoostingRegressor(random_state=67)
        param_grid = [
            {'learning_rate': [0.1, 0.3, 1.0, 3.0], 'n_estimators': [10, 30, 100, 300, 500, 600],
             'max_depth': [3, 5, 7, 9]}
        ]
        print("Gradiant boosting")
        print("Random Forest")
        self.accuracy_checker.grid_search_cross_val(reg, x_sj, y_sj.total_cases, param_grid)
        self.accuracy_checker.grid_search_cross_val(reg, x_iq, y_iq.total_cases, param_grid)
