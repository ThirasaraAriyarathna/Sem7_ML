import numpy as np

from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV




class AccuracyChecker:

    def __init__(self):
        pass

    def train_predict_score(self, reg, x, y):
        reg.fit(x, y)
        y_pred = reg.predict(x)
        return mean_absolute_error(y_true=y, y_pred=y_pred)

    def train_cross_val_score(self, reg, x, y, scoring='neg_mean_absolute_error'):
        reg.fit(x, y)
        scores = np.abs(cross_val_score(reg, x, y, scoring=scoring))
        print("Scores: {}".format(scores))
        print("Avg Score: {}".format(scores.mean()))

    def grid_search_cross_val(self, reg, X, y, param_grid, scoring='neg_mean_absolute_error'):
        grid = GridSearchCV(reg, param_grid=param_grid, scoring=scoring)
        grid.fit(X, y)
        print("Best score: {}".format(np.abs(grid.best_score_)))
        print("Best params: {}".format(grid.best_params_))

    def cross_validate_out_of_sample(self, y_pred, y_cross):
        print(mean_absolute_error(y_true=y_cross, y_pred=y_pred))
