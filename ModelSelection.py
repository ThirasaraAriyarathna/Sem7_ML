from AccuracyChecking import AccuracyChecker

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor



class ModelSelector:

    def __init__(self):
        self.accuracy_checker = AccuracyChecker()
        pass

    def linear_regression_model(self, x_sj, y_sj, x_iq, y_iq):
        reg = LinearRegression(n_jobs=-1)
        print('San Juan:')
        self.accuracy_checker.train_cross_val_score(reg, x_sj, y_sj)
        print('\nIquitos:')
        self.accuracy_checker.train_cross_val_score(reg, x_iq, y_iq)

    def kn_neighbors_model(self, x_sj, y_sj, x_iq, y_iq):
        reg = KNeighborsRegressor(n_jobs=-1)
        print('San Juan:')
        self.accuracy_checker.train_cross_val_score(reg, x_sj, y_sj)
        print('\nIquitos:')
        self.accuracy_checker.train_cross_val_score(reg, x_iq, y_iq)

    def svr_model(self, x_sj, y_sj, x_iq, y_iq, kernel):
        reg = SVR(kernel=kernel)
        print('San Juan:')
        self.accuracy_checker.train_cross_val_score(reg, x_sj, y_sj)
        print('\nIquitos:')
        self.accuracy_checker.train_cross_val_score(reg, x_iq, y_iq)

    def gradient_boosting_model(self, x_sj, y_sj, x_iq, y_iq):
        reg = GradientBoostingRegressor(criterion='mae', random_state=67)
        print('San Juan:')
        self.accuracy_checker.train_cross_val_score(reg, x_sj, y_sj)
        print('\nIquitos:')
        self.accuracy_checker.train_cross_val_score(reg, x_iq, y_iq)

    def radom_forest_model(self, x_sj, y_sj, x_iq, y_iq):
        reg = RandomForestRegressor(criterion='mae', n_jobs=-1, random_state=67)
        print('San Juan:')
        self.accuracy_checker.train_cross_val_score(reg, x_sj, y_sj)
        print('\nIquitos:')
        self.accuracy_checker.train_cross_val_score(reg, x_iq, y_iq)

    def mlp_model(self, x_sj, y_sj, x_iq, y_iq):
        reg = MLPRegressor(max_iter=3000, random_state=67)
        print('San Juan:')
        self.accuracy_checker.train_cross_val_score(reg, x_sj, y_sj)
        print('\nIquitos:')
        self.accuracy_checker.train_cross_val_score(reg, x_iq, y_iq)

