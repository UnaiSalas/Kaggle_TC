import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import numpy as np
import matplotlib
from pylab import *
import seaborn as sns
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from bootcampviztools import *

# Cargar datos
df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

# Información del DataFrame
df_train.info()

# Extraer marcas de GPU y CPU
df_train['GPU_Brand'] = df_train['Gpu'].str.split().str[0]
df_train['CPU_Brand'] = df_train['Cpu'].str.split().str[0]
df_test['GPU_Brand'] = df_test['Gpu'].str.split().str[0]
df_test['CPU_Brand'] = df_test['Cpu'].str.split().str[0]

# Obtener variables dummy
df_train_mod = pd.get_dummies(df_train, columns=['GPU_Brand', 'CPU_Brand', 'OpSys', 'Cpu', 'Gpu', 'ScreenResolution', 'Memory', 'TypeName', 'Company'], dtype=int)
df_test_mod = pd.get_dummies(df_test, columns=['GPU_Brand', 'CPU_Brand', 'OpSys', 'Cpu', 'Gpu', 'ScreenResolution', 'Memory', 'TypeName', 'Company'], dtype=int)

X_train = df_train_mod.drop(columns=['Price_euros'])
y_train = df_train_mod['Price_euros']
X_test = df_test_mod.drop(columns=['Price_euros'])
y_test = df_test_mod['Price_euros']

# Convertir 'Weight' y 'Ram' a numérico
X_train['Weight'] = X_train['Weight'].str.replace('kg', '').astype(float)
X_train['Ram'] = X_train['Ram'].str.replace('GB', '').astype(float)
X_test['Weight'] = X_test['Weight'].str.replace('kg', '').astype(float)
X_test['Ram'] = X_test['Ram'].str.replace('GB', '').astype(float)

# Filtrar solo las columnas numéricas y calcular la matriz de correlación
X_train['Price_euros'] = y_train
numerical_df = X_train.select_dtypes(include=['int64', 'int32', 'float64'])
correlation_matrix = numerical_df.corr()
price_correlations = correlation_matrix['Price_euros'].sort_values(ascending=False)
significant_correlations = price_correlations[price_correlations > 0].index.tolist()
significant_correlations.remove('Price_euros')

# Preparar los conjuntos de datos finales para el entrenamiento y la prueba
X_train_mod_final = X_train[significant_correlations]
X_test_mod_final = X_test[significant_correlations]

# Entrenar el modelo
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_mod_final, y_train)

# Hacer predicciones
predicciones = model.predict(X_test_mod_final)

# Evaluar el modelo
print('MAE: ', metrics.mean_absolute_error(y_test, predicciones))
print('MSE: ', metrics.mean_squared_error(y_test, predicciones))
print('RMSE: ', np.sqrt(metrics.mean_squared_error(y_test, predicciones)))
print('R2: ', model.score(X_test_mod_final, y_test))

# Búsqueda de hiperparámetros
param_grid = {
    'n_estimators': [50, 100, 200, 300, 400, 500],
    'max_depth': [None, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    'min_samples_split': [2, 5, 10, 15, 20],
    'min_samples_leaf': [1, 2, 4, 6, 8]
}

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train_mod_final, y_train)

# Mejor modelo
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_
print("Mejores hiperparámetros: ", best_params)

# Predicciones con el mejor modelo
y_pred_train = best_model.predict(X_train_mod_final)
y_pred_test = best_model.predict(X_test_mod_final)

# Evaluación del mejor modelo
mse_train = mean_squared_error(y_train, y_pred_train)
mse_test = mean_squared_error(y_test, y_pred_test)

print(f"MSE en entrenamiento: {mse_train}")
print(f"MSE en prueba: {mse_test}")
print('MAE en entrenamiento: ', metrics.mean_absolute_error(y_train, y_pred_train))
print('MAE en prueba: ', metrics.mean_absolute_error(y_test, y_pred_test))