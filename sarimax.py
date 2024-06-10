import itertools
import warnings
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error

warnings.filterwarnings('ignore')

# Veri yükleme ve ön işleme
data = sm.datasets.co2.load_pandas()
y = data.data
y = y['co2'].resample('MS').mean()
y = y.fillna(y.bfill())
train = y[:'1997-12-01']
test = y['1998-01-01':]

# Model görselleştirme fonksiyonu
def plot_co2(train, test, y_pred, title):
    mae = mean_absolute_error(test, y_pred)
    train["1985":].plot(legend=True, label="TRAIN", title=f"{title}, MAE: {round(mae,2)}")
    test.plot(legend=True, label="TEST", figsize=(6, 4))
    y_pred.plot(legend=True, label="PREDICTION")
    plt.show()

# SARIMA modeli eğitme ve tahmin
model = SARIMAX(train, order=(1, 0, 1), seasonal_order=(0, 0, 0, 12))
sarima_model = model.fit(disp=0)
y_pred_test = sarima_model.get_forecast(steps=48)
y_pred = y_pred_test.predicted_mean
y_pred = pd.Series(y_pred, index=test.index)
plot_co2(train, test, y_pred, "SARIMA Modeli")

# PDQ ve sezonluk PDQ parametrelerinin kombinasyonları
p = d = q = range(0, 2)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]

# En iyi AIC değerini bulmak ve görselleştirmek
def sarima_optimizer_aic(train, pdq_range, seasonal_pdq_range):
    best_aic = float("inf")
    best_order = None
    best_seasonal_order = None

    # PDQ ve sezonluk PDQ kombinasyonlarını döngüye sokma
    for param in pdq_range:
        for param_seasonal in seasonal_pdq_range:
            try:
                sarimax_model = SARIMAX(train, order=param, seasonal_order=param_seasonal)
                results = sarimax_model.fit(disp=0)
                aic = results.aic
                if aic < best_aic:
                    best_aic = aic
                    best_order = param
                    best_seasonal_order = param_seasonal
                print(f'SARIMA{param}x{param_seasonal}12 - AIC: {aic}')
            except Exception as e:
                print(f"Hata oluştu: {e} - Parametre: SARIMA{param}x{param_seasonal}12")
                continue

    # Sonuçların döndürülmesi
    print(f'En İyi SARIMA Model Parametreleri: SARIMA{best_order}x{best_seasonal_order}12 - AIC: {best_aic}')
    return best_order, best_seasonal_order

best_order, best_seasonal_order = sarima_optimizer_aic(train, pdq, seasonal_pdq)

# En iyi modelin parametrelerini kullanarak görselleştirme
model_best = SARIMAX(train, order=best_order, seasonal_order=best_seasonal_order)
sarima_model_best = model_best.fit(disp=0)
y_pred_best_test = sarima_model_best.get_forecast(steps=48)
y_pred_best = y_pred_best_test.predicted_mean
y_pred_best = pd.Series(y_pred_best, index=test.index)
plot_co2(train, test, y_pred_best, f"En İyi SARIMA Modeli {best_order}x{best_seasonal_order}12")



p = d = q = range(0, 2)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]


def sarima_optimizer_mae(train, pdq, seasonal_pdq):
    best_mae, best_order, best_seasonal_order = float("inf"), None, None
    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                model = SARIMAX(train, order=param, seasonal_order=param_seasonal)
                sarima_model = model.fit(disp=0)
                y_pred_test = sarima_model.get_forecast(steps=48)
                y_pred = y_pred_test.predicted_mean
                mae = mean_absolute_error(test, y_pred)
                if mae < best_mae:
                    best_mae, best_order, best_seasonal_order = mae, param, param_seasonal
                print('SARIMA{}x{}12 - MAE:{}'.format(param, param_seasonal, mae))
            except:
                continue
    print('SARIMA{}x{}12 - MAE:{}'.format(best_order, best_seasonal_order, best_mae))
    return best_order, best_seasonal_order

best_order, best_seasonal_order = sarima_optimizer_mae(train, pdq, seasonal_pdq)

model = SARIMAX(train, order=best_order, seasonal_order=best_seasonal_order)
sarima_final_model = model.fit(disp=0)

y_pred_test = sarima_final_model.get_forecast(steps=48)
y_pred = y_pred_test.predicted_mean
y_pred = pd.Series(y_pred, index=test.index)

plot_co2(train, test, y_pred, "SARIMA")
