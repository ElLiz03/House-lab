import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import plotly.express as px

# TensorFlow импорты 
try:
    import tensorflow as tf
    from tensorflow import keras
    tf_version = tf.__version__
    print(f"✅ TensorFlow версия: {tf_version}")
    TF_AVAILABLE = True
except Exception as e:
    print(f"❌ TensorFlow не доступен: {e}")
    print("Пропускаем нейронные сети")
    TF_AVAILABLE = False
    tf = None
    keras = None

# Scikit-learn импорты
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, accuracy_score, classification_report
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# FactorAnalyzer импорты
try:
    from factor_analyzer import FactorAnalyzer
    FACTOR_ANALYZER_AVAILABLE = True
except ImportError:
    print("⚠️ FactorAnalyzer не доступен, пропускаем факторный анализ")
    FACTOR_ANALYZER_AVAILABLE = False

import warnings
warnings.filterwarnings('ignore')

# Загружаем данные
df = pd.read_csv("selling_apartments.csv")
pd.set_option('display.max_columns', None)
print(f"Размер данных: {df.shape}")

# Обработка данных
del df['published']
del df['updated']
del df['agencyName']
del df['enterances']
del df['build_year']
del df['build_serias']
del df['max_levels']
del df['min_levels']
del df['build_type']
del df['areaRating']

df = df[df['city'] == 'Москва']
df[['current_level', 'all_level']] = df['level'].str.split('/', expand=True)
df['current_level'] = pd.to_numeric(df['current_level'], errors='coerce')
df['all_level'] = pd.to_numeric(df['all_level'], errors='coerce')

del df['level']
del df['city']

print(df.head())

# Обработка пропущенных значений
missing_values = df.isnull().sum()
print("Пропущенные значения:")
print(missing_values)

df['kitchen_area'] = df.groupby('rooms')['kitchen_area'].transform(lambda x: x.fillna(x.median()))
df['living_area'] = df.groupby('rooms')['living_area'].transform(lambda x: x.fillna(x.median()))

df['material'].fillna(df['material'].mode()[0], inplace=True)
df['all_level'].fillna(df['all_level'].mode()[0], inplace=True)
df['all_level'] = df['all_level'].astype('int64')

mode_value = df.loc[df['object_type'] != '0', 'object_type'].mode()[0]
df['object_type'] = df['object_type'].replace('0', mode_value)

df = df.drop(df[df['build_overlap'] == '0'].index)
df = df.drop(df[df['rubbish_chute'] == '0'].index)

values_to_replace = ['0', 'Не заполнено', 'Специализированный жилищный фонд','Жилой дом блокированной застройки']
df['type'] = df['type'].replace(values_to_replace, 'Иное')

df = df[(df['area'] > 0) | (df['area'].isna())]
df.loc[(df['rooms'] == 0) & (df['area'] > 0), 'rooms'] = 1

df["gas"] = df["gas"].replace("0", "Нет")

heating_replacement = {
    'Индивидуальный тепловой пункт (ИТП)': 'ИТП',
    'Автономная котельная (крышная, встроенно-пристроенная)': 'Автономная котельная',
    'Квартирное отопление (квартирный котел)': 'Квартирное',
    'Печное': 'Без отопления',
    '0': 'Центральное',
    'Нет': 'Без отопления'
}
df['heating'] = df['heating'].replace(heating_replacement)

replacement_dict = {
    'Смешанные': 'Другие',
    'Деревянные': 'Другие',
    'Железобетонная панель': 'Железобетон',
    'Каменные, кирпичные': 'Кирпич',
    '0': 'Железобетон',
    'Не заполнено': 'Железобетон'
}
df['build_walls'] = df['build_walls'].replace(replacement_dict)

df['material'] = df['material'].replace('Дерево', 'Другие')
df['material'] = df['material'].replace('Блоки', 'Другие')
df['heating'] = df['heating'].replace('Квартирное', 'Центральное')

numeric_cols = ['kitchen_area', 'living_area', 'area', 'price_by_meter', 'price', 'remoute_from_center']

print("Описательная статистика:")
print(df[numeric_cols].describe())

# Визуализация
plt.figure(figsize=(15, 8))
for i, col in enumerate(numeric_cols, 1):
    plt.subplot(2, 3, i)
    sns.boxplot(y=df[col])
    plt.title(col)
plt.tight_layout()
plt.show()

# Обработка выбросов
def treat_outliers(df, column, method='remove', factor=1.5):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - factor * IQR
    upper_bound = Q3 + factor * IQR

    if method == 'remove':
        return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    elif method == 'cap':
        df[column] = np.where(df[column] < lower_bound, lower_bound,
                             np.where(df[column] > upper_bound, upper_bound, df[column]))
        return df
    elif method == 'transform':
        df[column] = np.log1p(df[column])
        return df
    return df

df = treat_outliers(df, 'kitchen_area', method='cap')
df = treat_outliers(df, 'living_area', method='cap')
df = treat_outliers(df, 'area', method='cap')
df = treat_outliers(df, 'price_by_meter', method='remove')
df = treat_outliers(df, 'price', method='remove')
df = treat_outliers(df, 'remoute_from_center', method='transform')

# Удаление дубликатов
df.drop_duplicates(inplace=True)
print(f"После очистки: {df.shape}")

# Кодирование категориальных переменных
cat_cols = df.select_dtypes(include=['object']).columns
label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Корреляционная матрица
plt.figure(figsize=(12, 8))
numeric_df = df.select_dtypes(include=["int64", "float64"])
corr = numeric_df.corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", center=0)
plt.title("Корреляция числовых признаков")
plt.tight_layout()
plt.show()

# Подготовка данных для моделей
X = df.drop(['price', 'price_by_meter'], axis=1, errors='ignore')
y = df['price']

# Если есть категориальные признаки после LabelEncoder
cat_cols_encoded = [col for col in X.columns if col in label_encoders]
for col in cat_cols_encoded:
    X[col] = X[col].astype('int')

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Случайный лес - регрессия
print("\nСлучайный лес - регрессия")

rf_regressor = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

# Кросс-валидация
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(rf_regressor, X_train_scaled, y_train, 
                           cv=kf, scoring='r2', n_jobs=-1)

print(f"Кросс-валидация R²: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

# Обучение и оценка
rf_regressor.fit(X_train_scaled, y_train)
y_pred_rf = rf_regressor.predict(X_test_scaled)

mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)
mae_rf = mean_absolute_error(y_test, y_pred_rf)

print(f"\nТестовая выборка:")
print(f"MSE: {mse_rf:.2f}")
print(f"R²: {r2_rf:.4f}")
print(f"MAE: {mae_rf:.2f}")

# Визуализация результатов
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_rf, alpha=0.5, s=20)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
         'r--', lw=2, label='Идеальная линия')
plt.xlabel('Фактическая цена')
plt.ylabel('Предсказанная цена')
plt.title('Случайный лес: Фактические vs Предсказанные значения')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Важность признаков
feature_importance = pd.DataFrame({
    'Признак': X.columns,
    'Важность': rf_regressor.feature_importances_
}).sort_values('Важность', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Важность', y='Признак', data=feature_importance.head(15))
plt.title('Топ-15 важнейших признаков (Случайный лес)')
plt.tight_layout()
plt.show()

# Случайный лес - классификация
print("\nСлучайный лес - классификация")

# Создание категорий цен
price_categories = pd.qcut(df['price'], q=3, labels=['Дешевый', 'Средний', 'Дорогой'])
X_clf = df.drop(['price', 'price_by_meter'], axis=1, errors='ignore')
y_clf = price_categories

X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
    X_clf, y_clf, test_size=0.2, random_state=42, stratify=y_clf
)

scaler_clf = StandardScaler()
X_train_clf_scaled = scaler_clf.fit_transform(X_train_clf)
X_test_clf_scaled = scaler_clf.transform(X_test_clf)

rf_classifier = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

# Кросс-валидация для классификации
cv_scores_clf = cross_val_score(rf_classifier, X_train_clf_scaled, y_train_clf,
                               cv=kf, scoring='accuracy', n_jobs=-1)

print(f"Кросс-валидация Accuracy: {cv_scores_clf.mean():.4f} (±{cv_scores_clf.std():.4f})")

# Обучение и оценка
rf_classifier.fit(X_train_clf_scaled, y_train_clf)
y_pred_clf = rf_classifier.predict(X_test_clf_scaled)

accuracy_clf = accuracy_score(y_test_clf, y_pred_clf)

print(f"\nТестовая выборка:")
print(f"Accuracy: {accuracy_clf:.4f}")
print("\nОтчет по классификации:")
print(classification_report(y_test_clf, y_pred_clf))

# Матрица ошибок
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test_clf, y_pred_clf)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Дешевый', 'Средний', 'Дорогой'],
            yticklabels=['Дешевый', 'Средний', 'Дорогой'])
plt.title('Матрица ошибок (Случайный лес)')
plt.ylabel('Фактический класс')
plt.xlabel('Предсказанный класс')
plt.tight_layout()
plt.show()

# Градиентный бустинг
print("\nGradient Boosting - регрессия")

gb_regressor = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)

# Кросс-валидация для Gradient Boosting
cv_scores_gb = cross_val_score(gb_regressor, X_train_scaled, y_train,
                              cv=kf, scoring='r2', n_jobs=-1)

print(f"Кросс-валидация R²: {cv_scores_gb.mean():.4f} (±{cv_scores_gb.std():.4f})")

# Обучение и оценка
gb_regressor.fit(X_train_scaled, y_train)
y_pred_gb = gb_regressor.predict(X_test_scaled)

mse_gb = mean_squared_error(y_test, y_pred_gb)
r2_gb = r2_score(y_test, y_pred_gb)
mae_gb = mean_absolute_error(y_test, y_pred_gb)

print(f"\nТестовая выборка:")
print(f"MSE: {mse_gb:.2f}")
print(f"R²: {r2_gb:.4f}")
print(f"MAE: {mae_gb:.2f}")

# Нейронная сеть
if TF_AVAILABLE and keras is not None:
    print("\nНейронная сеть (TensorFlow/Keras)")
    
    try:
        # Создаем нейронную сеть
        model = keras.Sequential([
            keras.layers.Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(1)
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        print("Архитектура модели:")
        model.summary()
        
        # Обучение
        history = model.fit(
            X_train_scaled, y_train,
            validation_split=0.2,
            epochs=50,
            batch_size=32,
            verbose=0
        )
        
        # Предсказание
        y_pred_nn = model.predict(X_test_scaled).flatten()
        
        mse_nn = mean_squared_error(y_test, y_pred_nn)
        r2_nn = r2_score(y_test, y_pred_nn)
        mae_nn = mean_absolute_error(y_test, y_pred_nn)
        
        print(f"\nТестовая выборка:")
        print(f"MSE: {mse_nn:.2f}")
        print(f"R²: {r2_nn:.4f}")
        print(f"MAE: {mae_nn:.2f}")
        
        # Визуализация обучения
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Ошибка обучения')
        plt.plot(history.history['val_loss'], label='Ошибка валидации')
        plt.xlabel('Эпоха')
        plt.ylabel('MSE')
        plt.legend()
        plt.title('Обучение нейронной сети')
        
        plt.subplot(1, 2, 2)
        plt.scatter(y_test, y_pred_nn, alpha=0.5, s=20)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
                 'r--', lw=2, label='Идеальная линия')
        plt.xlabel('Фактическая цена')
        plt.ylabel('Предсказанная цена')
        plt.title('Нейронная сеть: Предсказания')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Ошибка при работе с нейронной сетью: {e}")
else:
    print("\n⚠️ TensorFlow/Keras не доступен, пропускаем нейронные сети")

# Сравнение моделей
print("\nСравнение моделей регрессии")

models_comparison = pd.DataFrame({
    'Модель': ['Случайный лес', 'Gradient Boosting'],
    'R²': [r2_rf, r2_gb],
    'MSE': [mse_rf, mse_gb],
    'MAE': [mae_rf, mae_gb]
})

if TF_AVAILABLE and keras is not None and 'r2_nn' in locals():
    models_comparison = pd.concat([
        models_comparison,
        pd.DataFrame({
            'Модель': ['Нейронная сеть'],
            'R²': [r2_nn],
            'MSE': [mse_nn],
            'MAE': [mae_nn]
        })
    ], ignore_index=True)

print(models_comparison.sort_values('R²', ascending=False))

# Визуализация сравнения моделей
plt.figure(figsize=(10, 6))
x = np.arange(len(models_comparison))
width = 0.25

plt.bar(x - width, models_comparison['R²'], width, label='R²', color='skyblue')
plt.bar(x, models_comparison['MSE'] / 1e10, width, label='MSE (×10¹⁰)', color='lightcoral')
plt.bar(x + width, models_comparison['MAE'] / 1e6, width, label='MAE (×10⁶)', color='lightgreen')

plt.xlabel('Модель')
plt.ylabel('Метрики (нормализованные)')
plt.title('Сравнение моделей регрессии')
plt.xticks(x, models_comparison['Модель'], rotation=45)
plt.legend()
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.show()

# Факторный анализ
if FACTOR_ANALYZER_AVAILABLE:
    print("\nФакторный анализ")
    
    try:
        numeric_cols_fa = ['rooms', 'kitchen_area', 'living_area', 'area',
                          'price_by_meter', 'price', 'remoute_from_center',
                          'current_level', 'all_level']
        
        fa_data = df[numeric_cols_fa].dropna()
        
        if len(fa_data) > 0:
            scaler_fa = StandardScaler()
            scaled_fa_data = scaler_fa.fit_transform(fa_data)
            
            fa = FactorAnalyzer(rotation=None, impute="drop")
            fa.fit(scaled_fa_data)
            
            ev, v = fa.get_eigenvalues()
            
            plt.figure(figsize=(10, 6))
            plt.scatter(range(1, len(ev) + 1), ev)
            plt.plot(range(1, len(ev) + 1), ev)
            plt.title('График каменистой осыпи (Scree Plot)')
            plt.xlabel('Факторы')
            plt.ylabel('Собственные значения')
            plt.axhline(y=1, color='r', linestyle='--')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
            
            n_factors = sum(ev > 1)
            print(f"Рекомендуемое число факторов: {n_factors}")
            
            if n_factors > 0:
                fa = FactorAnalyzer(n_factors=n_factors, rotation="varimax")
                fa.fit(scaled_fa_data)
                
                loadings = pd.DataFrame(
                    fa.loadings_,
                    index=numeric_cols_fa,
                    columns=[f"Factor_{i+1}" for i in range(n_factors)]
                )
                
                plt.figure(figsize=(12, 6))
                sns.heatmap(loadings, annot=True, cmap='coolwarm', center=0, 
                           fmt=".2f", linewidths=0.5)
                plt.title('Матрица факторных нагрузок')
                plt.tight_layout()
                plt.show()
                
                for i in range(n_factors):
                    print(f"\nФактор {i+1}:")
                    top_features = loadings[f'Factor_{i+1}'].abs().sort_values(ascending=False).head(3)
                    for feature, loading in top_features.items():
                        print(f"  {feature}: {loading:.3f}")
        
    except Exception as e:
        print(f"Ошибка при факторном анализе: {e}")

# Кластеризация
print("\nКластеризация (K-Means)")

cluster_features = ['price', 'area', 'price_by_meter', 'kitchen_area', 'remoute_from_center']
cluster_data = df[cluster_features].dropna()

if len(cluster_data) > 0:
    scaler_cluster = StandardScaler()
    scaled_cluster = scaler_cluster.fit_transform(cluster_data)
    
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)
        kmeans.fit(scaled_cluster)
        wcss.append(kmeans.inertia_)
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 11), wcss, marker='o')
    plt.title('Метод локтя для определения оптимального числа кластеров')
    plt.xlabel('Количество кластеров')
    plt.ylabel('WCSS')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    optimal_clusters = 3
    kmeans = KMeans(n_clusters=optimal_clusters, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(scaled_cluster)
    
    cluster_stats = df.groupby('cluster')[cluster_features].mean()
    print("\nСредние значения по кластерам:")
    print(cluster_stats)
    
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(df['longitude'], df['latitude'], 
                         c=df['cluster'], cmap='viridis', alpha=0.6, s=20)
    plt.colorbar(scatter, label='Кластер')
    plt.title('Географическое распределение кластеров')
    plt.xlabel('Долгота')
    plt.ylabel('Широта')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# Предсказание
print("\nИнтерактивное предсказание цены")

def predict_price(features, model_type='rf'):
    """Функция для предсказания цены"""
    input_df = pd.DataFrame([features])
    
    # Добавляем недостающие признаки
    for col in X.columns:
        if col not in input_df.columns:
            input_df[col] = df[col].median()
    
    input_df = input_df[X.columns]
    
    # Кодируем категориальные признаки
    for col in cat_cols_encoded:
        if col in input_df.columns and col in label_encoders:
            try:
                input_df[col] = label_encoders[col].transform([input_df[col].iloc[0]])[0]
            except:
                input_df[col] = 0
    
    input_scaled = scaler.transform(input_df)
    
    if model_type == 'rf':
        prediction = rf_regressor.predict(input_scaled)[0]
        model_name = "Случайный лес"
    elif model_type == 'gb':
        prediction = gb_regressor.predict(input_scaled)[0]
        model_name = "Gradient Boosting"
    else:
        prediction = rf_regressor.predict(input_scaled)[0]
        model_name = "Случайный лес"
    
    return prediction, model_name

# Пример использования
sample_features = {
    'area': 65.0,
    'kitchen_area': 12.0,
    'living_area': 35.0,
    'rooms': 2,
    'current_level': 5,
    'all_level': 9,
    'remoute_from_center': 2.5,
    'build_oldest': 15,
    'gas': 1,
    'heating': 1,
    'material': 2,
    'object_type': 1,
    'type': 1,
    'build_walls': 2,
    'build_overlap': 1,
    'rubbish_chute': 1,
    'latitude': 55.7558,
    'longitude': 37.6173
}

print("Пример предсказания цены:")
print("-" * 30)

for key, value in list(sample_features.items())[:10]:  # Покажем первые 10 признаков
    print(f"{key}: {value}")

print("...")

# Предсказание разными моделями
for model_type in ['rf', 'gb']:
    predicted_price, model_name = predict_price(sample_features, model_type)
    print(f"\n{model_name}: {predicted_price:,.2f} руб")

print("\n✅ Анализ завершен!")