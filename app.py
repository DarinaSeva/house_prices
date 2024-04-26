import streamlit as st
import pandas as pd
# import joblib  # Используется для загрузки модели, если модель сохранена с помощью joblib
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Надстройки!
import sklearn
sklearn.set_config(transform_output="pandas")

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score

# Надстройки Дарины

from sklearn.linear_model import Lasso
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler, MinMaxScaler, OrdinalEncoder
from sklearn.model_selection import GridSearchCV, KFold
from category_encoders import TargetEncoder
import catboost 
from custom_transformers import LotFrontageImputer, ColumnDroper


# Предполагаем, что модель и функция предсказания уже определены как выше

def welcome_page():
    st.title('Проект команды Catboost!')
    st.image('DALL·E_2024_04_26_13_33_12_A_playful_and_imaginative_artwork_featuring.webp', use_column_width=True)  # Путь к изображению может быть изменён
    st.header('Проект House Prices Prediction in Ames')
    st.write('''
        Наше приложение поможет вам узнать предполагаемую цену на дома. Просто перейдите на страницу предсказаний,
        чтобы использовать нашу модель машинного обучения.
        ''')
    st.header('Наши results')
    st.image('photo_2024-04-26 16.48.08.jpeg', use_column_width=True)

import pickle


def predict_price():
    model = pickle.load('ml_pipeline.pkl')
    uploaded_file = st.file_uploader("Select CSV-file", type=["csv"])

    if uploaded_file is not None:
        @st.cache_data 
        def load_data(file):
            return pd.read_csv(file)

        df = load_data(uploaded_file)

        df['HasOpenPorch'] = (df['OpenPorchSF'] == 0) * 1
        df['HasEnclosedPorch'] = (df['EnclosedPorch'] == 0) * 1
        df['Has3SsnPorch'] = (df['3SsnPorch'] == 0) * 1
        df['HasScreenPorch'] = (df['ScreenPorch'] == 0) * 1

        # И вынести общую площадь крыльца
        df['Total_porch_sf'] = (df['OpenPorchSF'] + df['3SsnPorch'] +
                                    df['EnclosedPorch'] + df['ScreenPorch'] +
                                    df['WoodDeckSF'])

        df['haspool'] = df['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
        df['YrBltAndRemod'] = df['YearBuilt'] + df['YearRemodAdd']
        df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
        # Добавляем общую площадь ванных комнат
        df['TotalBath'] = df['FullBath'] + df['HalfBath']
        to_drop = ['OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'WoodDeckSF', 'PoolArea', 'TotalBsmtSF', '1stFlrSF',
                '2ndFlrSF', 'FullBath', 'HalfBath', 'Utilities', 'PoolQC', 'Id']
        df.drop(to_drop, axis=1, inplace=True)

        if df.empty:
            st.write("Data file is empty or invalid.")
        else:
            # Ensure that the data in df is prepared for prediction (e.g., in the correct format)
            prediction = model.predict(df)
            # Exponentiate the prediction if your model outputs log-transformed predictions
            exp_prediction = np.exp(prediction[0])
            st.write('Prediction:', round(exp_prediction, 2))
    else:
        st.write("No file uploaded.")
    

def exploration_page():
    st.title('Исследовательский анализ данных')
    st.write('Здесь представлены графики для анализа данных о домах')

    # Загрузка данных
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
    data = pd.concat([train, test], ignore_index=True)

# Генерация графиков

# Гистограмма для одной из интересующих переменных, например, SalePrice
    st.subheader('Распределение цен на дома (SalePrice)')
    fig, ax = plt.subplots()
    sns.histplot(data['SalePrice'], kde=True, ax=ax)
    ax.set_title('Распределение цен на дома')
    st.pyplot(fig)

    # Диаграмма рассеяния для анализа взаимосвязи между GrLivArea и SalePrice
    st.subheader('Зависимость цены от жилой площади (GrLivArea)')
    fig, ax = plt.subplots()
    sns.scatterplot(x='GrLivArea', y='SalePrice', data=data, ax=ax)
    ax.set_title('Зависимость цены от жилой площади')
    st.pyplot(fig)


    # График 1
    st.subheader('График 1: Зависимость между площадью жилой площади и ценой продажи')
    fig, ax = plt.subplots()
    ax.scatter(x=train['GrLivArea'], y=train['SalePrice'])
    plt.ylabel('SalePrice', fontsize=13)
    plt.xlabel('GrLivArea', fontsize=13)
    st.pyplot(fig)

    # График 2
    st.subheader('График 2: Зависимость между площадью открытой веранды и ценой продажи')
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.scatterplot(x='OpenPorchSF', y='SalePrice', data=train, ax=ax)
    plt.xticks(rotation=90)
    plt.xlabel('Площадь открытой веранды')
    plt.ylabel('Цена продажи')
    plt.title('Зависимость между площадью открытой веранды и ценой продажи')
    st.pyplot(fig)

    # График 3
    st.subheader('График 3: Зависимость между количеством комнат и ценой продажи')
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.scatter(train['TotRmsAbvGrd'], train['SalePrice'], alpha=0.5)
    plt.xlabel('Количество комнат')
    plt.ylabel('Цена продажи')
    plt.title('Зависимость между количеством комнат и ценой продажи')
    st.pyplot(fig)

    # График 4
    st.subheader('График 4: Зависимость между площадью участка и ценой продажи')
    filtered_train = train[train['LotArea'] <= 50000]
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.scatterplot(x='LotArea', y='SalePrice', data=filtered_train, ax=ax)
    plt.xticks(rotation=90)
    plt.xlabel('Площадь участка')
    plt.ylabel('Цена продажи')
    plt.title('Зависимость между площадью участка и ценой продажи')
    st.pyplot(fig)

    # График 5
    st.subheader('График 5: Зависимость между общей жилой площадью и ценой продажи')
    filtered_train_1 = train[train['GrLivArea'] <= 4000]
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.scatterplot(x='GrLivArea', y='SalePrice', data=filtered_train_1, ax=ax)
    plt.xticks(rotation=90)
    plt.xlabel('Общая жилая площадь')
    plt.ylabel('Цена продажи')
    plt.title('Зависимость между общей жилой площадью и ценой продажи')
    st.pyplot(fig)

    # График 6
    st.subheader('График 6: Влияние типа продажи на цену продажи')
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='SaleType', y='SalePrice', data=train, ax=ax)
    plt.xlabel('Тип продажи')
    plt.ylabel('Цена продажи')
    plt.title('Влияние типа продажи на цену продажи')
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # График 7
    st.subheader('График 7: Матрица корреляции')
    cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF']
    train_matrix = train[cols].corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(train_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
    plt.title('Матрица корреляции')
    st.pyplot(fig)


def main():
    page = st.sidebar.selectbox("Выберите страницу", ["Приветственная страница", "Предсказание цены", "Исследовательский анализ данных"])

    if page == "Приветственная страница":
        welcome_page()
    elif page == "Исследовательский анализ данных":
        exploration_page()
    elif page == "Предсказание цены":
        predict_price()
    

if __name__ == '__main__':
    main()
