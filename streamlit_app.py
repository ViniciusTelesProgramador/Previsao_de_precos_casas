# Importando as bibliotecas necessárias
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder

# Função para carregar os dados
@st.cache
def load_data():
    train_url = "https://github.com/Shreyas3108/house-price-prediction/blob/master/kc_house_data.csv" 
    data = pd.read_csv(train_url)
    return data

# Função para treinar o modelo de Regressão
def train_model(data):
    # Pré-processamento simples
    data = data.dropna()  # Removendo dados ausentes para simplificação
    X = data[['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath']]  # Features
    y = data['SalePrice']  # Target

    # Dividindo o conjunto de dados em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Treinando o modelo de Random Forest
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Avaliando o modelo
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    return model, rmse

# Função para realizar predições com entrada do usuário
def predict_price(model, user_input):
    prediction = model.predict([user_input])
    return prediction[0]

# Carregar os dados
st.title("Previsão de Preços de Casas")
st.write("Esta aplicação prevê o preço de uma casa com base em suas características.")

data = load_data()

# Mostrar dados
if st.checkbox("Mostrar dados brutos"):
    st.write(data.head())

# Treinar o modelo
model, rmse = train_model(data)

st.write(f"Modelo treinado com sucesso! RMSE do modelo: {rmse:.2f}")

# Coletar dados do usuário
st.sidebar.header("Insira as características da casa:")
overall_qual = st.sidebar.slider('Qualidade geral (1-10)', 1, 10, 5)
gr_liv_area = st.sidebar.number_input('Área (pés quadrados)', value=1500)
garage_cars = st.sidebar.slider('Número de carros na garagem', 0, 5, 2)
total_bsmt_sf = st.sidebar.number_input('Área do porão (pés quadrados)', value=800)
full_bath = st.sidebar.slider('Número de banheiros completos', 0, 3, 2)

# Gerar previsão
user_input = [overall_qual, gr_liv_area, garage_cars, total_bsmt_sf, full_bath]
predicted_price = predict_price(model, user_input)

# Mostrar o preço previsto
st.subheader(f"Preço previsto para a casa: ${predicted_price:,.2f}")
