import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Função para carregar o dataset
@st.cache_data
def load_data():
    train_url = "https://github.com/ViniciusTelesProgramador/Previsao_de_precos_casas/blob/main/kc_house_data.csv"  
    # Tenta carregar o CSV especificando o delimitador e ignorando linhas com erros
    try:
        data = pd.read_csv(train_url, delimiter=',', error_bad_lines=False, encoding='utf-8')
        return data
    except Exception as e:
        st.error(f"Erro ao carregar o dataset: {e}")
        return None

# Carregar o dataset
data = load_data()

# Se o dataset for carregado corretamente, exibir os dados
if data is not None:
    st.write(data.head())

    # Preparar os dados para o modelo
    X = data[['GrLivArea', 'OverallQual', 'GarageCars']]  # Variáveis independentes
    y = data['SalePrice']  # Variável dependente

    # Dividir os dados em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Treinar o modelo de regressão linear
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Fazer previsões no conjunto de teste
    y_pred = model.predict(X_test)

    # Calcular e exibir o erro quadrático médio
    mse = mean_squared_error(y_test, y_pred)
    st.write(f"Erro Quadrático Médio: {mse}")
else:
    st.write("O dataset não foi carregado.")
