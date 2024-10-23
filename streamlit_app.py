import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Função para carregar os dados
@st.cache_data
def load_data():
    # Substituir pelo seu arquivo de dados
    data = pd.read_csv('house_prices.csv')
    return data

# Chamar a função para carregar os dados
data = load_data()

# Título e descrição da aplicação
st.title('Previsão de Preços de Casas')
st.write("Este protótipo prevê o preço de uma casa com base em suas características.")
st.write("Visualização do conjunto de dados:")
st.dataframe(data.head())

# Selecionar características e alvo (preço)
X = data[['area', 'bedrooms', 'bathrooms', 'location']]  # Características
y = data['price']  # Preço da casa (alvo)

# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Treinar o modelo
model = LinearRegression()
model.fit(X_train, y_train)

# Prever no conjunto de teste
y_pred = model.predict(X_test)

# Calcular o erro quadrático médio (RMSE)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
st.write(f"Erro quadrático médio (RMSE): {rmse:.2f}")

# Coletar dados do usuário para previsão
st.sidebar.header('Insira as características da casa')
area = st.sidebar.number_input('Área (em metros quadrados)', min_value=20, max_value=1000, value=100)
bedrooms = st.sidebar.number_input('Número de quartos', min_value=1, max_value=10, value=2)
bathrooms = st.sidebar.number_input('Número de banheiros', min_value=1, max_value=10, value=1)
location = st.sidebar.selectbox('Localização', ('Centro', 'Subúrbio', 'Periferia'))

# Fazer a previsão
input_data = pd.DataFrame([[area, bedrooms, bathrooms, location]], columns=['area', 'bedrooms', 'bathrooms', 'location'])
predicted_price = model.predict(input_data)

st.sidebar.write(f"Preço estimado: R$ {predicted_price[0]:,.2f}")

# Visualizações
st.write("Relação entre área e preço:")
plt.figure(figsize=(8, 6))
plt.scatter(data['area'], data['price'], alpha=0.5)
plt.xlabel('Área (m²)')
plt.ylabel('Preço (R$)')
st.pyplot(plt)
