import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

@st.cache_data
def load_data():
    # URL bruta do seu repositório
    train_url = "https://raw.githubusercontent.com/ViniciusTelesProgramador/Previsao_de_precos_casas/main/kc_house_data.csv"
    try:
        data = pd.read_csv(train_url, delimiter=',', on_bad_lines='skip', encoding='utf-8')
        return data
    except Exception as e:
        st.error(f"Erro ao carregar o dataset: {e}")
        return None

def main():
    st.title("Previsão de Preços de Casas")

    # Carregar os dados
    data = load_data()
    if data is not None:
        st.write("Dados Carregados:")
        st.dataframe(data)

        # Exibir informações sobre os dados
        st.write("Informações sobre o dataset:")
        st.write(data.info())

        # Gráfico de correlação
        st.write("Gráfico de Correlação:")
        # Filtrar apenas colunas numéricas
        numeric_data = data.select_dtypes(include='number')
        
        # Verificar se há colunas numéricas
        if not numeric_data.empty:
            plt.figure(figsize=(10, 6))
            sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm', fmt='.2f')
            st.pyplot(plt)
        else:
            st.warning("Não há colunas numéricas disponíveis para calcular a correlação.")

        # Sidebar para configurações
        st.sidebar.title("Configurações do Modelo")
        features = st.sidebar.multiselect("Selecione as características para prever o preço:", options=data.columns.tolist())
        target = st.sidebar.selectbox("Selecione a variável alvo (preço):", options=['price'])

        if st.sidebar.button("Treinar Modelo"):
            if len(features) > 0:
                # Dividir os dados
                X = data[features]
                y = data[target]

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                # Treinar o modelo
                model = LinearRegression()
                model.fit(X_train, y_train)

                # Exibir os coeficientes do modelo
                st.write("Coeficientes do Modelo:")
                for feature, coef in zip(features, model.coef_):
                    st.write(f"{feature}: {coef:.2f}")

                # Fazer previsões
                predictions = model.predict(X_test)

                # Cálculo de métricas de avaliação
                mse = mean_squared_error(y_test, predictions)
                r2 = r2_score(y_test, predictions)
                st.write(f"MSE: {mse:.2f}")
                st.write(f"R²: {r2:.2f}")

                # Gráfico de previsões vs valores reais
                plt.figure(figsize=(10, 6))
                plt.scatter(y_test, predictions)
                plt.xlabel("Valores Reais")
                plt.ylabel("Previsões")
                plt.title("Previsões vs Valores Reais")
                plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--')  # linha de referência
                st.pyplot(plt)

                # Salvar o modelo
                joblib.dump(model, 'modelo_previsao.pickle')
                st.success("Modelo treinado e salvo como 'modelo_previsao.pickle'.")

if __name__ == "__main__":
    main()
