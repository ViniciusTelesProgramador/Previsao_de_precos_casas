import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from io import StringIO  # Importação correta

@st.cache_data
def load_data():
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
        buffer = StringIO()  # Usar StringIO para capturar o output de info()
        data.info(buf=buffer)
        s = buffer.getvalue()
        st.text(s)  # Exibir a informação do dataset

        # Gráfico de Correlação
        st.subheader("Gráfico de Correlação")

        # Filtrar apenas colunas numéricas
        numeric_data = data.select_dtypes(include=['float64', 'int64'])
        
        if not numeric_data.empty:
            plt.figure(figsize=(12, 8))
            sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm', fmt='.2f')
            st.pyplot(plt)
        else:
            st.warning("Não há colunas numéricas suficientes para gerar o gráfico de correlação.")

        # Selecione as colunas para o modelo
        features = st.multiselect("Selecione as características para prever o preço:", options=data.columns.tolist())
        target = st.selectbox("Selecione a variável alvo (preço):", options=['price'])

        if st.button("Treinar Modelo"):
            if len(features) > 0:
                # Dividir os dados
                X = data[features]
                y = data[target]

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                # Treinar o modelo
                model = LinearRegression()
                model.fit(X_train, y_train)

                # Salvar o modelo usando joblib
                joblib.dump(model, 'modelo_previsao_precos.pkl')

                # Exibir os coeficientes do modelo
                st.write("Coeficientes do Modelo:")
                for feature, coef in zip(features, model.coef_):
                    st.write(f"{feature}: {coef}")

                # Fazer previsões
                predictions = model.predict(X_test)

                # Gráfico de previsões vs valores reais
                plt.figure(figsize=(10, 6))
                plt.scatter(y_test, predictions)
                plt.xlabel("Valores Reais")
                plt.ylabel("Previsões")
                plt.title("Previsões vs Valores Reais")
                plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--')  # linha de referência
                st.pyplot(plt)

                # Previsão manual
                st.subheader("Prever o Preço de uma Nova Casa")
                input_data = {}
                for feature in features:
                    input_data[feature] = st.number_input(f"Informe o valor de {feature}:", min_value=0.0)

                if st.button("Prever Preço da Casa"):
                    # Criar um DataFrame com as características
                    input_df = pd.DataFrame(input_data, index=[0])

                    # Fazer a previsão
                    predicted_price = model.predict(input_df)

                    # Exibir o preço previsto
                    st.write(f"O preço previsto da casa é: ${predicted_price[0]:,.2f}")

if __name__ == "__main__":
    main()
