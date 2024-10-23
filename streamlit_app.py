import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from io import StringIO  

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

    st.write("""
    Este aplicativo permite prever o preço de casas com base em diversas características.
    Você pode selecionar quais características usar para treinar o modelo e visualizar os resultados das previsões.
    """)

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
        st.write("""
        O gráfico de correlação mostra como as diferentes características do dataset se relacionam entre si.
        Valores próximos a 1 ou -1 indicam uma forte correlação positiva ou negativa, respectivamente.
        """)

        # Filtrar apenas colunas numéricas
        numeric_data = data.select_dtypes(include=['float64', 'int64'])
        
        if not numeric_data.empty:
            plt.figure(figsize=(12, 8))
            sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm', fmt='.2f')
            st.pyplot(plt)
        else:
            st.warning("Não há colunas numéricas suficientes para gerar o gráfico de correlação.")

        # Selecione as colunas para o modelo
        st.subheader("Seleção de Características")
        st.write("""
        Selecione as características (colunas) que você deseja usar para prever o preço das casas.
        Você pode escolher várias características, e o modelo irá utilizá-las para treinar a previsão.
        """)
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
                st.write("""
                Os coeficientes indicam a influência de cada característica no preço previsto.
                Um coeficiente positivo significa que um aumento nessa característica tende a aumentar o preço,
                enquanto um coeficiente negativo indica que um aumento tende a diminuir o preço.
                """)
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

if __name__ == "__main__":
    main()
