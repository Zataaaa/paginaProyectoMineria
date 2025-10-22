import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

# 🎨 Configuración visual
st.set_page_config(page_title="Seguridad Vial 2024", layout="wide")
sns.set_style("whitegrid")

# 🏷️ Título
st.title("Seguridad Vial - Enero 2024")

# 📥 Cargar datos
archivo = st.file_uploader("Carga tu archivo CSV", type=["csv"])
if archivo is not None:
    df = pd.read_csv(archivo)
    st.subheader("Vista previa del dataset original")
    st.dataframe(df)

    # 🔍 Diagnóstico de nulos
    st.subheader("Valores nulos por columna")
    st.dataframe(df.isnull().sum())

    # 🧹 Eliminar columnas con más del 50% de nulos, excepto columnas clave
    columnas_clave = ['EDAD LESIONADO', 'GENERO LESIONADO']
    columnas_a_eliminar = [col for col in df.columns if df[col].isnull().mean() > 0.5 and col not in columnas_clave]
    df.drop(columns=columnas_a_eliminar, inplace=True)
    st.write(f"Se eliminaron {len(columnas_a_eliminar)} columnas con más del 50% de valores nulos.")
    st.write("Columnas eliminadas:", columnas_a_eliminar)

    # 🔁 Limpieza basada en columnas clave
    columnas_existentes = [col for col in columnas_clave if col in df.columns]
    if columnas_existentes:
        df_clean = df.dropna(subset=columnas_existentes)
        st.write(f"Se eliminaron {len(df) - len(df_clean)} filas con nulos en columnas clave.")
    else:
        st.warning("Las columnas clave no están disponibles. Se usará el dataset original.")
        df_clean = df.copy()

    st.subheader("Dataset después de limpieza")
    st.dataframe(df_clean)

    # 📊 Visualización: género de lesionados
    if 'GENERO LESIONADO' in df_clean.columns:
        st.subheader("Frecuencia por género de lesionado")
        fig1, ax1 = plt.subplots()
        df_clean['GENERO LESIONADO'].value_counts().plot(kind='bar', ax=ax1, color='lightblue')
        ax1.set_xlabel("Género")
        ax1.set_ylabel("Cantidad")
        st.pyplot(fig1)

    # 📊 Visualización: edad de lesionados
    if 'EDAD LESIONADO' in df_clean.columns:
        st.subheader("Distribución de edad de lesionados")
        fig2, ax2 = plt.subplots()
        sns.histplot(df_clean['EDAD LESIONADO'], bins=20, kde=True, ax=ax2, color='orange')
        ax2.set_xlabel("Edad del lesionado")
        st.pyplot(fig2)

    # 🔁 Transformación: normalización de edad
    if 'EDAD LESIONADO' in df_clean.columns and df_clean['EDAD LESIONADO'].notnull().any():
        scaler = MinMaxScaler()
        df_clean['EDAD NORMALIZADA'] = scaler.fit_transform(df_clean[['EDAD LESIONADO']])
        st.subheader("Edad normalizada")
        st.line_chart(df_clean['EDAD NORMALIZADA'])

    # 📘 Justificación
    st.markdown("""
    ### Justificación
    - Se muestra el dataset original para entender la estructura y calidad de los datos.
    - Se eliminaron columnas con más del 50% de valores nulos, preservando las variables clave.
    - Se eliminaron filas con nulos en edad y género para asegurar consistencia.
    - Se visualizó la distribución de edad antes y después de la limpieza.
    - Se normalizó la edad para facilitar comparaciones entre variables.
    """)






