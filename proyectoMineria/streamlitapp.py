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

    # 🧹 Eliminar columnas con más del 50% de nulos
    umbral = 0.5
    columnas_a_eliminar = df.columns[df.isnull().mean() > umbral]
    df.drop(columns=columnas_a_eliminar, inplace=True)
    st.write(f"Se eliminaron {len(columnas_a_eliminar)} columnas con más del {int(umbral*100)}% de valores nulos.")
    st.write("Columnas eliminadas:", list(columnas_a_eliminar))

    # 🔁 Limpieza basada en columnas clave
    columnas_clave = ['EDAD LESIONADO', 'GENERO LESIONADO']
    df_clean = df.dropna(subset=columnas_clave)

    # 📊 Mostrar dataset limpio
    st.subheader("Dataset después de limpieza")
    st.write(f"Se eliminaron {len(df) - len(df_clean)} filas con nulos en edad o género del lesionado.")
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
    if 'EDAD LESIONADO' in df_clean.columns:
        scaler = MinMaxScaler()
        df_clean['EDAD NORMALIZADA'] = scaler.fit_transform(df_clean[['EDAD LESIONADO']])
        st.subheader("Edad normalizada")
        st.line_chart(df_clean['EDAD NORMALIZADA'])

    # 📘 Justificación
    st.markdown("""
    ### Justificación
    - Se muestra el dataset original para entender la estructura y calidad de los datos.
    - Se eliminaron columnas con más del 50% de valores nulos para mejorar la calidad del análisis.
    - Se eliminaron valores nulos en edad y género para asegurar consistencia.
    - Se visualizó la distribución de edad antes y después de la limpieza.
    - Se normalizó la edad para facilitar comparaciones entre variables.
    """)




