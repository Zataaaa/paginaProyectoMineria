import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(page_title="Preparación de Datos", layout="wide")
st.title("Preparación y Visualización de Datos de Siniestros")

archivo = st.file_uploader("Carga tu archivo CSV", type=["csv"])
if archivo:
    df_original = pd.read_csv(archivo)
    df = df_original.copy()

    st.subheader("Dataset Original (con nulos y duplicados)")
    st.dataframe(df_original)

    df.replace("\\N", pd.NA, inplace=True)
    df = df.drop_duplicates()
    if 'CAUSA SINIESTRO' in df.columns:
        df = df.drop(columns=['CAUSA SINIESTRO'])

    cols_categoricas = ['COLONIA', 'TIPO VEHICULO', 'COLOR', 'NIVEL DAÑO VEHICULO',
                        'PUNTO DE IMPACTO', 'CIUDAD', 'GENERO LESIONADO', 'CALLE',
                        'RELACION LESIONADOS']
    cols_categoricas = [col for col in cols_categoricas if col in df.columns]
    for col in cols_categoricas:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].mode()[0], inplace=True)

    cols_numericas = ['EDAD LESIONADO', 'MODELO']
    cols_numericas = [col for col in cols_numericas if col in df.columns]
    for col in cols_numericas:
        if col != 'EDAD LESIONADO' and df[col].isnull().sum() > 0:
            df[col].fillna(round(df[col].mean()), inplace=True)

    cols_binarias = ['AMBULANCIA', 'ARBOL', 'PIEDRA', 'DORMIDO', 'GRUA', 'OBRA CIVIL',
                     'PAVIMENTO MOJADO', 'EXPLOSION LLANTA', 'VOLCADURA', 'PERDIDA TOTAL',
                     'CONDUCTOR DISTRAIDO', 'FUGA', 'ALCOHOL', 'MOTOCICLETA', 'BICICLETA',
                     'SEGURO', 'TAXI', 'ANIMAL']
    cols_binarias = [col for col in cols_binarias if col in df.columns]
    df[cols_binarias] = df[cols_binarias].fillna(0)

    if 'FALLECIDO' in df.columns:
        df['FALLECIDO'] = df['FALLECIDO'].map({'SI': 1, 'NO': 0}).fillna(0).astype(int)
    if 'HOSPITALIZADO' in df.columns:
        df['HOSPITALIZADO'] = df['HOSPITALIZADO'].map({'SI': 1, 'NO': 0}).fillna(0).astype(int)

    cols_binarias = ['HOSPITALIZADO', 'FALLECIDO'] + cols_binarias
    cols_binarias = [col for col in cols_binarias if col in df.columns]
    df[cols_binarias] = df[cols_binarias].astype(int)

    if 'SINIESTRO' in df.columns:
        df.set_index('SINIESTRO', inplace=True)

    st.subheader("Dataset Limpio (sin duplicados, con imputación parcial)")
    st.dataframe(df)

    
    fig, ax = plt.subplots()
    sns.kdeplot(df_original['EDAD LESIONADO'].dropna(), label="Original", color="salmon", fill=True, alpha=0.4, ax=ax)
    sns.kdeplot(df['EDAD LESIONADO'].dropna(), label="Limpio", color="seagreen", fill=True, alpha=0.4, ax=ax)
    ax.set_xlabel("Edad")
    ax.set_ylabel("Densidad")
    ax.set_title("Distribución de Edad: Antes vs Después")
    ax.legend()
    st.pyplot(fig)
    
   

    if 'MODELO' in df.columns:
        st.subheader(" Transformación por Normalización")
        scaler = MinMaxScaler()
        df_norm = df.copy()
        df_norm['MODELO_NORMALIZADO'] = scaler.fit_transform(df[['MODELO']])

        fig3, ax3 = plt.subplots()
        sns.boxplot(data=df_norm[['MODELO', 'MODELO_NORMALIZADO']], ax=ax3)
        ax3.set_title("Comparación antes y después de normalizar años carros")
        st.pyplot(fig3)

    st.subheader(" Causas más frecuentes de siniestros")
    causas = ['CONDUCTOR DISTRAIDO', 'ALCOHOL', 'PAVIMENTO MOJADO', 'DORMIDO', 'EXPLOSION LLANTA']
    causas = [c for c in causas if c in df.columns]
    conteo_causas = df[causas].sum().sort_values(ascending=False)

    fig_causas, ax_causas = plt.subplots()
    sns.barplot(x=conteo_causas.values, y=conteo_causas.index, ax=ax_causas)
    ax_causas.set_xlabel("Número de siniestros")
    ax_causas.set_ylabel("Causa")
    ax_causas.set_title("Causas más frecuentes")
    st.pyplot(fig_causas)

    st.subheader("Zonas con mayor concentración de accidentes")
    if 'CIUDAD' in df.columns:
        top_ciudades = df['CIUDAD'].value_counts().head(10)
        fig_ciudades, ax_ciudades = plt.subplots()
        sns.barplot(x=top_ciudades.values, y=top_ciudades.index, ax=ax_ciudades)
        ax_ciudades.set_xlabel("Número de siniestros")
        ax_ciudades.set_ylabel("Ciudad")
        ax_ciudades.set_title("Top 10 ciudades con más siniestros")
        st.pyplot(fig_ciudades)
        
    st.subheader(" Discretización de Edad de Lesionados")

    if 'EDAD LESIONADO' in df.columns:
        bins = [0, 18, 30, 50, 70, 100]
        labels = ['Menor', 'Joven', 'Adulto', 'Mayor', 'Anciano']
        df['EDAD_GRUPO'] = pd.cut(df['EDAD LESIONADO'], bins=bins, labels=labels, right=False)

        grupo_counts = df['EDAD_GRUPO'].value_counts().sort_index()

        fig_disc, ax_disc = plt.subplots()
        sns.barplot(x=grupo_counts.index, y=grupo_counts.values, palette="viridis", ax=ax_disc)
        ax_disc.set_xlabel("Grupo de Edad")
        ax_disc.set_ylabel("Número de Lesionados")
        ax_disc.set_title("Distribución por Grupo de Edad (Discretización)")
        st.pyplot(fig_disc)









