import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.seasonal import seasonal_decompose

# Configuración de la página para un diseño más amplio
st.set_page_config(layout="wide", page_title="Dashboard de análisis de accidentes")

# CSS personalizado para mejorar el diseño
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6;
    }
    .sidebar .sidebar-content {
        background: #262730;
    }
    .Widget>label {
        color: #262730;
        font-family: sans-serif;
    }
    .stPlotlyChart {
        background-color: #ffffff;
        border-radius: 5px;
        box-shadow: 0 2px 4px rgba(0,0,0,.1);
    }
</style>
""", unsafe_allow_html=True)

# Cargar y preprocesar los datos
@st.cache_data
def load_data():
    df = pd.read_csv("clean_data.csv")
    df['FECHA_ACCIDENTE'] = pd.to_datetime(df['FECHA_ACCIDENTE'], format='%d-%m-%Y', errors='coerce')
    df = df.dropna(subset=['FECHA_ACCIDENTE'])
    
    # Crear columnas de mes y año
    df['Month'] = df['FECHA_ACCIDENTE'].dt.month
    df['Year'] = df['FECHA_ACCIDENTE'].dt.year
    
    # Crear grupos de edad
    df['Age_Group'] = pd.cut(df['EDAD'], bins=[0, 18, 30, 40, 50, 60, 100], labels=['<18', '18-30', '31-40', '41-50', '51-60', '60+'])
    
    return df

df = load_data()

# Filtros en la barra lateral
st.sidebar.title("Filtros")
selected_years = st.sidebar.multiselect("Selecciona Años", sorted(df['Year'].unique()), default=df['Year'].max())
selected_depts = st.sidebar.multiselect("Selecciona Departamentos", sorted(df['DEPARTAMENTO'].unique()))
selected_gravity = st.sidebar.multiselect("Selecciona Gravedad del Accidente", sorted(df['GRAVEDAD_ACCIDENTE'].unique()))

# Aplicar filtros
mask = df['Year'].isin(selected_years)
if selected_depts:
    mask &= df['DEPARTAMENTO'].isin(selected_depts)
if selected_gravity:
    mask &= df['GRAVEDAD_ACCIDENTE'].isin(selected_gravity)
filtered_df = df[mask]

# Calcular indicadores clave
total_accidents = len(filtered_df)
avg_days_off = filtered_df['DIAS_DESCANZO'].mean()
total_cost = filtered_df['MONTO_DESCANSO'].sum()
most_common_type = filtered_df['TIPO_ACCIDENTE'].mode()[0]
most_affected_part = filtered_df['PARTE_AFECTADA'].mode()[0]

# Dashboard principal
st.title("Dashboard de análisis de accidentes en Perú")

# Métricas clave
col1, col2, col3= st.columns(3)
col1.metric("Accidentes Totales", f"{total_accidents:,}")
col2.metric("Promedio Días de Descanso", f"{avg_days_off:.2f}")
col3.metric("Costo Total", f"${total_cost:,.2f}")
col4, col5 = st.columns(2)
# Ajustar el tamaño de la métrica
col4.metric("Tipo más Común", most_common_type)
col5.metric("Parte más Afectada", most_affected_part)

# Crear dos columnas para gráficos
left_column, right_column = st.columns(2)

with left_column:
    # Accidentes por Departamento (Top 10) - Gráfico de Barras Horizontal
    st.subheader("Top 10 Departamentos por Número de Accidentes")
    dept_counts = filtered_df['DEPARTAMENTO'].value_counts().head(10)
    fig_dept = px.bar(dept_counts, x=dept_counts.values, y=dept_counts.index, orientation='h',
                      color=dept_counts.values, color_continuous_scale='Viridis')
    fig_dept.update_layout(height=400, xaxis_title="Número de Accidentes", yaxis_title="Departamento")
    st.plotly_chart(fig_dept, use_container_width=True)

    # Accidentes por Mes - Gráfico de Líneas
    st.subheader("Accidentes por Mes")
    month_counts = filtered_df.groupby(['Year', 'Month']).size().unstack(level=0)
    fig_month = px.line(month_counts, x=month_counts.index, y=month_counts.columns, 
                        labels={'value': 'Número de Accidentes', 'Month': 'Mes', 'variable': 'Año'},
                        color_discrete_sequence=px.colors.qualitative.Set2)
    fig_month.update_layout(height=400)
    st.plotly_chart(fig_month, use_container_width=True)

with right_column:
    # Accidentes por Gravedad - Gráfico de Pastel
    st.subheader("Accidentes por Gravedad")
    gravity_counts = filtered_df['GRAVEDAD_ACCIDENTE'].value_counts()
    fig_gravity = px.pie(gravity_counts, values=gravity_counts.values, names=gravity_counts.index,
                         color_discrete_sequence=px.colors.sequential.RdBu)
    fig_gravity.update_traces(textposition='inside', textinfo='percent+label')
    fig_gravity.update_layout(height=400)
    st.plotly_chart(fig_gravity, use_container_width=True)

    # Accidentes por Grupo de Edad y Género - Gráfico de Barras Agrupadas
    st.subheader("Accidentes por Grupo de Edad y Género")
    age_gender_counts = filtered_df.groupby(['Age_Group', 'SEXO_TRABAJADOR']).size().unstack(level=1)
    fig_age_gender = px.bar(age_gender_counts, x=age_gender_counts.index, y=age_gender_counts.columns,
                            labels={'value': 'Número de Accidentes', 'Age_Group': 'Grupo de Edad', 'variable': 'Género'},
                            color_discrete_sequence=px.colors.qualitative.Set3)
    fig_age_gender.update_layout(height=400, xaxis_title="Grupo de Edad", yaxis_title="Número de Accidentes")
    st.plotly_chart(fig_age_gender, use_container_width=True)

# Mapa de Perú mostrando la distribución de accidentes
st.subheader("Distribución de Accidentes en Perú")

# Obtener datos del mapa de Perú (puede que necesites descargar esto por separado)
import json
with open(r'C:\Users\USER\Desktop\MTPE\Analitica_MTPE\peru_departamental_simple.geojson') as f:
    peru_map = json.load(f)

# Agregar datos por departamento
dept_data = filtered_df.groupby('DEPARTAMENTO').agg({
    'MONTO_DESCANSO': 'sum',
    'DIAS_DESCANZO': 'mean',
    'EDAD': 'mean'
}).reset_index()

# Crear el mapa coroplético
fig_map = px.choropleth_mapbox(
    dept_data,
    geojson=peru_map,
    locations='DEPARTAMENTO',
    color='MONTO_DESCANSO',
    featureidkey="properties.NOMBDEP",
    center={"lat": -9.1900, "lon": -75.0152},
    mapbox_style="carto-positron",
    zoom=4,
    hover_data=['DIAS_DESCANZO', 'EDAD'],
    color_continuous_scale=px.colors.sequential.Plasma,
    range_color=(0, 350000)  # Ajustado para reflejar el valor máximo real
)
fig_map.update_layout(height=600, margin={"r":0,"t":0,"l":0,"b":0})
st.plotly_chart(fig_map, use_container_width=True)

# Análisis de Clustering
st.subheader("Análisis de Clustering de Accidentes")

# Preparar datos para clustering
cluster_data = filtered_df[['EDAD', 'DIAS_DESCANZO', 'MONTO_DESCANSO']].dropna()
scaler = StandardScaler()
scaled_data = scaler.fit_transform(cluster_data)

# Realizar clustering K-means
n_clusters = 5
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(scaled_data)

# Añadir etiquetas de cluster al dataframe
cluster_data['Cluster'] = cluster_labels

# Crear un gráfico de dispersión 3D
fig_cluster = px.scatter_3d(cluster_data, x='EDAD', y='DIAS_DESCANZO', z='MONTO_DESCANSO',
                            color='Cluster', hover_data=['EDAD', 'DIAS_DESCANZO', 'MONTO_DESCANSO'],
                            labels={'EDAD': 'Edad', 'DIAS_DESCANZO': 'Días de Descanso', 'MONTO_DESCANSO': 'Costo'},
                            color_continuous_scale=px.colors.sequential.Viridis)
fig_cluster.update_layout(height=700)
st.plotly_chart(fig_cluster, use_container_width=True)

# Características de los clusters
st.subheader("Características de los Clusters")
cluster_summary = cluster_data.groupby('Cluster').agg({
    'EDAD': 'mean',
    'DIAS_DESCANZO': 'mean',
    'MONTO_DESCANSO': 'mean'
}).round(2)
st.write(cluster_summary)

# 1. Análisis del Tipo de Accidente
st.subheader("Análisis del Tipo de Accidente")

# Gráfico de Pareto de tipos de accidentes
accident_type_counts = filtered_df['TIPO_ACCIDENTE'].value_counts()
cumulative_percentage = accident_type_counts.cumsum() / accident_type_counts.sum() * 100

fig_pareto = go.Figure()
fig_pareto.add_trace(go.Bar(x=accident_type_counts.index, y=accident_type_counts.values, name='Número de Accidentes'))
fig_pareto.add_trace(go.Scatter(x=accident_type_counts.index, y=cumulative_percentage, mode='lines', name='Porcentaje Acumulado', yaxis='y2'))

fig_pareto.update_layout(
    title='Gráfico de Pareto de Tipos de Accidentes',
    xaxis=dict(title='Tipo de Accidente'),
    yaxis=dict(title='Cantidad'),
    yaxis2=dict(title='Porcentaje Acumulado', overlaying='y', side='right', range=[0, 100]),
    height=500,
    legend=dict(x=1.1, y=5)  # Mover la leyenda más a la derecha
)
st.plotly_chart(fig_pareto, use_container_width=True)

# 2. Análisis de la Severidad de las Lesiones
st.subheader("Análisis de la Severidad de las Lesiones")

# Calcular promedio de días de descanso y costo por tipo de lesión
injury_severity = filtered_df.groupby('NATURALEZA_LESION').agg({
    'DIAS_DESCANZO': 'mean',
    'MONTO_DESCANSO': 'mean',
    'EDAD': 'mean'
}).reset_index()

fig_injury = px.scatter(injury_severity, x='DIAS_DESCANZO', y='MONTO_DESCANSO', 
                        size='EDAD', color='NATURALEZA_LESION', 
                        hover_name='NATURALEZA_LESION',
                        labels={'DIAS_DESCANZO': 'Promedio de Días de Descanso', 
                                'MONTO_DESCANSO': 'Promedio de Costo',
                                'EDAD': 'Promedio de Edad'},
                        title='Severidad de las Lesiones: Días de Descanso vs Costo')
st.plotly_chart(fig_injury, use_container_width=True)

st.subheader("Análisis del Perfil del Trabajador")

# Filtrar filas donde 'DIAS_DESCANZO' o 'MONTO_DESCANSO' son cero
filtered_df = filtered_df[(filtered_df['DIAS_DESCANZO'] != 0) & (filtered_df['MONTO_DESCANSO'] != 0)]

# Verificar si el dataframe filtrado está vacío
if filtered_df.empty:
    st.error("No hay datos válidos disponibles para calcular el puntaje de riesgo.")
else:
    # Crear un puntaje de riesgo basado en días de descanso y costo
    filtered_df['risk_score'] = (filtered_df['DIAS_DESCANZO'] * filtered_df['MONTO_DESCANSO']) / 1000

    # Analizar el puntaje de riesgo por características del trabajador
    worker_risk = filtered_df.groupby(['SEXO_TRABAJADOR', 'CATEGORIA_OCUPACIONAL', 'GRADI_INSTRUCCION']).agg({
        'risk_score': 'mean',
        'EDAD': 'mean'
    }).reset_index()

    # Verificar si la columna risk_score contiene solo ceros
    if worker_risk['risk_score'].sum() == 0:
        st.error("Los puntajes de riesgo calculados suman cero, no se puede crear el mapa de árbol.")
    else:
        fig_worker = px.treemap(worker_risk, 
                                path=['SEXO_TRABAJADOR', 'CATEGORIA_OCUPACIONAL', 'GRADI_INSTRUCCION'],
                                values='risk_score',
                                color='EDAD',
                                color_continuous_scale='RdYlBu_r',
                                title='Perfil de Riesgo del Trabajador')
        st.plotly_chart(fig_worker, use_container_width=True)

# 4. Patrones Temporales
st.subheader("Patrones Temporales de Accidentes")

# Extraer la hora de FECHA_ACCIDENTE
filtered_df['hour'] = filtered_df['FECHA_ACCIDENTE'].dt.hour

# Crear un mapa de calor de accidentes por día de la semana y hora
accident_heatmap = filtered_df.groupby([filtered_df['FECHA_ACCIDENTE'].dt.dayofweek, 'hour']).size().unstack()

fig_heatmap = px.imshow(accident_heatmap, 
                        labels=dict(x="Hora del Día", y="Día de la Semana", color="Número de Accidentes"),
                        x=accident_heatmap.columns,
                        y=['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo'],
                        title="Mapa de Calor de Accidentes por Día y Hora")
st.plotly_chart(fig_heatmap, use_container_width=True)

# 5. Análisis Multivariado
st.subheader("Análisis Multivariado: PCA")

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Seleccionar columnas numéricas para PCA
numeric_cols = ['EDAD', 'DIAS_DESCANZO', 'MONTO_DESCANSO']
X = filtered_df[numeric_cols]

# Estandarizar las características
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Realizar PCA
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# Crear un DataFrame con los resultados de PCA
pca_df = pd.DataFrame(data=X_pca[:, :2], columns=['PC1', 'PC2'])
pca_df['GRAVEDAD_ACCIDENTE'] = filtered_df['GRAVEDAD_ACCIDENTE']

# Graficar resultados de PCA
fig_pca = px.scatter(pca_df, x='PC1', y='PC2', color='GRAVEDAD_ACCIDENTE',
                     title='PCA de Variables Numéricas Coloreado por Severidad del Accidente')
st.plotly_chart(fig_pca, use_container_width=True)

# Mostrar la proporción de varianza explicada
explained_variance = pca.explained_variance_ratio_
st.write(f"Proporción de varianza explicada: PC1 = {explained_variance[0]:.2f}, PC2 = {explained_variance[1]:.2f}")

# Mostrar datos en bruto
st.subheader("Muestra de Datos en Bruto")
st.write(filtered_df.head(100))

# Añadir un botón de descarga para los datos filtrados
csv = filtered_df.to_csv(index=False)
st.download_button(
    label="Descargar datos filtrados como CSV",
    data=csv,
    file_name="filtered_accident_data.csv",
    mime="text/csv",
)