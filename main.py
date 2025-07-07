import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder
import plotly.graph_objects as go
import plotly.express as px

@st.cache_data
def cargar_datos():
    df = pd.read_csv("ventas_antimicrobial.csv")
    df['fecha'] = pd.to_datetime(df['fecha'])

    industria_map = {
        'cmpc': 'forestal_construccion',
        'lippi': 'textil',
        'petco': 'mascotas'
    }
    df['industria'] = df['cliente'].map(industria_map)

    df['mes'] = df['fecha'].dt.month
    df['año'] = df['fecha'].dt.year
    df['trimestre'] = df['fecha'].dt.quarter
    df['es_fin_de_mes'] = df['fecha'].dt.is_month_end.astype(int)

    ventas_cliente = (
        df.groupby(['cliente', 'año'])['unidades']
        .sum().reset_index().rename(columns={'unidades': 'ventas_anuales_cliente'})
    )
    df = df.merge(ventas_cliente, on=['cliente', 'año'], how='left')

    df = df.sort_values(by=['cliente', 'fecha'])
    df['lag_1'] = df.groupby('cliente')['unidades'].shift(1)
    df = df.dropna()

    le_cliente = LabelEncoder()
    le_industria = LabelEncoder()
    df['cliente_enc'] = le_cliente.fit_transform(df['cliente'])
    df['industria_enc'] = le_industria.fit_transform(df['industria'])

    return df

# --------- Entrenamiento del modelo ---------
def entrenar_modelo(df):
    features = [
        'cliente_enc', 'industria_enc', 'mes', 'año', 'trimestre',
        'es_fin_de_mes', 'ventas_anuales_cliente', 'lag_1'
    ]
    X = df[features]
    y = df['unidades']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=4)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return model, X_test, y_test, y_pred

# --------- Gráfico 1: Real vs Predicción ---------
def graficar_calibracion(X_test, y_test, y_pred, df):
    plot_df = X_test.copy()
    plot_df['fecha'] = df.loc[X_test.index, 'fecha']
    plot_df['cliente'] = df.loc[X_test.index, 'cliente']
    plot_df['y_true'] = y_test
    plot_df['y_pred'] = y_pred

    clientes = plot_df['cliente'].unique()
    color_map = dict(zip(clientes, px.colors.qualitative.Set2))
    fig = go.Figure()

    for cliente in clientes:
        cliente_df = plot_df[plot_df['cliente'] == cliente].sort_values(by='fecha')
        color = color_map[cliente]

        fig.add_trace(go.Scatter(
            x=cliente_df['fecha'],
            y=cliente_df['y_true'],
            mode='lines+markers',
            name=f'{cliente} - Real',
            line=dict(dash='solid', color=color)
        ))
        fig.add_trace(go.Scatter(
            x=cliente_df['fecha'],
            y=cliente_df['y_pred'],
            mode='lines+markers',
            name=f'{cliente} - Predicho',
            line=dict(dash='dot', color=color)
        ))

    fig.update_layout(
        title='📈 Real vs Predicción por Cliente',
        xaxis_title='Fecha',
        yaxis_title='Unidades',
        legend_title='Serie',
        template='plotly_white'
    )
    return fig

# --------- Gráficos 2 y 3: Barras agrupadas y apiladas ---------
def graficar_predicciones_futuras(model, df):
    ultimas_fechas = df.groupby('cliente')['fecha'].max().reset_index()
    proyecciones = []

    for _, row in ultimas_fechas.iterrows():
        cliente = row['cliente']
        base_fecha = row['fecha']
        cliente_data = df[df['cliente'] == cliente].sort_values(by='fecha').iloc[-1]
        for i in range(1, 4):
            nueva_fecha = pd.to_datetime(f"{2025}-{i}-01")
            datos = {
                'cliente': cliente,
                'industria': cliente_data['industria'],
                'mes': nueva_fecha.month,
                'año': nueva_fecha.year,
                'trimestre': (nueva_fecha.month - 1) // 3 + 1,
                'es_fin_de_mes': 0,
                'ventas_anuales_cliente': cliente_data['ventas_anuales_cliente'],
                'lag_1': cliente_data['unidades'],
                'cliente_enc': cliente_data['cliente_enc'],
                'industria_enc': cliente_data['industria_enc'],
                'fecha': nueva_fecha
            }
            proyecciones.append(datos)

    df_pred = pd.DataFrame(proyecciones)
    features = [
        'cliente_enc', 'industria_enc', 'mes', 'año', 'trimestre',
        'es_fin_de_mes', 'ventas_anuales_cliente', 'lag_1'
    ]
    df_pred['pred_unidades'] = model.predict(df_pred[features])
    df_pred['mes_nombre'] = df_pred['fecha'].dt.strftime('%b')
    df_pred['mes_nombre'] = pd.Categorical(df_pred['mes_nombre'], categories=['Jan', 'Feb', 'Mar'], ordered=True)

    # Gráfico 1: Agrupado
    fig1 = px.bar(
        df_pred,
        x='mes_nombre',
        y='pred_unidades',
        color='cliente',
        barmode='group',
        title='📊 Predicción agrupada por mes y cliente',
        labels={'mes_nombre': 'Mes', 'pred_unidades': 'Unidades'}
    )

    # Gráfico 2: Apilado
    fig2 = px.bar(
        df_pred,
        x='mes_nombre',
        y='pred_unidades',
        color='cliente',
        barmode='stack',
        title='📊 Predicción total mensual',
        labels={'mes_nombre': 'Mes', 'pred_unidades': 'Unidades'}
    )

    return fig1, fig2

# --------- App Streamlit ---------
st.set_page_config(page_title="Predicción de Ventas Antimicrobial", layout="wide")
st.title("Predicción de Ventas - Producto Antimicrobial")

df = cargar_datos()
model, X_test, y_test, y_pred = entrenar_modelo(df)

st.subheader("1. Calibración del modelo")
fig_cal = graficar_calibracion(X_test, y_test, y_pred, df)
st.plotly_chart(fig_cal, use_container_width=True)

st.subheader("2. Predicción para Q1 2025")
fig_bar_group, fig_bar_stack = graficar_predicciones_futuras(model, df)
col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(fig_bar_group, use_container_width=True)
with col2:
    st.plotly_chart(fig_bar_stack, use_container_width=True)
