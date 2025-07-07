# Predicción de Ventas - Producto Antimicrobial

Predicción de Ventas Antimicrobial es una aplicación desarrollada en Streamlit que permite cargar datos históricos de ventas, entrenar un modelo de Machine Learning y visualizar las predicciones para el primer trimestre de 2025. La app está orientada a analizar el comportamiento por cliente y su industria, generando reportes visuales para la toma de decisiones comerciales.

## Funcionalidades

- Carga automática de datos históricos desde CSV  

- Entrenamiento de modelo de predicción (XGBoost)  

- Predicciones futuras de unidades vendidas (Q1 2025)  

- Gráfico de calibración Real vs Predicción por cliente  

- Gráficos de barras agrupadas y apiladas de ventas proyectadas  

## Modelo de Machine Learning

Se utiliza el modelo `XGBoost Regressor` con los siguientes features para la predicción:

 - `cliente_enc`	            Codificación numérica del cliente
 - `industria_enc`	            Codificación numérica de la industria
 - `mes`, `año`, `trimestre`	Variables temporales
 - `es_fin_de_mes`	            Indicador si la fecha es fin de mes
 - `ventas_anuales_cliente`     Total de unidades vendidas por cliente ese año
 - `lag_1`	                    Ventas del cliente en el periodo inmediatamente anterior

 ## Datos de entrada

 El archivo ventas_antimicrobial.csv debe contener al menos las siguientes columnas:

 - fecha: fecha de la venta

 - cliente: nombre del cliente

 - unidades: cantidad de unidades vendidas

La industria se asigna automáticamente según el cliente (puede personalizarse en el script).

## Ejecutar la APP

### Clonar Respositorio
```bash
 git clone https://github.com/crcordova/sales-prediction
 cd sales-prediction
```
 ### Instalar dependencias

```bash
 pip install -r requirements.txt
```
 ### Ejecutar app

```bash
 streamlit run main.py
```

### Probar con Mock Data
ejecutar jupyter `mock_data.ipynb` que generara un archivo de prueba con 3 cliente

## Visualizaciones incluidas

### Calibración del modelo
Gráfico de líneas comparando los valores reales y predichos por cliente a lo largo del tiempo.

### Predicción Q1 2025
 - Barras agrupadas: ventas proyectadas por cliente para cada mes.

 - Barras apiladas: visión consolidada de las ventas totales por mes.


## Futuras Mejoras

 - Subida manual de archivos desde la interfaz  

 - Ajuste de parámetros del modelo desde la app  

 - Predicciones multi-trimestre  

 - Exportación de reportes PDF  