import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go



####

# URL de la foto que quieras aqu iabajo
URL_IMAGEN_FONDO = "https://www.rwcarbon.com/media/uploads_ckeditor/rw-carbon-fiber-diffuser-audi-rs6-1.jpg"


page_bg_img = f"""
<style>
/* 1. Establecer la imagen de fondo en el contenedor principal */
[data-testid="stAppViewContainer"] {{
    background-image: url("{URL_IMAGEN_FONDO}");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: fixed;
}}

/* 2. Opcional: Hacer el fondo de la barra lateral gris */
[data-testid="stSidebar"] {{
    background-color: gray !important;
}}

/* 3. Opcional: Asegurar que el área principal de contenido también sea transparente (si la usas) */
.main {{
    background-color: transparent;
}}

p {{
    color: #D65452;
}}


</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)


####

dfa = pd.read_csv("audi.csv")


st.sidebar.header("Filtos")


Model_select_filter = st.sidebar.multiselect("Select Model: ", dfa["model"].unique(), default= dfa["model"].unique())
audi_year_slider = st.sidebar.slider("Select car year: ", dfa["year"].min(), dfa["year"].max() , (dfa["year"].min(), dfa["year"].max()))

filtered_df = dfa[(dfa["model"].isin(Model_select_filter)) & (dfa["year"].between(audi_year_slider[0],audi_year_slider[1]))]

# Configuración inicial de la página de Streamlit
st.set_page_config(
    page_title="Dashboard Audi",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título principal del dashboard
st.markdown("<h1 style ='text-align: center; color: #D65452;'> Dashboard de Análisis de Autos Usados - Audi</h1>", unsafe_allow_html = True)


# Función para cargar los datos desde el archivo CSV
@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv("audi.csv")
    # Conversión de columnas numéricas a tipo numérico, manejando errores
    numeric_columns = ['year', 'price', 'mileage', 'tax', 'mpg', 'engineSize']
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    # Conversión de la columna 'year' a entero
    df['year'] = df['year'].astype(int)
    return df

# Bloque try-except para manejar la carga de datos
try:
    df = load_data()
    # Insignia de confirmación de carga de datos
    st.badge(
        f" Se cargaron correctamente {len(df):,} registros",
        icon=":material/check:",
        color="green"
    )

    # Sección de indicadores principales
    st.header("Indicadores Principales")

    # Cálculo de métricas clave
    total_cars = len(df)
    avg_price = df['price'].mean()
    avg_mileage = df['mileage'].mean()
    avg_engine = df['engineSize'].mean()

    # Columnas para mostrar las métricas principales
    met_col1, met_col2, met_col3, met_col4 = st.columns(4)

    # Métrica: Precio promedio
    with met_col1:
        st.metric(
            "Precio promedio",
            f"£{avg_price:,.2f}",
            help="Promedio de los precios de venta de los vehículos en libras esterlinas"
        )

    # Métrica: Kilometraje promedio
    with met_col2:
        st.metric(
            "Kilometraje promedio",
            f"{avg_mileage:,.0f} km",
            help="Promedio de kilómetros/millas recorridos por los vehículos"
        )

    # Métrica: Total de autos
    with met_col3:
        st.metric(
            "Total de autos",
            f"{total_cars:,}",
            help="Número total de vehículos disponibles en el conjunto de datos"
        )

    # Métrica: Cilindrada promedio
    with met_col4:
        st.metric(
            "Cilindrada promedio (L)",
            f"{avg_engine:.2f}",
            help="Promedio de la cilindrada del motor en litros"
        )

    # Definición de las pestañas para organizar el contenido
    tab_relaciones, tab_modelos, tab_distribucion, tab_tendencia, tab_eda = st.tabs([
        "Relación entre Variables",
        "Análisis de Modelos",
        "Distribución",
        "Tendencia por Año",
        "Exploratory data analysis"
    ])

    # Contenido de la pestaña "Tendencia por Año"

    with tab_tendencia:
        st.subheader("Tendencia de métricas por año")

        # Opciones de métricas para el filtro de tendencia
        metric_options = [
            "Precio promedio",
            "Precio mediano",
            "Kilometraje promedio",
            "Eficiencia de combustible promedio (MPG)",
            "Cilindrada promedio",
            "Impuesto promedio",
            "Número de autos"
        ]
        # Filtro para seleccionar la métrica a visualizar
        selected_metric = st.selectbox("Selecciona la métrica", metric_options, key="metric_trend_select")

        # Agregación de datos por año
        df_year = df.groupby('year').agg({
            'price': ['mean', 'median'],
            'mileage': 'mean',
            'mpg': 'mean',
            'engineSize': 'mean',
            'tax': 'mean',
            'model': 'count'
        }).reset_index()

        # Renombrar columnas para mayor claridad
        df_year.columns = [
            'year',
            'precio_promedio',
            'precio_mediano',
            'kilometraje_promedio',
            'mpg_promedio',
            'cilindrada_promedio',
            'impuesto_promedio',
            'numero_autos'
        ]

        # Mapeo de la métrica seleccionada a la columna correspondiente
        metric_mapping = {
            "Precio promedio": 'precio_promedio',
            "Precio mediano": 'precio_mediano',
            "Kilometraje promedio": 'kilometraje_promedio',
            "Eficiencia de combustible promedio (MPG)": 'mpg_promedio',
            "Cilindrada promedio": 'cilindrada_promedio',
            "Impuesto promedio": 'impuesto_promedio',
            "Número de autos": 'numero_autos'
        }

        y_column = metric_mapping[selected_metric]

        # Título dinámico del gráfico de líneas
        title_line_chart = f"{selected_metric} por año"

        # Gráfico de líneas de Plotly
        fig_line = px.line(
            df_year,
            x='year',
            y=y_column,
            markers=True,
            title=title_line_chart,
            labels={'year': 'Año', y_column: selected_metric},
            height=400
        )

        fig_line.update_traces(line_color='#0072B5', line_width=3)
        fig_line.update_layout(hovermode='x unified')

        st.plotly_chart(fig_line, use_container_width=True)

    # Contenido de la pestaña "Análisis de Modelos"
    with tab_modelos:
        st.subheader("Análisis de modelos")

        # Número de modelos únicos y valor por defecto para el slider
        num_models = df['model'].nunique()
        default_top_n = min(10, num_models)

        # Filtro Slider para seleccionar el número de modelos a mostrar
        top_n = st.slider(
            "Selecciona el número de modelos a mostrar", min_value=5, max_value=num_models,
            value=default_top_n, step=1, key="model_slider"
        )

        # Opciones de métricas para el filtro de modelos
        model_metric_options = [
            "Precio promedio",
            "Precio mediano",
            "Kilometraje promedio",
            "Número de autos"
        ]
        # Filtro para seleccionar la métrica de ordenación de modelos
        selected_model_metric = st.selectbox("Ordenar por", model_metric_options, key="model_metric_select")

        # Agregación de datos por modelo
        df_model = df.groupby('model').agg({
            'price': ['mean', 'median'],
            'mileage': 'mean',
            'model': 'count'
        }).reset_index()

        # Renombrar columnas para mayor claridad
        df_model.columns = [
            'model',
            'precio_promedio',
            'precio_mediano',
            'kilometraje_promedio',
            'numero_autos'
        ]

        # Mapeo de la métrica seleccionada a la columna correspondiente
        model_metric_mapping = {
            "Precio promedio": 'precio_promedio',
            "Precio mediano": 'precio_mediano',
            "Kilometraje promedio": 'kilometraje_promedio',
            "Número de autos": 'numero_autos'
        }

        order_column = model_metric_mapping[selected_model_metric]

        # Ordenar y seleccionar los modelos principales
        df_model_sorted = df_model.sort_values(order_column, ascending=False).head(top_n)

        # Gráfico de barras horizontal de Plotly para modelos
        fig_model = px.bar(
            df_model_sorted,
            y='model',
            x=order_column,
            orientation='h',
            title=f"Top {top_n} modelos según {selected_model_metric.lower()}",
            color=order_column,
            color_continuous_scale='Sunset',
            height=500
        )

        fig_model.update_layout(
            yaxis={'categoryorder': 'total ascending'},
            xaxis_title=selected_model_metric,
            yaxis_title="Modelo"
        )

        st.plotly_chart(fig_model, use_container_width=True)

    # Contenido de la pestaña "Distribución"
    with tab_distribucion:
        st.subheader("Distribución de transmisión y tipo de combustible")

        # Columnas para mostrar los gráficos de pastel
        dist_col1, dist_col2 = st.columns(2)

        # Gráfico de pastel para la distribución por tipo de transmisión
        with dist_col1:
            trans_counts = df['transmission'].value_counts().reset_index()
            trans_counts.columns = ['transmission', 'count']
            fig_trans = px.pie(
                trans_counts,
                values='count',
                names='transmission',
                title="Distribución por tipo de transmisión",
                hole=0.4,
                color_discrete_sequence=px.colors.sequential.Purpor
            )
            fig_trans.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_trans, use_container_width=True)

        # Gráfico de pastel para la distribución por tipo de combustible
        with dist_col2:
            fuel_counts = df['fuelType'].value_counts().reset_index()
            fuel_counts.columns = ['fuelType', 'count']
            fig_fuel = px.pie(
                fuel_counts,
                values='count',
                names='fuelType',
                title="Distribución por tipo de combustible",
                hole=0.4,
                color_discrete_sequence=px.colors.sequential.Viridis
            )
            fig_fuel.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_fuel, use_container_width=True)

    # Contenido de la pestaña "Relación entre Variables"
    with tab_relaciones:
        st.subheader("Relación entre variables")

        # Variables numéricas disponibles para el análisis de relación
        numeric_vars = ['price', 'mileage', 'mpg', 'engineSize', 'tax', 'year']

        # Filtro para seleccionar la variable del eje X
        x_var = st.selectbox(
            "Variable en el eje X", numeric_vars, index=numeric_vars.index('mileage'), key="x_var_select"
        )

        # Opciones para el eje Y, excluyendo la variable del eje X
        y_options = [var for var in numeric_vars if var != x_var]
        # Filtro para seleccionar la variable del eje Y
        y_var = st.selectbox(
            "Variable en el eje Y", y_options, index=0, key="y_var_select"
        )

        # Opciones para colorear los puntos del gráfico de dispersión
        color_options = ['Ninguno', 'fuelType', 'transmission', 'model']
        # Filtro para seleccionar la variable de color
        selected_color = st.selectbox(
            "Color por", color_options, index=0, key="color_select"
        )

        # Etiquetas amigables para los ejes de los gráficos
        axis_labels = {
            'price': 'Precio (£)',
            'mileage': 'Kilometraje',
            'mpg': 'Eficiencia (MPG)',
            'engineSize': 'Cilindrada (L)',
            'tax': 'Impuesto (£)',
            'year': 'Año'
        }

        # Creación del gráfico de dispersión según la selección de color
        if selected_color == 'Ninguno':
            fig_scatter = px.scatter(
                df,
                x=x_var,
                y=y_var,
                labels={x_var: axis_labels.get(x_var, x_var), y_var: axis_labels.get(y_var, y_var)},
                title=f"Relación entre {axis_labels.get(x_var, x_var).lower()} y {axis_labels.get(y_var, y_var).lower()}",
                opacity=0.7,
                trendline='ols'
            )
        else:
            fig_scatter = px.scatter(
                df,
                x=x_var,
                y=y_var,
                color=selected_color,
                labels={
                    x_var: axis_labels.get(x_var, x_var),
                    y_var: axis_labels.get(y_var, y_var),
                    selected_color: selected_color
                },
                title=f"Relación entre {axis_labels.get(x_var, x_var).lower()} y {axis_labels.get(y_var, y_var).lower()} por {selected_color}",
                opacity=0.7,
                trendline='ols'
            )

        fig_scatter.update_traces(marker=dict(size=6))

        st.plotly_chart(fig_scatter, use_container_width=True)

    with tab_eda:
        st.title("Hola")
        st.title("Dataframe Head: ")
        st.dataframe(filtered_df.head())
        st.title("Dataframe Describe: ")
        st.dataframe(filtered_df.describe())
        st.title("Dataframe Dtypes: ")
        st.write(filtered_df.dtypes)

        st.title("Precio VS Año: ")

        fig, ax = plt.subplots(figsize=(10, 6))


        sns.scatterplot(x="year", y="price", data=filtered_df, color="red", ax=ax)

        sns.regplot(x="year", y="price", data=filtered_df, scatter=False, color="blue", ax=ax)
        st.pyplot(fig)
        plt.close(fig)

        # Plotly graphs
        st.title("Distribución de Tipos de Transmisión")
        fig_pie = px.pie(dfa, names="transmission", color="transmission", title="Distribución de Tipos de Transmisión")
        st.plotly_chart(fig_pie)

        st.title("Mapa de Calor de Correlación de Características Numéricas")
        numeric_df = dfa.select_dtypes(include=['number'])
        corr_matrix = numeric_df.corr()

        fig_heatmap = go.Figure(data=go.Heatmap(
                           z=corr_matrix.values,
                           x=corr_matrix.columns,
                           y=corr_matrix.index,
                           colorscale='Reds'))

        fig_heatmap.update_layout(
            title='Mapa de Calor de Correlación de Características Numéricas',
            xaxis_title='Características',
            yaxis_title='Características'
        )
        st.plotly_chart(fig_heatmap)



# Manejo de excepciones si falla la carga de datos
except Exception as e:
    st.error(f"Error al cargar el archivo audi.csv: {e}")
    st.stop()
