import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import joblib
import sqlite3  # <--- AGREGADO: Para leer la base de datos portátil
from datetime import datetime

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Tablero AFOCAT", page_icon="🇵🇪", layout="wide")

# ==============================================================================
# 1. FUNCIONES DE CARGA (OPTIMIZADAS)
# ==============================================================================

@st.cache_resource
def cargar_inteligencia():
    """Carga el modelo y los encoders una sola vez en memoria."""
    try:
        model = joblib.load('modelo_xgboost_afocat_final.joblib')
        encoders = joblib.load('encoders_afocat.joblib')
        return model, encoders
    except Exception as e:
        return None, None

@st.cache_data(ttl=600) 
def cargar_datos_siniestros():
    """Conecta a la Base de Datos Portátil (SQLite)."""
    try:
        # --- CONEXIÓN PORTÁTIL (SQLITE) ---
        # Ya no usamos usuario/password, solo el archivo
        conn = sqlite3.connect('afocat_db.sqlite')
        
        query = "SELECT * FROM historial_predicciones"
        df = pd.read_sql(query, conn)
        conn.close() # Cerramos conexión
        
        # --- CAPA DE TRADUCCIÓN (Igual que antes) ---
        df.rename(columns={
            'DEPARTAMENTO': 'Dpto',
            'PROVINCIA': 'Prov',
            'DISTRITO': 'Distrito',
            'VEHÍCULO': 'Vehículo',
            'PREDICCION_IA': 'Severidad',
            'PROBABILIDAD_RIESGO': 'probabilidad',
            'COORDENADAS LATITUD': 'latitud',
            'COORDENADAS  LONGITUD': 'longitud',
            'HORA SINIESTRO': 'Hora_Raw'
        }, inplace=True)

        # Extraer Año, Mes y Hora Numérica
        df['FECHA SINIESTRO'] = pd.to_datetime(df['FECHA SINIESTRO'])
        df['Año'] = df['FECHA SINIESTRO'].dt.year
        
        # Mapa de meses manual para asegurar español
        mapa_meses = {1:'Enero', 2:'Febrero', 3:'Marzo', 4:'Abril', 5:'Mayo', 6:'Junio', 
                      7:'Julio', 8:'Agosto', 9:'Septiembre', 10:'Octubre', 11:'Noviembre', 12:'Diciembre'}
        df['MesNum'] = df['FECHA SINIESTRO'].dt.month
        df['Mes'] = df['MesNum'].map(mapa_meses)
        
        # Limpieza de Hora
        df['Hora'] = pd.to_numeric(df['Hora_Raw'].astype(str).str.split(':').str[0], errors='coerce').fillna(0).astype(int)

        # Crear Modalidad fallback
        if 'MODALIDAD' not in df.columns:
            df['Modalidad'] = df['Vehículo']
            
        # Crear columnas limpias para filtros
        df['Dpto_Clean'] = df['Dpto'].astype(str).str.strip().str.upper()
        df['Prov_Clean'] = df['Prov'].astype(str).str.strip().str.upper()
        df['Dist_Clean'] = df['Distrito'].astype(str).str.strip().str.upper()

        return df
    except Exception as e:
        st.error(f"Error cargando base de datos: {e}")
        return pd.DataFrame()

# --- CARGA INICIAL ---
df = cargar_datos_siniestros()
model, encoders = cargar_inteligencia()

if df.empty:
    st.warning("⚠️ No se encontraron datos. Asegúrate de haber ejecutado 'carga_datos.py' primero.")
    df = pd.DataFrame(columns=['Año', 'Mes', 'MesNum', 'Dpto', 'Dpto_Clean', 'Prov_Clean', 'Dist_Clean', 'Severidad', 'probabilidad'])

# ==============================================================================
# MENÚ LATERAL
# ==============================================================================
with st.sidebar:
    st.title("Navegación")
    vista_actual = st.radio(
        "Selecciona una vista:",
        ["📊 Dashboard Histórico", "🔮 Simulador de Riesgo (IA)"]
    )
    
    st.markdown("---")
    if st.button("🔄 Refrescar Datos", help="Recarga desde el archivo local"):
        st.cache_data.clear()
        st.rerun()

# ==============================================================================
# VISTA 1: DASHBOARD HISTÓRICO
# ==============================================================================
if vista_actual == "📊 Dashboard Histórico":
    
    st.sidebar.header("🔍 Filtros de Análisis")

    # 1. Filtro Año
    lista_anos = sorted(df['Año'].dropna().unique(), reverse=True)
    lista_anos.insert(0, "Todos")
    filtro_ano = st.sidebar.selectbox("Año:", lista_anos)

    # 2. Filtro Mes
    if 'MesNum' in df.columns:
        meses_ordenados = df[['Mes', 'MesNum']].drop_duplicates().sort_values('MesNum')
        lista_meses = meses_ordenados['Mes'].tolist()
    else:
        lista_meses = []
    lista_meses.insert(0, "Todos")
    filtro_mes = st.sidebar.selectbox("Mes:", lista_meses)

    # 3. Filtro Departamento
    lista_deptos = sorted(df['Dpto'].astype(str).unique())
    filtro_depto = st.sidebar.multiselect("Departamentos:", lista_deptos, default=lista_deptos)

    # --- LÓGICA DE FILTRADO ---
    df_filtrado = df.copy()

    if filtro_ano != "Todos":
        df_filtrado = df_filtrado[df_filtrado['Año'] == filtro_ano]

    if filtro_mes != "Todos":
        df_filtrado = df_filtrado[df_filtrado['Mes'] == filtro_mes]

    if filtro_depto:
        df_filtrado = df_filtrado[df_filtrado['Dpto'].isin(filtro_depto)]
    else:
        df_filtrado = df_filtrado[0:0]

    mapa_colores_severidad = {"FALLECIDO": "red", "LESIONADO": "orange", "ILESO": "green"}

    # --- UI DASHBOARD ---
    st.title("🇵🇪 Monitor de Inteligencia Vial")
    st.markdown(f"**Registros:** {len(df_filtrado)}")

    # KPIs
    if not df_filtrado.empty:
        k1, k2, k3, k4, k5 = st.columns(5)
        total = len(df_filtrado)
        n_falle = len(df_filtrado[df_filtrado['Severidad'] == 'FALLECIDO'])
        n_lesion = len(df_filtrado[df_filtrado['Severidad'] == 'LESIONADO'])
        n_ileso = len(df_filtrado[df_filtrado['Severidad'] == 'ILESO'])
        riesgo_val = df_filtrado['probabilidad'].mean()

        k1.metric("Total Eventos", f"{total:,}")
        k2.metric("🔴 Fallecidos", f"{n_falle}")
        k3.metric("🟠 Lesionados", f"{n_lesion}")
        k4.metric("🟢 Ilesos", f"{n_ileso}")
        k5.metric("📉 Riesgo Promedio", f"{riesgo_val:.1f}%")
    else:
        st.info("Selecciona filtros para ver datos.")

    st.markdown("---")

    col_mapa, col_top = st.columns([1.5, 1]) 

    with col_mapa:
        st.subheader("🗺️ Mapa de Riesgo")
        if not df_filtrado.empty:
            if 'Vía' not in df_filtrado.columns: df_filtrado['Vía'] = 'NO REGISTRADO'
            
            fig_mapa = px.scatter_mapbox(
                df_filtrado,
                lat="latitud", lon="longitud", color="Severidad", size="probabilidad",
                hover_name="Distrito",
                custom_data=["Prov", "probabilidad", "Vía"], 
                color_discrete_map=mapa_colores_severidad,
                zoom=4.5, height=500, mapbox_style="open-street-map"
            )
            fig_mapa.update_traces(
                hovertemplate="<b>%{hovertext}</b><br>Provincia: %{customdata[0]}<br>Riesgo: %{customdata[1]}%<br>Vía: %{customdata[2]}<extra></extra>"
            )
            fig_mapa.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
            st.plotly_chart(fig_mapa, use_container_width=True)

    with col_top:
        st.subheader("🚗 Severidad por Vehículo")
        if not df_filtrado.empty:
            df_vehiculos = df_filtrado.groupby(['Vehículo', 'Severidad']).size().reset_index(name='Cantidad')
            total_por_vehiculo = df_vehiculos.groupby('Vehículo')['Cantidad'].sum().sort_values(ascending=True)
            orden_vehiculos = total_por_vehiculo.index.tolist()

            fig_bar = px.bar(
                df_vehiculos, x='Cantidad', y='Vehículo', color='Severidad', orientation='h',
                text='Cantidad', category_orders={'Vehículo': orden_vehiculos},
                color_discrete_map=mapa_colores_severidad,
                labels={"Cantidad": "Siniestros", "Vehículo": ""}
            )
            fig_bar.update_traces(hovertemplate="<b>%{y}</b><br>%{legendgroup}: %{x}<extra></extra>")
            st.plotly_chart(fig_bar, use_container_width=True)

    col_uso, col_pie = st.columns([2, 1])
    with col_uso:
        st.subheader("📦 Modalidad")
        if not df_filtrado.empty:
            df_uso = df_filtrado['Modalidad'].value_counts().reset_index()
            df_uso.columns = ['Modalidad', 'Cantidad']
            fig_tree = px.treemap(df_uso, path=['Modalidad'], values='Cantidad', color='Cantidad', color_continuous_scale='Blues')
            fig_tree.update_traces(hovertemplate="<b>%{label}</b><br>Cantidad: %{value}<extra></extra>")
            st.plotly_chart(fig_tree, use_container_width=True)

    with col_pie:
        st.subheader("👥 Género")
        if not df_filtrado.empty and 'SEXO' in df_filtrado.columns:
            conteo_sexo = df_filtrado['SEXO'].value_counts().reset_index()
            conteo_sexo.columns = ['Género', 'Cantidad']
            fig_pie = px.pie(conteo_sexo, values='Cantidad', names='Género', hole=0.4, color_discrete_sequence=['#4682B4', '#CD5C5C'])
            st.plotly_chart(fig_pie, use_container_width=True)

    st.subheader("⏰ Frecuencia por Hora")
    if not df_filtrado.empty:
        por_hora = df_filtrado.groupby('Hora').size().reset_index(name='Cantidad')
        fig_hora = px.bar(por_hora, x='Hora', y='Cantidad')
        fig_hora.update_traces(marker_color='#00CC96')
        fig_hora.update_xaxes(tickmode='linear', dtick=1)
        st.plotly_chart(fig_hora, use_container_width=True)


# ==============================================================================
# VISTA 2: SIMULADOR DE RIESGO
# ==============================================================================
elif vista_actual == "🔮 Simulador de Riesgo (IA)":

    st.title("🔮 Simulador de Riesgo (Motor XGBoost)")
    st.markdown("El modelo evalúa las condiciones y predice la severidad usando Inteligencia Artificial.")
    st.markdown("---")

    col_inputs, col_results = st.columns([1, 1.8], gap="medium")

    with col_inputs:
        st.subheader("📝 Datos del Evento")
        
        # --- UBICACIÓN DINÁMICA ---
        st.markdown("##### 📍 ¿Dónde ocurre?")
        
        lista_dptos = sorted(df['Dpto_Clean'].unique()) if not df.empty else ["LIMA"]
        dpto_sel = st.selectbox("Departamento", lista_dptos)
        
        df_provs = df[df['Dpto_Clean'] == dpto_sel] if not df.empty else df
        lista_prov = sorted(df_provs['Prov_Clean'].unique()) if not df_provs.empty else ["LIMA"]
        prov_sel = st.selectbox("Provincia", lista_prov)
        
        df_dist = df_provs[df_provs['Prov_Clean'] == prov_sel] if not df_provs.empty else df
        lista_dist = sorted(df_dist['Dist_Clean'].unique()) if not df_dist.empty else ["LIMA"]
        dist_sel = st.selectbox("Distrito", lista_dist)
        
        st.markdown("##### 🚘 Detalles")
        
        lista_vehiculos = encoders['VEHÍCULO'].classes_
        vehiculo_input = st.selectbox("Tipo de Vehículo", lista_vehiculos)
        
        c1, c2 = st.columns(2)
        with c1:
            hora_actual = datetime.now().hour
            hora_input = st.number_input("Hora (0-23h)", 0, 23, hora_actual)
        with c2:
            edad_input = st.number_input("Edad Conductor", 15, 90, 30)
        
        sexo_input = st.selectbox("Sexo", encoders['SEXO'].classes_)
        tipo_persona_input = st.selectbox("Rol", encoders['TIPO PERSONA'].classes_)
        zona_input = st.selectbox("Zona", encoders['ZONA'].classes_)
        
        st.markdown("---")
        umbral_riesgo = st.slider("Sensibilidad (Umbral)", 0.1, 0.9, 0.40, help="Nivel para marcar alerta fatal")

        calcular_click = st.button("🎲 Calcular con IA", use_container_width=True, type="primary")

    with col_results:
        if calcular_click:
            if model is None:
                st.error("Error: El modelo IA no está cargado.")
            else:
                try:
                    # 1. PREPARAR DATOS
                    input_data = pd.DataFrame({
                        'EDAD': [edad_input],
                        'SEXO': [encoders['SEXO'].transform([sexo_input])[0]],
                        'TIPO PERSONA': [encoders['TIPO PERSONA'].transform([tipo_persona_input])[0]],
                        'VEHÍCULO': [encoders['VEHÍCULO'].transform([vehiculo_input])[0]],
                        'MODALIDAD DE TRANSPORTE': [7], # Default Particular
                        'DEPARTAMENTO': [10], # Default
                        'ZONA': [encoders['ZONA'].transform([zona_input])[0]],
                        'TIPO DE VÍA': [4], # Default
                        'MES': [datetime.now().month], 
                        'HORA_ENTERA': [hora_input],
                        'ES_FIN_SEMANA': [1 if hora_input > 18 else 0], 
                        'RANGO_HORARIO': [3 if hora_input > 18 else 1],
                        'CONDICIÓN CLIMÁTICA': [0],
                        'SUPERFICIE DE CALZADA': [2]
                    })

                    # 2. PREDICCIÓN
                    probs = model.predict_proba(input_data)[0]
                    prob_fallecido = probs[0]
                    prob_ileso = probs[1]
                    prob_lesionado = probs[2]

                    # 3. RESULTADOS
                    riesgo_grave_pct = (prob_fallecido + prob_lesionado) * 100
                    
                    if prob_fallecido > umbral_riesgo:
                        titulo = "ALTO RIESGO (FALLECIDO)"
                        color = "red"
                    elif prob_lesionado > prob_ileso:
                        titulo = "RIESGO MEDIO (LESIONADO)"
                        color = "orange"
                    else:
                        titulo = "RIESGO BAJO (ILESO)"
                        color = "green"

                    st.success(f"Analizando: **{dist_sel}, {prov_sel}** -> Resultado: **{titulo}**")

                    # 4. VELOCÍMETRO
                    fig_gauge = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = riesgo_grave_pct,
                        number = {'suffix': "%", 'font': {'size': 35}},
                        title = {'text': "<b>PROBABILIDAD DE GRAVEDAD</b><br><span style='font-size:0.8em;color:gray'>(Lesión o Muerte)</span>"},
                        gauge = {
                            'axis': {'range': [None, 100]},
                            'bar': {'color': "rgba(0,0,0,0)"}, 
                            'steps': [
                                {'range': [0, 40], 'color': "#00cc96"},
                                {'range': [40, 70], 'color': "#ffa15a"},
                                {'range': [70, 100], 'color': "#ef553b"}
                            ],
                            'threshold': {
                                'line': {'color': "black", 'width': 4}, 'thickness': 0.75, 'value': riesgo_grave_pct
                            }
                        }
                    ))
                    fig_gauge.update_layout(height=240, margin=dict(l=40, r=40, t=40, b=10))
                    st.plotly_chart(fig_gauge, use_container_width=True)

                    # 5. BARRAS
                    data_probs = pd.DataFrame({
                        "Estado": ["FALLECIDO (0)", "ILESO (1)", "LESIONADO (2)"],
                        "Probabilidad": [prob_fallecido, prob_ileso, prob_lesionado]
                    })
                    fig_probs = px.bar(
                        data_probs, x="Probabilidad", y="Estado", orientation='h', text_auto='.1%',
                        color="Estado", color_discrete_map={"FALLECIDO (0)": "red", "LESIONADO (2)": "orange", "ILESO (1)": "green"}
                    )
                    fig_probs.update_layout(height=180, showlegend=False, margin=dict(t=0,b=0), xaxis=dict(showticklabels=False), yaxis=dict(title=None))
                    st.plotly_chart(fig_probs, use_container_width=True)

                except Exception as e:
                    st.error(f"Error en predicción: {e}")

        else:
             st.info("👈 Ingresa datos y presiona 'Calcular con IA'.")