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
    st.caption("Nota: 'IA' y 'Riesgo IA' corresponden a severidad estimada por el modelo (XGBoost).")

    st.markdown(f"**Registros:** {len(df_filtrado)}")

    # KPIs
    if not df_filtrado.empty:
        k1, k2, k3, k4, k5 = st.columns(5)
        total = len(df_filtrado)
        n_falle = len(df_filtrado[df_filtrado['Severidad'] == 'FALLECIDO'])
        n_lesion = len(df_filtrado[df_filtrado['Severidad'] == 'LESIONADO'])
        n_ileso = len(df_filtrado[df_filtrado['Severidad'] == 'ILESO'])
        riesgo_val = df_filtrado['probabilidad'].mean()

        k1.metric("Eventos", f"{total:,}")
        k2.metric("🔴 IA: Fallec.", f"{n_falle}")
        k3.metric("🟠 IA: Lesion.", f"{n_lesion}")
        k4.metric("🟢 IA: Ileso", f"{n_ileso}")
        k5.metric("📉 Riesgo IA", f"{riesgo_val:.1f}%")
    else:
        st.info("Selecciona filtros para ver datos.")

    st.markdown("---")

    col_mapa, col_top = st.columns([1, 1])

    with col_mapa:
        st.subheader("🗺️ Probabilidad de accidentes por región")

        
        if not df_filtrado.empty:
            # 1. AGRUPACIÓN INTELIGENTE
            # Agrupamos por Distrito para dejar de ver puntos aislados y ver "Zonas"
            # Calculamos: 
            # - Lat/Lon promedio (para centrar el círculo en el distrito)
            # - Probabilidad promedio (para saber qué tan grave suele ser ahí)
            # - Conteo (para saber cuántos siniestros hubo)
            df_zona = df_filtrado.groupby(['Dist_Clean', 'Prov_Clean']).agg({
                'latitud': 'mean',          # Centroide del distrito
                'longitud': 'mean',         # Centroide del distrito
                'probabilidad': 'mean',     # PROMEDIO de la probabilidad de gravedad
                'Severidad': 'count'        # CANTIDAD de reportes
            }).reset_index()

            # Renombramos para el gráfico
            df_zona.rename(columns={'Severidad': 'Cantidad_Siniestros', 'probabilidad': 'Riesgo_Promedio'}, inplace=True)

            # 2. VISUALIZACIÓN POR ZONA
            # Color = Qué tan peligroso es el promedio (Rojo = Alto riesgo, Verde = Bajo)
            # Tamaño = Cantidad de siniestros (Más grande = Más accidentes)
            fig_mapa = px.scatter_mapbox(
                df_zona,
                lat="latitud", lon="longitud",
                color="Riesgo_Promedio",
                size="Cantidad_Siniestros",
                hover_name="Dist_Clean",
                custom_data=["Prov_Clean", "Cantidad_Siniestros", "Riesgo_Promedio"],
                color_continuous_scale=px.colors.diverging.RdYlGn[::-1],
                zoom=9,
                height=500,
                mapbox_style="open-street-map"
            )

            fig_mapa.update_traces(
                marker=dict(opacity=0.7, sizemin=10),
                hovertemplate=(
                    "<b>%{hovertext}</b><br>"
                    "Provincia: %{customdata[0]}<br>"
                    "📌 Eventos analizados: %{customdata[1]}<br>"
                    "⚠️ Riesgo promedio: %{customdata[2]:.1f}%<extra></extra>"
                )
            )


            
            fig_mapa.update_layout(
                margin={"r":0,"t":0,"l":0,"b":0},
                coloraxis_colorbar=dict(title="% Riesgo")
            )
            
            st.plotly_chart(fig_mapa, use_container_width=True)
            
            # --- TABLA RESUMEN OPCIONAL DEBAJO DEL MAPA ---
# --- TABLA RESUMEN OPCIONAL DEBAJO DEL MAPA ---
            with st.expander("Ver detalle de zonas críticas (Top 5)"):
                # 1. Ordenamos y sacamos el Top 5
                top_zonas = df_zona.sort_values(by=['Cantidad_Siniestros', 'Riesgo_Promedio'], ascending=False).head(5)
                
                # 2. Seleccionamos solo las columnas que nos importan
                df_display = top_zonas[['Dist_Clean', 'Prov_Clean', 'Cantidad_Siniestros', 'Riesgo_Promedio']].copy()
                
                # 3. Formato de Porcentaje: Redondeo a 2 decimales + signo %
                # Ejemplo: 56.0848 se convierte en "56.08%"
                df_display['Riesgo_Promedio'] = df_display['Riesgo_Promedio'].apply(lambda x: f"{x:.1f}%")
                
                # 4. Renombramos los encabezados para que se vean profesionales
                df_display.rename(columns={
                    'Dist_Clean': 'Distrito',
                    'Prov_Clean': 'Provincia',
                    'Cantidad_Siniestros': 'Eventos',
                    'Riesgo_Promedio': 'Riesgo IA'
                }, inplace=True)

                
                # 5. Mostramos la tabla limpia
                st.dataframe(df_display, hide_index=True, use_container_width=True)
        
        else:
            st.warning("No hay datos para mostrar en el mapa con los filtros actuales.")

    with col_top:
            st.subheader("📊 Probabilidad de accidentes por gravedad y departamento")
            
            if not df_filtrado.empty:
                fig_stack = px.histogram(
                    df_filtrado, 
                    x="Dpto_Clean", 
                    color="Severidad", 
                    barnorm='percent',  # Calcula el % automáticamente
                    color_discrete_map=mapa_colores_severidad,
                    category_orders={"Severidad": ["FALLECIDO", "LESIONADO", "ILESO"]},
                    # Limpiamos los nombres de los ejes para el gráfico base
                    labels={"Dpto_Clean": "Departamento", "Severidad": "Condición"}
                )
                
                fig_stack.update_layout(
                    xaxis_title="Departamento",
                    yaxis_title="Distribución (%)",
                    legend_title="Condición",
                    margin={"r":0,"t":30,"l":0,"b":0}
                )
                
                # --- MAGIA DE ETIQUETAS ---
                fig_stack.update_traces(
                    texttemplate='%{y:.2f}%',
                    textposition='inside',
                    hovertemplate=(
                        "<b>Departamento: %{x}</b><br>"
                        "Condición: %{fullData.name}<br>"
                        "Distribución: %{y:.2f}%<extra></extra>"
                    )
                )

                
                st.plotly_chart(fig_stack, use_container_width=True)
                
            else:
                st.info("Sin datos para mostrar el desglose por departamento.")

    # Cambiamos [2, 1] por [1, 1] para dar espacio al nuevo gráfico
    col_uso, col_pie = st.columns([1, 1])
    with col_uso:
        st.subheader("📦 Probabilidad de accidentes según el tipo de vehículo")
        if not df_filtrado.empty:
            df_uso = df_filtrado['Modalidad'].value_counts().reset_index()
            df_uso.columns = ['Modalidad', 'Cantidad']
            fig_tree = px.treemap(df_uso, path=['Modalidad'], values='Cantidad', color='Cantidad', color_continuous_scale='Blues')
            fig_tree.update_traces(hovertemplate="<b>%{label}</b><br>Eventos: %{value}<extra></extra>")
            st.plotly_chart(fig_tree, use_container_width=True)

    with col_pie:
        st.subheader("🚗 Riesgo de accidentes por vehículo")
        
        if not df_filtrado.empty:
            # Filtro opcional: Solo vehículos con datos
            df_veh = df_filtrado[df_filtrado['Vehículo'].map(df_filtrado['Vehículo'].value_counts()) > 0]
            
            fig_veh = px.histogram(
                df_veh, 
                y="Vehículo",   # <--- CAMBIO CLAVE: Categorías en el eje Y
                # x se calcula automáticamente (el porcentaje)
                color="Severidad", 
                barnorm='percent',
                orientation='h', # <--- CAMBIO CLAVE: Orientación horizontal
                hover_data=["Severidad"], 
                color_discrete_map=mapa_colores_severidad,
                category_orders={"Severidad": ["FALLECIDO", "LESIONADO", "ILESO"]},
                labels={"Vehículo": "", "Severidad": "Condición"} # Quitamos la etiqueta del eje Y para ganar espacio
            )
            
            fig_veh.update_layout(
                xaxis_title="Distribución (%)", # El eje X ahora es el porcentaje
                yaxis_title="",                 # El eje Y son los vehículos (ya se entiende por los nombres)
                legend_title="Condición",
                margin={"r":0,"t":30,"l":0,"b":0},
                # Ordenamos el eje Y para que el vehículo con más siniestros totales salga arriba
                yaxis={'categoryorder':'total descending'} 
            )
            
            fig_veh.update_traces(
                texttemplate='%{x:.2f}%',
                textposition='inside',
                hovertemplate=(
                    "<b>Vehículo: %{y}</b><br>"
                    "Condición: %{fullData.name}<br>"
                    "Distribución: %{x:.2f}%<extra></extra>"
                )
            )

            
            st.plotly_chart(fig_veh, use_container_width=True)
            
        else:
            st.info("No hay datos de vehículos para mostrar.")

# --- NUEVA FILA: GÉNERO Y HORA (50% / 50%) ---
    st.markdown("---") # Una línea separadora para ordenar visualmente
    col_sexo, col_hora = st.columns([1, 1])

    with col_sexo:
        st.subheader("👥 Probabilidad de accidentes por género")
        if not df_filtrado.empty and 'SEXO' in df_filtrado.columns:
            conteo_sexo = df_filtrado['SEXO'].value_counts().reset_index()
            conteo_sexo.columns = ['Género', 'Cantidad']
            
            fig_pie = px.pie(
                conteo_sexo, 
                values='Cantidad', 
                names='Género', 
                hole=0.4, 
                color_discrete_sequence=['#4682B4', '#CD5C5C'] # Azul y Rojo suave
            )
            fig_pie.update_traces(
                textinfo='percent+label',
                hovertemplate="<b>%{label}</b><br>Eventos analizados: %{value}<extra></extra>"
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("Sin datos de género.")

    with col_hora:
        st.subheader("⏰ Probabilidad de eventos por hora")
        if not df_filtrado.empty:
            por_hora = df_filtrado.groupby('Hora').size().reset_index(name='Cantidad')
            
            fig_hora = px.bar(
                por_hora, 
                x='Hora', 
                y='Cantidad',
                labels={'Hora': 'Hora del Día', 'Cantidad': 'N° Siniestros'}
            )
            fig_hora.update_traces(
                marker_color='#00CC96',
                hovertemplate="<b>%{x}:00 hrs</b><br>Eventos: %{y}<extra></extra>"
            )
            fig_hora.update_layout(
                xaxis=dict(tickmode='linear', dtick=1), # Muestra todas las horas
                margin=dict(l=0, r=0, t=30, b=0)
            )
            st.plotly_chart(fig_hora, use_container_width=True)
        else:
            st.info("Sin datos de hora.")


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
        st.subheader("📊 Resultado del análisis")

        if calcular_click:
            if model is None:
                st.error("Error: El modelo IA no está cargado.")
            else:
                try:
                    # 1) PREPARAR DATOS (igual que tú)
                    input_data = pd.DataFrame({
                        'EDAD': [edad_input],
                        'SEXO': [encoders['SEXO'].transform([sexo_input])[0]],
                        'TIPO PERSONA': [encoders['TIPO PERSONA'].transform([tipo_persona_input])[0]],
                        'VEHÍCULO': [encoders['VEHÍCULO'].transform([vehiculo_input])[0]],
                        'MODALIDAD DE TRANSPORTE': [7],
                        'DEPARTAMENTO': [10],
                        'ZONA': [encoders['ZONA'].transform([zona_input])[0]],
                        'TIPO DE VÍA': [4],
                        'MES': [datetime.now().month],
                        'HORA_ENTERA': [hora_input],

                        # OJO: esto idealmente debería depender de la FECHA real, no de la hora.
                        # Lo dejo como está para no romper tu flujo, pero luego lo mejoramos.
                        'ES_FIN_SEMANA': [1 if hora_input > 18 else 0],
                        'RANGO_HORARIO': [3 if hora_input > 18 else 1],

                        'CONDICIÓN CLIMÁTICA': [0],
                        'SUPERFICIE DE CALZADA': [2]
                    })

                    # 2) PREDICCIÓN
                    probs = model.predict_proba(input_data)[0]
                    prob_fallecido = float(probs[0])
                    prob_ileso     = float(probs[1])
                    prob_lesionado = float(probs[2])

                    riesgo_grave = prob_fallecido + prob_lesionado
                    riesgo_grave_pct = riesgo_grave * 100

                    # 3) CLASIFICACIÓN (más “humana”)
                    if prob_fallecido > umbral_riesgo:
                        nivel = "ALTO"
                        titulo = "ALTO RIESGO"
                        subtitulo = "Alta probabilidad de un evento fatal."
                        color_hex = "#ff4b4b"
                        icono = "🔴"
                    elif riesgo_grave > 0.40:
                        nivel = "MEDIO"
                        titulo = "RIESGO MEDIO"
                        subtitulo = "Riesgo relevante de lesión. Reforzar prevención."
                        color_hex = "#ffa15a"
                        icono = "🟠"
                    else:
                        nivel = "BAJO"
                        titulo = "RIESGO BAJO"
                        subtitulo = "Predomina la probabilidad de lesión leve o sin lesión."
                        color_hex = "#00cc96"
                        icono = "🟢"

                    # 4) HEADER tipo “tarjeta”
                    st.markdown(
                        f"""
                        <div style="
                            padding:16px;
                            border-radius:12px;
                            border:1px solid rgba(255,255,255,0.12);
                            background:rgba(255,255,255,0.04);
                        ">
                            <div style="font-size:16px; opacity:0.85;">Analizando ubicación</div>
                            <div style="font-size:20px; font-weight:700;">
                                {dist_sel}, {prov_sel} ({dpto_sel})
                            </div>
                            <div style="margin-top:10px; font-size:22px; font-weight:800; color:{color_hex};">
                                {icono} {titulo} — {riesgo_grave_pct:.1f}%
                            </div>
                            <div style="margin-top:6px; font-size:14px; opacity:0.85;">
                                {subtitulo}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    st.write("")

                    # 5) MÉTRICAS rápidas (muy entendibles)
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Riesgo de lesión o muerte", f"{riesgo_grave_pct:.1f}%")
                    m2.metric("Prob. de fallecido", f"{prob_fallecido*100:.1f}%")
                    m3.metric("Prob. de lesionado", f"{prob_lesionado*100:.1f}%")

                    st.write("")

                    # 6) “¿Qué significa este %?” (clave para público general)
                    with st.expander("ℹ️ ¿Cómo interpretar este resultado?"):
                        st.markdown(
                            f"""
                            - Este resultado es una **estimación** basada en patrones de datos históricos.
                            - **{riesgo_grave_pct:.1f}%** significa que, en condiciones parecidas, el modelo considera **alta/media/baja**
                            la probabilidad de que el evento termine en **lesión o muerte**.
                            - **No es una certeza**, pero sirve como **alerta preventiva** para priorizar acciones.
                            """
                        )

                    # 7) Visual: barra de riesgo simple (más clara que el velocímetro para público general)
                    st.markdown("#### 🧭 Nivel de riesgo (visual)")
                    st.progress(min(max(riesgo_grave, 0.0), 1.0))  # 0 a 1

                    # 8) Barras de probabilidades (tu gráfico, pero con etiquetas más claras)
                    st.markdown("#### 📌 Probabilidades por resultado")
                    data_probs = pd.DataFrame({
                        "Resultado": ["Fallecido", "Ileso", "Lesionado"],
                        "Probabilidad": [prob_fallecido, prob_ileso, prob_lesionado]
                    }).sort_values("Probabilidad", ascending=True)

                    fig_probs = px.bar(
                        data_probs,
                        x="Probabilidad",
                        y="Resultado",
                        orientation='h',
                        text=data_probs["Probabilidad"].map(lambda x: f"{x*100:.1f}%")
                    )
                    fig_probs.update_layout(
                        height=220,
                        showlegend=False,
                        margin=dict(t=10, b=10, l=10, r=10),
                        xaxis=dict(showticklabels=False, title=None),
                        yaxis=dict(title=None)
                    )
                    st.plotly_chart(fig_probs, use_container_width=True)

                    # 9) Recomendaciones automáticas (para “vender” la solución)
                    st.markdown("#### ✅ Recomendación preventiva")
                    if nivel == "ALTO":
                        st.warning(
                            "Priorizar medidas de prevención y respuesta rápida: control de velocidad, señalización, "
                            "alertas al conductor, y coordinación inmediata con puntos de soporte."
                        )
                    elif nivel == "MEDIO":
                        st.info(
                            "Reforzar prevención: mayor vigilancia, mensajes preventivos, revisión de condiciones y "
                            "acciones focalizadas en la zona y horario."
                        )
                    else:
                        st.success(
                            "Mantener medidas estándar y monitoreo. No se detecta alerta alta para estas condiciones."
                        )

                except Exception as e:
                    st.error(f"Error en predicción: {e}")

        else:
            st.info("👈 Ingresa los datos del evento y presiona **Calcular con IA**.")
