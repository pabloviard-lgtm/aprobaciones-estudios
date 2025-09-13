import streamlit as st
import pandas as pd
from lifelines import KaplanMeierFitter
import plotly.express as px
import numpy as np

# Configuración de la página
st.set_page_config(page_title="Proyector de Aprobaciones", layout="wide")

# Título
st.title('Proyector de Aprobaciones para Estudios Clínicos')
st.markdown("---")

# Upload del archivo
st.header("1. Sube tu archivo CSV")
st.write("El archivo debe contener las columnas: 'Milestone', 'Dias_Hasta_Aprobacion', 'Estado_Aprobacion' (1=aprobado, 0=no aprobado).")

uploaded_file = st.file_uploader("Selecciona un archivo CSV", type="csv")

if uploaded_file is not None:
    try:
        # Leer el archivo
        df = pd.read_csv(uploaded_file)
        
        # Verificar columnas
        required_columns = ['Milestone', 'Dias_Hasta_Aprobacion', 'Estado_Aprobacion']
        if not all(col in df.columns for col in required_columns):
            st.error("Error: El archivo debe contener las columnas: 'Milestone', 'Dias_Hasta_Aprobacion', 'Estado_Aprobacion'")
        else:
            st.success("✅ Archivo cargado correctamente")
            
            # Seleccionar milestone
            st.header("2. Selecciona un Milestone")
            milestones = df['Milestone'].unique()
            selected_milestone = st.selectbox("Elige un Milestone:", options=milestones)
            
            if selected_milestone:
                st.header("3. Resultados del Análisis")
                
                # Filtrar datos
                df_filtered = df[df['Milestone'] == selected_milestone].copy()
                df_filtered = df_filtered.dropna(subset=['Dias_Hasta_Aprobacion', 'Estado_Aprobacion'])
                df_filtered['Estado_Aprobacion'] = pd.to_numeric(df_filtered['Estado_Aprobacion'], errors='coerce').fillna(0).astype(int)
                
                if df_filtered.empty:
                    st.warning(f"No hay datos para el Milestone: {selected_milestone}")
                elif df_filtered['Estado_Aprobacion'].sum() == 0:
                    st.warning(f"No hay eventos de aprobación para {selected_milestone}")
                else:
                    # Crear modelo Kaplan-Meier
                    kmf = KaplanMeierFitter()
                    T = df_filtered['Dias_Hasta_Aprobacion']
                    E = df_filtered['Estado_Aprobacion']
                    kmf.fit(T, event_observed=E)
                    
                    # Gráfico
                    survival_column = kmf.survival_function_.columns[0]
                    fig = px.line(
                        x=kmf.survival_function_.index,
                        y=kmf.survival_function_[survival_column],
                        labels={'x': 'Días desde el envío', 'y': 'Probabilidad de NO Aprobación'},
                        title=f'Curva de Supervivencia para: {selected_milestone} (n={len(df_filtered)})'
                    )
                    fig.update_layout(
                        xaxis_title="Días desde el envío",
                        yaxis_title="Probabilidad de NO Aprobación",
                        template='plotly_white'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Probabilidades en saltos de 10%
                    st.subheader("Probabilidades de Aprobación por Porcentaje:")
                    
                    survival_probs = kmf.survival_function_[survival_column]
                    times = kmf.survival_function_.index
                    
                    for percentile in range(10, 101, 10):
                        target_survival_prob = (100 - percentile) / 100.0
                        mask = survival_probs <= target_survival_prob
                        
                        if mask.any():
                            time_at_percentile = times[mask].min()
                            st.write(f"📊 **{percentile}%** de probabilidad de aprobación en: **{int(time_at_percentile)} días**")
                        else:
                            max_time = times.max()
                            current_prob = (1 - survival_probs.iloc[-1]) * 100
                            st.write(f"⚠️ **{percentile}%** de probabilidad: No se alcanza (máximo {current_prob:.1f}% en {int(max_time)} días)")
                    
                    # Info adicional
                    with st.expander("Ver información detallada de los datos"):
                        st.write(f"**Total de registros:** {len(df_filtered)}")
                        st.write(f"**Distribución de Estado_Aprobacion:**")
                        st.write(df_filtered['Estado_Aprobacion'].value_counts())
                        st.write("**Primeras filas de datos:**")
                        st.dataframe(df_filtered.head())
                        
    except Exception as e:
        st.error(f"Error al procesar el archivo: {str(e)}")
else:
    st.info("Por favor, sube un archivo CSV para comenzar")

# Información de ayuda
with st.sidebar:
    st.header("ℹ️ Instrucciones")
    st.write("""
    1. Sube un archivo CSV con las columnas requeridas
    2. Selecciona el Milestone a analizar
    3. Revisa los resultados y probabilidades
    """)
    
    st.header("📋 Ejemplo de CSV")
    st.write("""
    Milestone,Dias_Hasta_Aprobacion,Estado_Aprobacion
    IRB,25,1
    IRB,30,1
    IRB,45,1
    MOH,40,1
    MOH,55,0
    """)
