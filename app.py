import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
import pandas as pd
from lifelines import KaplanMeierFitter
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO
import base64
import numpy as np

# Inicializa la aplicación Dash
app = dash.Dash(__name__)
server = app.server

# Diseño de la interfaz de la aplicación
app.layout = html.Div(children=[
    html.H1(children='Proyector de Aprobaciones para Estudios Clínicos'),
    html.Hr(),

    html.Div([
        html.H3("1. Sube tu archivo CSV"),
        html.P("El archivo debe contener las columnas: 'Milestone', 'Dias_Hasta_Aprobacion', 'Estado_Aprobacion' (1=aprobado, 0=no aprobado)."),
        dcc.Upload(
            id='upload-data',
            children=html.Div([
                'Arrastra o ',
                html.A('selecciona un archivo')
            ]),
            style={
                'width': '100%',
                'height': '60px',
                'lineHeight': '60px',
                'borderWidth': '1px',
                'borderStyle': 'dashed',
                'borderRadius': '5px',
                'textAlign': 'center',
                'margin': '10px'
            },
            multiple=False
        ),
    ]),

    html.Hr(),

    html.Div([
        html.H3("2. Selecciona un Milestone"),
        dcc.Dropdown(
            id='milestone-dropdown',
            placeholder="Selecciona un Milestone (IRB, MOH, etc.)",
            multi=False
        ),
    ]),

    html.Hr(),
    
    html.Div([
        html.H3("3. Resultados del Análisis"),
        dcc.Loading(
            id="loading-1",
            type="default",
            children=[
                dcc.Graph(id='survival-graph'),
                html.H4("Probabilidades de Aprobación por Porcentaje:", style={'marginTop': 20}),
                html.Div(id='probability-text', style={'fontSize': 16, 'marginTop': 10}),
                html.Div(id='debug-info', style={'fontSize': 12, 'color': 'gray', 'marginTop': 10})
            ]
        )
    ]),

])

# Callback para procesar el archivo subido y actualizar el dropdown
@app.callback(
    Output('milestone-dropdown', 'options'),
    Output('milestone-dropdown', 'value'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def update_dropdown(contents, filename):
    if contents is not None:
        try:
            # Decodifica el contenido base64
            content_type, content_string = contents.split(',')
            decoded = base64.b64decode(content_string)
            
            # Lee el contenido del CSV
            df = pd.read_csv(StringIO(decoded.decode('utf-8')))
            
            # Verifica que las columnas necesarias existan
            required_columns = ['Milestone', 'Dias_Hasta_Aprobacion', 'Estado_Aprobacion']
            if not all(col in df.columns for col in required_columns):
                return [{'label': 'Error: Faltan columnas requeridas', 'value': 'error'}], None
            
            # Obtiene los valores únicos de la columna 'Milestone'
            milestones = df['Milestone'].unique()
            options = [{'label': i, 'value': i} for i in milestones]
            
            # Devuelve las opciones y el primer valor por defecto
            return options, milestones[0] if len(milestones) > 0 else None
        except Exception as e:
            print(f"Error: {e}")  # Para debugging
            return [{'label': f'Error al cargar el archivo: {str(e)}', 'value': 'error'}], None
    return [], None

# Callback para generar el gráfico y el texto de probabilidad
@app.callback(
    Output('survival-graph', 'figure'),
    Output('probability-text', 'children'),
    Output('debug-info', 'children'),
    Input('milestone-dropdown', 'value'),
    State('upload-data', 'contents')
)
def update_output(selected_milestone, contents):

    if not selected_milestone or not contents or selected_milestone == 'error':
        return go.Figure(), "Por favor, sube un archivo y selecciona un Milestone para ver los resultados.", ""

    try:
        # Decodifica el contenido base64
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        df = pd.read_csv(StringIO(decoded.decode('utf-8')))
        
        # Filtra el DataFrame por el Milestone seleccionado
        df_filtered = df[df['Milestone'] == selected_milestone].copy()
        
        # Verifica que haya datos después del filtrado
        if df_filtered.empty:
            return go.Figure(), f"No hay datos para el Milestone: {selected_milestone}", ""

        # Información de debug
        debug_info = f"Datos: {len(df_filtered)} registros. Estado_Aprobacion: {df_filtered['Estado_Aprobacion'].value_counts().to_dict()}"

        # Preprocesamiento de datos
        df_filtered = df_filtered.dropna(subset=['Dias_Hasta_Aprobacion', 'Estado_Aprobacion'])
        df_filtered['Estado_Aprobacion'] = pd.to_numeric(df_filtered['Estado_Aprobacion'], errors='coerce').fillna(0).astype(int)
        
        # Verifica que haya al menos algunos eventos (aprobaciones)
        if df_filtered['Estado_Aprobacion'].sum() == 0:
            return go.Figure(), f"No hay eventos de aprobación para {selected_milestone}", debug_info

        # Nombres de columnas para el análisis
        T = df_filtered['Dias_Hasta_Aprobacion']
        E = df_filtered['Estado_Aprobacion']

        # Crea el modelo de Kaplan-Meier
        kmf = KaplanMeierFitter()
        kmf.fit(T, event_observed=E)

        # Verifica que el survival function tenga datos
        if kmf.survival_function_.empty:
            return go.Figure(), f"No se pudo generar el modelo para {selected_milestone}", debug_info

        # Obtiene el nombre correcto de la columna (puede variar)
        survival_column = kmf.survival_function_.columns[0]
        
        # Genera el gráfico
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

        # Calcula las probabilidades en saltos de 10%
        prob_text = []
        percentiles = list(range(10, 101, 10))  # [10, 20, 30, ..., 100]
        
        for percentile in percentiles:
            try:
                # Calcula la probabilidad de NO aprobación objetivo
                target_survival_prob = (100 - percentile) / 100.0
                
                # Encuentra el tiempo donde la probabilidad de NO aprobación es <= al objetivo
                survival_probs = kmf.survival_function_[survival_column]
                times = kmf.survival_function_.index
                
                # Encuentra el primer tiempo donde la probabilidad es <= al objetivo
                mask = survival_probs <= target_survival_prob
                if mask.any():
                    # Toma el primer tiempo que cumple la condición
                    time_at_percentile = times[mask].min()
                    prob_text.append(
                        html.P(f"📊 {percentile}% de probabilidad de aprobación en: {int(time_at_percentile)} días")
                    )
                else:
                    # Si nunca se alcanza esa probabilidad
                    max_time = times.max()
                    current_prob = (1 - survival_probs.iloc[-1]) * 100
                    prob_text.append(
                        html.P(f"⚠️  {percentile}% de probabilidad: No se alcanza (máximo {current_prob:.1f}% en {int(max_time)} días)", 
                               style={'color': 'orange'})
                    )
                    
            except Exception as e:
                prob_text.append(
                    html.P(f"❌ Error calculando {percentile}%: {str(e)}", style={'color': 'red'})
                )

        return fig, prob_text, debug_info
        
    except Exception as e:
        print(f"Error en update_output: {e}")  # Para debugging
        return go.Figure(), f"Error al procesar los datos: {str(e)}", f"Error: {str(e)}"

# Ejecuta la aplicación
if __name__ == '__main__':
    app.run(debug=True)
