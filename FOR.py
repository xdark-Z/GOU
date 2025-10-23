import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
from scipy import stats
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import io
import warnings
import torch

warnings.filterwarnings('ignore') # Ignorar advertencias de Python
#################################CONFIGURACIÓN INICIAL##############################################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Seleccionar dispositivo para cómputo (CUDA o CPU)

#################################FUNCIONES CACHEADAS Y DE VISUALIZACIÓN##############################################

# Cachear la función para optimizar rendimiento 

@st.cache_data 
def obtener_recursos_por_nivel(_df, nivel):
    """Función cacheada para obtener recursos únicos para un nivel."""
    if _df is None or nivel is None: # Verificar si el DataFrame o nivel son válidos
        return [] # Retornar lista vacía si no hay datos
    recursos = _df[_df['Nivel'] == nivel]['Recurso'].dropna().unique() # Filtrar y obtener recursos únicos
    return [r for r in recursos if r and pd.notna(r)] # Limpiar recursos nulos o vacíos

@st.cache_data  
def generar_visualizacion_frentes(_df_frentes, _restricciones, radio_restriccion, nivel_seleccionado):
    """Genera la figura de visualización de frentes. Cacheada para no recalcular."""
    frentes_para_viz = _df_frentes[_df_frentes['Nivel'] == nivel_seleccionado] # Filtrar frentes por el nivel seleccionado
    fig = go.Figure() # Inicializar figura de Plotly
    for _, info_frente in frentes_para_viz.iterrows(): # Iterar sobre cada frente
        fig.add_trace(go.Scatter(x=[info_frente['Xi'], info_frente['Xf']], y=[info_frente['Yi'], info_frente['Yf']], # Dibujar línea del frente
                                 mode='lines+markers', name=f"Frente {info_frente['Frentes']}", # Modo línea y marcadores
                                 line=dict(width=4), marker=dict(size=10)))

    for tipo_restriccion, df in _restricciones.items(): # Iterar sobre los tipos de restricciones
        if df is not None: # Si el DataFrame de restricción existe
            for _, restriccion in df.iterrows(): # Iterar sobre cada restricción
                
                # Círculo con restricciones
                fig.add_shape(type="circle", xref="x", yref="y",
                              x0=restriccion['X'] - radio_restriccion, y0=restriccion['Y'] - radio_restriccion,
                              x1=restriccion['X'] + radio_restriccion, y1=restriccion['Y'] + radio_restriccion, # Definir el área circular
                              line_color="red", line_dash="dash", opacity=0.5)
                nombre_restriccion = f"{tipo_restriccion}: {restriccion.get('Nombre', 'Sin Nombre')}" # Preparar nombre
                
                # Punto central de la restricción
                fig.add_trace(go.Scatter(x=[restriccion['X']], y=[restriccion['Y']], # Marcar punto central
                                         mode='markers', name=nombre_restriccion,
                                         marker=dict(color='red', size=8, symbol='x')))

    fig.update_layout(title=f'Ubicación del Frente y Restricciones para Nivel {nivel_seleccionado}', # Configurar título del gráfico
                      xaxis_title='Coordenada X', yaxis_title='Coordenada Y', height=600, showlegend=True)
    return fig # Retornar figura

# PLOTLY GRÁFICOS + GANTT 
def graficar_distribuciones_actividad(df): 
    """Crea un box plot para visualizar la distribución de tiempos de cada actividad."""
    fig = px.box(df, x="Duracion", y="Actividad", color="Recurso", # Crear box plot de duración por actividad y recurso
                 title="Distribución de Tiempos de Actividad por Ciclo",
                 labels={"Duracion": "Duración (horas)", "Actividad": "Actividad"}, 
                 orientation='h')
    fig.update_traces(quartilemethod="linear") 
    fig.update_layout(template='plotly_white', yaxis={'categoryorder':'total ascending'}) # Ordenar eje Y por valor total
    return fig # Dar gráfica 

def graficar_gantt_probabilistica(df_gantt, df_demoras=None): 
    """
    Crea una carta Gantt probabilística. Acepta un dataframe adicional
    para visualizar las demoras por cambios de equipo o eventos en rojo.
    """
    if df_gantt.empty: #verificar si df está vacío 
        return go.Figure()

    dfs_a_mostrar = [df_gantt.assign(tipo='actividad')] # Asignar tipo de actividad 'actividad'
    if df_demoras is not None and not df_demoras.empty:
        dfs_a_mostrar.append(df_demoras.assign(tipo='demora')) # Asignar tipo demora

    df_combinado = pd.concat(dfs_a_mostrar, ignore_index=True) #Combinar actividades y demoras en un solo DF
    df_combinado = df_combinado.sort_values(by='Start_p50', ascending=False) #Ordenar por tiempo de inicio P50 (para el gráfico)
    etiquetas_eje_y = df_combinado['Actividad'] #Obtener etiquetas de actividades

    fig = go.Figure() #Inicializar figura

    df_actividad = df_combinado[df_combinado['tipo'] == 'actividad'] #Filtrar solo actividades

                                    # RANGO P10-P90
    fig.add_trace(go.Bar(
        y=df_actividad['Actividad'],
        x=df_actividad['Finish_p90'] - df_actividad['Start_p10'], #Calcular longitud del rango [p10-90]
        base=df_actividad['Start_p10'], # Base en el inicio P10
        orientation='h', name='Rango Actividad (P10-P90)',             
        marker=dict(color='rgba(255, 165, 0, 0.5)', line_width=0), #Color naranja/transparente
        hoverinfo='none' #No mostrar info al pasar el mouse
    ))

# DURACIÓN PROBABLE (P50) para actividades

    textos_p50_actividad = [] #Lista para textos P50
    posiciones_texto_p50_actividad = [] #Lista para posiciones
    for _, fila in df_actividad.iterrows(): #Determinar posición del texto P50 (dentro o fuera)
        if fila['Duration_p50'] > 1.5:
            textos_p50_actividad.append(f'{fila["Duration_p50"]:.2f}h') #Formato de texto para duración P50
            posiciones_texto_p50_actividad.append('inside') #Posición dentro de la barra
        else:
            textos_p50_actividad.append(f'{fila["Duration_p50"]:.2f}h')
            posiciones_texto_p50_actividad.append('outside') #Posición fuera de la barra

    fig.add_trace(go.Bar(
        y=df_actividad['Actividad'],
        x=df_actividad['Duration_p50'], #Duración P50
        base=df_actividad['Start_p50'], #Base en el inicio P50
        orientation='h', name='Duración Probable (P50)',
        marker=dict(color='rgba(26, 118, 255, 0.8)'), #Color azul
        text=textos_p50_actividad,
        textposition=posiciones_texto_p50_actividad,
        insidetextanchor='middle' #Anclaje del texto en el centro
    ))

# DURACIÓN PROBABLE (P50) para demoras

    if df_demoras is not None and not df_demoras.empty: #Incluir demoras si existen
        df_demora_a_graficar = df_combinado[df_combinado['tipo'] == 'demora'] #Filtrar solo demoras
        textos_demora = []
        posiciones_texto_demora = []
        for _, fila in df_demora_a_graficar.iterrows(): #Determinar posición del texto P50 (dentro o fuera)
            if fila['Duration_p50'] > 1.5:
                textos_demora.append(f'{fila["Duration_p50"]:.2f}h')
                posiciones_texto_demora.append('inside')
            else:
                textos_demora.append(f'{fila["Duration_p50"]:.2f}h')
                posiciones_texto_demora.append('outside')

        fig.add_trace(go.Bar(
            y=df_demora_a_graficar['Actividad'],
            x=df_demora_a_graficar['Duration_p50'], base=df_demora_a_graficar['Start_p50'],
            orientation='h', name='Demora por Evento (P50)', #Nombre adaptado para ser más general
            marker=dict(color='rgba(255, 0, 0, 0.8)'), #Color rojo para demoras
            text=textos_demora,
            textposition=posiciones_texto_demora,
            insidetextanchor='middle'
        ))

    total_p90 = df_combinado['Finish_p90'].max() #Máximo tiempo P90
    fig.update_layout( #Configurar layout del gráfico
        title='Carta Gantt Probabilística de un Ciclo (con Eventos/Demoras)',
        xaxis_title=f'Tiempo Acumulado (horas) - Fin Pesimista (P90): {total_p90:.2f}h', #Título eje X
        yaxis_title='Actividades y Demoras', #Título eje Y
        barmode='overlay', #Barras superpuestas
        template='plotly_white',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=len(etiquetas_eje_y) * 35 + 200, #Ajustar altura dinámicamente según actividades
        yaxis=dict(categoryorder='array', categoryarray=etiquetas_eje_y.tolist()), # Ordenar actividades según DF combinado
        xaxis=dict(range=[0, total_p90 * 1.05]) #Rango del eje X
    )
    return fig 

#################################CLASE PRINCIPAL DE SIMULACIÓN#########################################################

class SimuladorTunel: 
    def __init__(self):
        self.df_actividades = None #Df de actividades del ciclo
        self.df_recursos = None #Df de recursos y sus características
        self.df_frentes = None #Df de frentes/túneles a simular
        self.restricciones = { #Df de restricciones geográficas
            'frente_hundimiento': None,
            'front_index': None,
            'obras_civiles': None
        }
        self.actividades_definidas_usuario = [] #Lista de actividades añadidas por el usuario
        self.planes_obras_civiles = {} #Planes de obras civiles definidos
        self.condiciones_geologicas = {} #Condiciones geológicas definidas
        self.planes_restriccion = {} #Restricciones a inputar de forma manual
    
    def cargar_datos_desde_excel(self, archivo_excel):
        """Cargar datos desde archivo Excel (Actividades, Recursos, Frentes)"""
        try:
            self.df_actividades = pd.read_excel(archivo_excel, sheet_name=0) #Cargar actividades
            self.df_recursos = pd.read_excel(archivo_excel, sheet_name=1) #Cargar recursos
            self.df_frentes = pd.read_excel(archivo_excel, sheet_name=2) #Cargar frentes
            self.df_actividades = self.df_actividades.fillna('') #Rellenar NaN con string vacío
            self.procesar_datos_actividad() #Procesar datos de actividad
            self._procesar_datos_recursos() #Procesar datos de recursos
            return True
        except Exception as e:
            st.error(f"Error al cargar archivo de datos: {str(e)}") #Mostrar error si falla la carga
            return False

    def _procesar_datos_recursos(self): 
        """
        Procesa el DataFrame de recursos para interpretar turnos, cantidades y
        la columna 'Demora_Cambio'.
        """
        if self.df_recursos is None: #Verificar si hay datos
            return

        def parsear_rango(rango_str): 
            """Función interna para convertir 'inicio-fin' en (inicio, fin)."""
            if isinstance(rango_str, str) and '-' in rango_str:
                try:
                    partes = [float(p.strip().replace(',', '.')) for p in rango_str.split('-')] #Parsear rango y reemplazar coma por punto
                    return min(partes), max(partes)
                except (ValueError, TypeError):
                    return 0, 0
            try:
                val = float(str(rango_str).strip().replace(',', '.'))
                return val, val
            except (ValueError, TypeError):
                return 0, 0

        #Procesamiento de cambio de turno
        if 'Turno_Cambio' in self.df_recursos.columns: 
            self.df_recursos[['turno_inicio', 'turno_fin']] = self.df_recursos['Turno_Cambio'].apply(lambda x: pd.Series(parsear_rango(x)))
        else: # Si no existe, inicializar a 0
            self.df_recursos['turno_inicio'] = 0
            self.df_recursos['turno_fin'] = 0

        #Procesamiento de Demora_Cambio
        if 'Demora_Cambio' in self.df_recursos.columns: 
            self.df_recursos['Demora_Cambio'] = pd.to_numeric(self.df_recursos['Demora_Cambio'], errors='coerce').fillna(0)
        else: # Si no existe, inicializar a 0
            self.df_recursos['Demora_Cambio'] = 0

        #Procesamiento de Cambio (Booleano)
        if 'Cambio' in self.df_recursos.columns: 
            self.df_recursos['Cambio'] = self.df_recursos['Cambio'].fillna(False).astype(bool)
        else: # Si no existe, inicializar a False
            self.df_recursos['Cambio'] = False

        #Procesamiento de Cantidad
        if 'Cantidad' in self.df_recursos.columns: 
            self.df_recursos['Cantidad'] = pd.to_numeric(self.df_recursos['Cantidad'].astype(str).str.replace(',', '.'), errors='coerce').fillna(1)
        else: # Si no existe, inicializar a 1
            self.df_recursos['Cantidad'] = 1


    def procesar_datos_actividad(self): 
        """Procesar datos de actividades para asegurar formato consistente (ID único, Distribución, Parámetros)"""
        for idx, fila in self.df_actividades.iterrows():
            #Crear ID único para identificar cada fila, incluso si se repiten actividades
            id_unico = f"{fila['Nivel']}_{fila.get('Secuencia', idx)}_{fila['Actividad']}_{fila.get('Recurso', 'default')}_{idx}" 
            self.df_actividades.at[idx, 'unique_id'] = id_unico
            
            #Asignar distribución por defecto
            if pd.isna(fila['Distribucion']) or fila['Distribucion'] == '':
                self.df_actividades.at[idx, 'Distribucion'] = 'cte' 
            
            #Procesar parámetros de distribución
            if isinstance(fila['Tiempo'], str) and '{' in str(fila['Tiempo']): 
                try:
                    params = eval(fila['Tiempo']) #Evaluar string a diccionario de parámetros
                    self.df_actividades.at[idx, 'params'] = params
                    #Intentar obtener tiempo base desde parámetros
                    if 'loc' in params: self.df_actividades.at[idx, 'tiempo_base'] = abs(params['loc'])
                    elif 'scale' in params: self.df_actividades.at[idx, 'tiempo_base'] = abs(params['scale'])
                    else: self.df_actividades.at[idx, 'tiempo_base'] = 1.0
                except: #En caso de error de evaluación
                    self.df_actividades.at[idx, 'params'], self.df_actividades.at[idx, 'tiempo_base'] = {}, 1.0
            else: #Si es un valor constante
                self.df_actividades.at[idx, 'params'] = {}
                try: 
                    self.df_actividades.at[idx, 'tiempo_base'] = float(fila['Tiempo']) if pd.notna(fila['Tiempo']) else 1.0 # Convertir a float
                except: 
                    self.df_actividades.at[idx, 'tiempo_base'] = 1.0

    def cargar_restricciones_desde_excel(self, archivo_excel): 
        """Cargar restricciones desde archivo Excel (Hundimiento, Restricción General, Obras Civiles)."""
        try:
            #Cargar las diferentes hojas de restricciones
            self.restricciones['frente_hundimiento'] = pd.read_excel(archivo_excel, sheet_name='FrenteHundimiento')
            self.restricciones['front_index'] = pd.read_excel(archivo_excel, sheet_name='Restriccion')
            self.restricciones['obras_civiles'] = pd.read_excel(archivo_excel, sheet_name='ObrasCiviles')
            
            #    Si la hoja 'Restriccion' no tiene la columna 'Nombre', la creamos
            if 'Nombre' not in self.restricciones['front_index'].columns: 
                self.restricciones['front_index']['Nombre'] = 'Restriccion_' + self.restricciones['front_index'].index.astype(str)

            return True
        except Exception as e:
            st.error(f"Error al cargar archivo de restricciones: {str(e)}")
            return False

#################################DISTRIBUCIONES LIGADAS A LA BD INICIAL##################################################

    def muestrear_de_distribucion(self, nombre_dist, params, size=1): 
        """Generar muestras (simulaciones) de distribuciones específicas de tiempo."""
        try:
            nombre_dist = nombre_dist.lower()
            if nombre_dist in ['constant', 'cte', 'constante']: #Caso de distribución constante
                return torch.full((int(size),), params.get('value', 1.0), device=device) #Tensor con valor constante

            dist_map = { #Mapeo de distribuciones de scipy
                'norm': (stats.norm.rvs, {}), 'lognorm': (stats.lognorm.rvs, {'s': params.get('s', 0.5)}),
                'weibull': (stats.weibull_min.rvs, {'c': params.get('c', 1.0)}), 'gamma': (stats.gamma.rvs, {'a': params.get('a', 1.99)}),
                'fisk': (stats.fisk.rvs, {'c': params.get('c', 1.0)}), 'rayleigh': (stats.rayleigh.rvs, {}),
                'foldcauchy': (stats.foldnorm.rvs, {'c': params.get('c', 1.0)}), 'foldnorm': (stats.foldnorm.rvs, {'c': params.get('c', 1.0)}),
                'ncx2': (stats.ncx2.rvs, {'df': params.get('df', 1), 'nc': params.get('nc', 1)})
            }

            if nombre_dist in dist_map: #Si es una distribución de scipy
                func_dist, params_forma = dist_map[nombre_dist]
                #Parámetros comunes de localización y escala + parámetros de forma
                kwargs = {'loc': params.get('loc', 0), 'scale': params.get('scale', 1), 'size': int(size), **params_forma} 
                samples = np.abs(func_dist(**kwargs)) #Obtener muestras (tiempo no negativo)
                return torch.tensor(samples, device=device, dtype=torch.float32).clamp(min=0) #Convertir a tensor de Torch y asegurar no negativos
            else: #Distribución normal por defecto (si no se reconoce)
                dist = torch.distributions.Normal(torch.tensor(params.get('loc', 0.0), device=device), torch.tensor(params.get('scale', 1.0), device=device))
                return dist.sample((int(size),)).clamp(min=0)
        except Exception as e:
            st.warning(f"Error en distribución {nombre_dist}: {e}. Usando valor constante.")
            return torch.full((int(size),), params.get('tiempo_base', 1.0), device=device) # Valor constante si hay error
    
    def calcular_valor_esperado(self, nombre_dist, params): 
        """Calcula el valor esperado (media) para las distribuciones soportadas."""
        nombre_dist = nombre_dist.lower()
        if nombre_dist in ['constant', 'cte', 'constante']:
            return params.get('value', 1.0) #Valor esperado = constante
        
        loc = params.get('loc', 0) #Parámetro de localización (media)
        scale = params.get('scale', 1) #Parámetro de escala

        if nombre_dist == 'norm':
            return loc #Media para Normal
        elif nombre_dist == 'lognorm':
            s = params.get('s', 0.5) #Parámetro de forma 's'
            return scale * np.exp(s**2 / 2) #Media para Log-Normal
        elif nombre_dist == 'weibull':
            c = params.get('c', 1.0) #Parámetro de forma 'c'
            return scale * stats.gamma.gamma(1 + 1/c) #Media para Weibull
        elif nombre_dist == 'gamma':
            a = params.get('a', 1.99) #Parámetro de forma 'a'
            return a * scale #Media para Gamma
        elif nombre_dist == 'fisk':
            c = params.get('c', 1.0)
            if c > 1:
                return scale * (np.pi/c) / np.sin(np.pi/c) #Media para Fisk si c>1
            return np.nan
        elif nombre_dist == 'rayleigh':
            return scale * np.sqrt(np.pi / 2) #Media para Rayleigh
        elif nombre_dist in ['foldcauchy', 'foldnorm']:
            return loc
        elif nombre_dist == 'ncx2':
            df = params.get('df', 1)
            nc = params.get('nc', 1)
            return df + nc
        else:
            return loc

    @st.cache_data #Cachear la función
    def graficar_comparacion_distribucion(_self, nombre_dist_original, tupla_params_original, 
                                   nombre_dist_modificada=None, tupla_params_modificada=None, titulo="Distribución"):
        """Crea un gráfico comparando la distribución original con la modificada usando Plotly."""
        fig = go.Figure() #Inicializar figura
        rango_x_max = 0 #Inicializar rango máximo del eje X

        def _graficar_dist_individual_go(fig_obj, nombre_dist, tupla_params, etiqueta, color, trazo): #Función interna para graficar una sola distribución
            params = dict(tupla_params) # Convertir tupla a diccionario
            nombre_dist = nombre_dist.lower()
            scale = params.get('scale', 0) if nombre_dist != 'cte' else 0 # Escala (para rango)
            loc = params.get('loc', params.get('value', params.get('tiempo_base', 1))) # Locación o valor constante (para rango)
            rango_x_actual_max = max(10, loc + 5 * scale) # Estimar rango X 

            if nombre_dist in ['constant', 'cte', 'constante']: # Caso de constante
                valor_constante = params.get('value', loc)
                fig_obj.add_vline(x=valor_constante, line_width=2.5, line_dash=trazo, line_color=color, # Línea vertical
                                  annotation_text=f"{etiqueta} (Valor={valor_constante:.2f})", annotation_position="top right")
                return max(rango_x_actual_max, valor_constante * 1.5) # Ajustar rango X

            x = np.linspace(0, max(rango_x_actual_max, 10), 1000) # Puntos para graficar
            mapa_pdf = { # Mapeo de funciones de densidad de probabilidad (PDF)
                'norm': (stats.norm.pdf, {}), 'lognorm': (stats.lognorm.pdf, {'s': params.get('s', 0.5)}),
                'weibull': (stats.weibull_min.pdf, {'c': params.get('c', 1.0)}), 'gamma': (stats.gamma.pdf, {'a': params.get('a', 1.99)}),
                'fisk': (stats.fisk.pdf, {'c': params.get('c', 1.0)}), 'rayleigh': (stats.rayleigh.pdf, {}),
                'foldcauchy': (stats.foldnorm.pdf, {'c': params.get('c', 1.0)}), 'foldnorm': (stats.foldnorm.pdf, {'c': params.get('c', 1.0)}),
                'ncx2': (stats.ncx2.pdf, {'df': params.get('df', 1), 'nc': params.get('nc', 1)})
            }

            y = np.zeros_like(x)
            if nombre_dist in mapa_pdf: #Si la distribución es soportada
                func_pdf, params_forma = mapa_pdf[nombre_dist]
                escala_pdf = params.get('scale', 1e-9)
                if escala_pdf <= 0: escala_pdf = 1e-9
                kwargs = {'loc': params.get('loc', 0), 'scale': escala_pdf, **params_forma}
                y = func_pdf(x, **kwargs) #Calcular la PDF

            fig_obj.add_trace(go.Scatter( #Graficar PDF como línea
                x=x, y=y, mode='lines', name=f"{etiqueta} ({nombre_dist})",
                line=dict(color=color, width=2, dash=trazo)
            ))
            fig_obj.add_trace(go.Scatter( #Rellenar área bajo la curva
                x=x, y=y, fill='tozeroy', mode='none',
                fillcolor=color, opacity=0.2,
                showlegend=False,
                hoverinfo='none'
            ))
            return rango_x_actual_max

        max_x1 = _graficar_dist_individual_go(fig, nombre_dist_original, tupla_params_original, "Original", "blue", "solid") #Graficar original
        rango_x_max = max(rango_x_max, max_x1) #Ajustar rango X

        if nombre_dist_modificada and tupla_params_modificada: # Si hay modificada
            max_x2 = _graficar_dist_individual_go(fig, nombre_dist_modificada, tupla_params_modificada, "Modificada", "red", "dash") #Graficar modificada
            rango_x_max = max(rango_x_max, max_x2) #Ajustar rango X

        fig.update_layout( #Configurar layout
            title=titulo,
            xaxis_title='Tiempo (horas)',
            yaxis_title='Densidad de Probabilidad',
            xaxis=dict(range=[0, rango_x_max]),
            yaxis=dict(range=[0, None]),
            legend_title_text='Distribución',
            template='plotly_white'
        )
        return fig

    def verificar_restricciones(self, x, y, radio_restriccion): 
        """Verificar si una posición viola restricciones operacionales y devolver la demora total."""
        demora_total = 0 #Demora inicial
        lista_restricciones = []
        for tipo_restriccion, df in self.restricciones.items(): #Iterar por tipo de restricción
            if df is not None:
                for _, restriccion in df.iterrows():
                    distancia = np.sqrt((x - restriccion['X'])**2 + (y - restriccion['Y'])**2) # Calcular distancia euclidiana
                    if distancia < radio_restriccion: # Si viola la restricción (está dentro del radio)
                        demora_total += restriccion.get('Demora', 1) # Sumar demora (por defecto 1)
                        lista_restricciones.append(tipo_restriccion) # Registrar el tipo de restricción
        return demora_total, lista_restricciones
    
    def _obtener_demoras_combinadas(self, nivel, demoras_manuales): 
        """Helper para unificar demoras de Excel (recursos/turno) y demoras manuales (por ciclo)."""
        todas_las_demoras = []
        if self.df_recursos is not None: # Demoras de Excel (por recurso/turno)
            # Filtrar recursos que cambian y tienen demora en el nivel
            demoras_excel = self.df_recursos[(self.df_recursos['Nivel'] == nivel) & (self.df_recursos['Cambio'] == True) & (self.df_recursos['Demora_Cambio'] > 0)]
            for _, info_demora in demoras_excel.iterrows():
                todas_las_demoras.append({
                    "recurso": info_demora['Recurso'],
                    "hora_inicio": info_demora['turno_inicio'],
                    "duracion": info_demora['Demora_Cambio'],
                    "ciclo_inicio": -1, # -1 indica que aplica en todos los ciclos para el Gantt (lógica de cambio de turno)
                    "ciclo_final": -1
                })
        for demora_manual in demoras_manuales: #Demoras manuales (por ciclo)
            if demora_manual.get('nivel') == nivel: #Solo las del nivel seleccionado
                 todas_las_demoras.append({
                    "recurso": demora_manual['recurso'],
                    "hora_inicio": demora_manual['turno'][0], #Hora de inicio del turno
                    "duracion": demora_manual['demora'],
                    "ciclo_inicio": demora_manual['ciclo_inicio'],
                    "ciclo_final": demora_manual['ciclo_final']
                })
        return todas_las_demoras
    
    def simular_avance_tunel(self, info_frente, largo_avance, num_simulaciones, 
                              limite_tiempo, radio_restriccion, tiempo_demora_restriccion,
                              porcentaje_modificacion_tiempo, tiempos_modificados, ajustes_recursos,
                              demoras_manuales=[], actividades_usuario=[], plan_obras_civiles=None, condicion_geologica=None,
                              parametros_averias_equipo=None, demora_interfase_tunel=0.0, plan_restriccion=None):
        """
        Simulación Monte Carlo del avance de un frente específico.
        Incorpora lógica de restricciones de posición, eventos por ciclo (obras, averías, geología) 
        y demoras de interfase.
        """
        nivel = info_frente['Nivel'] #Nivel del frente
        #Calcular la distancia total planeada
        distancia_planeada = np.sqrt((info_frente['Xf'] - info_frente['Xi'])**2 + (info_frente['Yf'] - info_frente['Yi'])**2) 
        #Calcular el vector unitario de dirección
        vector_direccion = np.array([info_frente['Xf'] - info_frente['Xi'], info_frente['Yf'] - info_frente['Yi']]) / (distancia_planeada or 1) 
        resultados = []

        lista_actividades_nivel = []
        if self.df_actividades is not None:
            lista_actividades_nivel.extend(self.df_actividades[self.df_actividades['Nivel'] == nivel].to_dict('records'))
        lista_actividades_nivel.extend([ua for ua in actividades_usuario if ua['Nivel'] == nivel]) #Incluir actividades de usuario
        df_actividades_nivel = pd.DataFrame(lista_actividades_nivel) #Df de actividades del nivel
        
        todas_las_demoras = self._obtener_demoras_combinadas(nivel, demoras_manuales) #Obtener demoras combinadas

        # Preparar restricciones de distancia antes de la simulación
        info_demoras_restriccion_aplicadas = {}

# Lógica para aplicar el plan de restricción seleccionado
        if plan_restriccion:
            #Obtener el DF de restricciones del tipo seleccionado en el plan
            df_restriccion = self.restricciones.get(plan_restriccion['type']) 
            if df_restriccion is not None:
            # Filtrar solo la restricción por el nombre del plan
                for _, fila in df_restriccion[df_restriccion['Nombre'] == plan_restriccion['name']].iterrows(): 
            #Calcular distancia de la restricción desde el inicio del frente
                    distancia_restriccion_desde_inicio = np.sqrt((fila['X'] - info_frente['Xi'])**2 + (fila['Y'] - info_frente['Yi'])**2) 
                    if distancia_restriccion_desde_inicio < distancia_planeada: # Si está dentro del frente
                        info_demoras_restriccion_aplicadas[f"{plan_restriccion['type']}_{fila.name}"] = {
                            'distancia': distancia_restriccion_desde_inicio,
                            'aplicada': False,
                            'tiempo_demora': fila.get('Demora', 1) * tiempo_demora_restriccion #Demora en horas
                        }
        
        for sim in range(num_simulaciones): #Bucle de simulación Monte Carlo
            tiempo_sim_total, distancia_total, contador_ciclos = 0.0, 0.0, 0 #Inicializar variables por simulación
            x_actual, y_actual = info_frente['Xi'], info_frente['Yi'] #Posición inicial
            demoras_manuales_aplicadas_esta_corrida = set() #Demoras manuales aplicadas (solo una vez por simulación)
            cw_aplicadas_esta_corrida = set() #Obras civiles aplicadas (solo una vez por simulación si no es recurrente)
            
            #Inicializar las restricciones para cada nueva simulación
            banderas_restriccion_aplicadas = {clave: False for clave in info_demoras_restriccion_aplicadas.keys()}
            
            #Resetear el largo de avance por ciclo para la simulación
            largo_avance_actual = largo_avance

            #Aplicar tiempo de Interfase de túnel (se aplica una vez al inicio)
            if demora_interfase_tunel > 0:
                tiempo_sim_total += demora_interfase_tunel # Sumar demora de interfase

            while distancia_total < distancia_planeada and tiempo_sim_total < limite_tiempo: # Bucle de ciclos
                
                #Incrementamos el contador de ciclos al inicio de cada iteración
                contador_ciclos += 1
                
                # --- APLICACIÓN DE RESTRICCIONES EN EL CICLO ESPECÍFICO O RANGO ---
                
# 1. Averías de equipos (se evalúan por ciclo dentro de un rango)
                duracion_averia = 0.0
                if parametros_averias_equipo:
                    for recurso, params in parametros_averias_equipo.items():
                        prob_averia = params.get('probability', 0) / 100.0 # Probabilidad de avería
                        ciclo_inicio_averia = params.get('cycle_start', -1)
                        ciclo_fin_averia = params.get('cycle_end', -1)

                        if ciclo_inicio_averia <= contador_ciclos <= ciclo_fin_averia: # Si el ciclo está en el rango
                            if np.random.rand() < prob_averia: # Aplicar avería con probabilidad
                                duracion_averia += params.get('duration', 0)

                if duracion_averia > 0:
                    tiempo_sim_total += duracion_averia # Sumar duración de avería
                    
# 2. Demoras de Obras Civiles (se aplican en un rango de ciclos)
                if plan_obras_civiles:
                    ciclo_inicio_cw = plan_obras_civiles.get('cycle_start', -1)
                    ciclo_fin_cw = plan_obras_civiles.get('cycle_end', -1)
                    
                    # Si está en rango y no se ha aplicado (aunque la lógica de Gantt permite que aplique por ciclo)
                    if ciclo_inicio_cw <= contador_ciclos <= ciclo_fin_cw and plan_obras_civiles.get('name') not in cw_aplicadas_esta_corrida: 
                        params_cw = plan_obras_civiles.get('params', {})
                        dist_cw = plan_obras_civiles.get('distribucion', 'cte')
                        duracion_obras_civiles = self.muestrear_de_distribucion(dist_cw, params_cw, 1)[0].item() # Muestrear duración
                        tiempo_sim_total += duracion_obras_civiles # Sumar a tiempo total
                        cw_aplicadas_esta_corrida.add(plan_obras_civiles.get('name')) # Marcar como aplicada

# 3. Demoras por restricciones de posición (solo si hay plan de restricción)
                demora_restriccion_este_ciclo = 0.0
                # Aplicar restricciones de posición solo si el ciclo actual está en el rango del plan
                if plan_restriccion and plan_restriccion.get('cycle_start') <= contador_ciclos <= plan_restriccion.get('cycle_end'): 
                    for id_res, info_res in info_demoras_restriccion_aplicadas.items():
                        # Si no se ha aplicado Y la distancia total avanzada alcanzó la posición de la restricción
                        if not banderas_restriccion_aplicadas[id_res] and distancia_total >= info_res['distancia']: 
                            demora_restriccion_este_ciclo += info_res['tiempo_demora'] # Sumar demora
                            banderas_restriccion_aplicadas[id_res] = True # Marcar como aplicada
                
                if demora_restriccion_este_ciclo > 0:
                    tiempo_sim_total += demora_restriccion_este_ciclo # Sumar a tiempo total
                
# 4. Condiciones geológicas (se aplican en un rango de ciclos)
                mod_geologica = 0 # Inicializar modificación geológica (porcentaje)
                if condicion_geologica:
                    ciclo_inicio_geol = condicion_geologica.get('cycle_start', -1)
                    ciclo_fin_geol = condicion_geologica.get('cycle_end', -1)
                    
                    if ciclo_inicio_geol <= contador_ciclos <= ciclo_fin_geol: # Si está en rango
                        # Modificación al largo de avance
                        factor_mod_avance = 1.0 + (condicion_geologica.get('advance_length_mod', 0) / 100.0)
                        largo_avance_actual = largo_avance * factor_mod_avance # Ajustar largo de avance del ciclo
                        
                        # Modificación a los recursos (porcentaje de ajuste de tiempo)
                        mod_geologica = condicion_geologica.get('resource_mods', {})
                    else:
                        largo_avance_actual = largo_avance # Volver a largo original fuera del rango

                # Resetear el tiempo de un ciclo para calcular su duración
                tiempo_ciclo = 0.0
                
                # Iterar sobre las actividades del ciclo para calcular la duración total
                for _, actividad in df_actividades_nivel.iterrows():
                    id_unico, recurso = actividad['unique_id'], actividad.get('Recurso', 'N/A')

                    # --- APLICACIÓN DE DEMORAS MANUALES DENTRO DEL CICLO ---
                    for idx_demora, demora in enumerate(todas_las_demoras):
                        id_demora = f"manual_delay_{idx_demora}"
                        # Si la demora manual aplica a este ciclo y es la primera vez en la simulación
                        if demora['ciclo_inicio'] <= contador_ciclos <= demora['ciclo_final']: 
                            if id_demora not in demoras_manuales_aplicadas_esta_corrida:
                                tiempo_ciclo += demora['duracion'] # Sumar duración
                                demoras_manuales_aplicadas_esta_corrida.add(id_demora) # Marcar como aplicada
                        
                    # Se aplican las modificaciones (Sensibilización, Geológicas, etc)
                    # Obtener parámetros modificados o los originales
                    params = st.session_state.modified_times.get(id_unico, {}).get('params', actividad.get('params', {}) or {'tiempo_base': actividad['tiempo_base']}).copy()
                    nombre_dist = st.session_state.modified_times.get(id_unico, {}).get('distribucion', actividad['Distribucion'])
                    
                    mod_res_global = ajustes_recursos.get(nivel, {}).get(recurso, 0) # Ajuste global por recurso
                    # Ajuste geológico (solo si el recurso y el nivel coinciden y estamos en el rango del ciclo)
                    mod_res_geol = mod_geologica.get(recurso, 0) if condicion_geologica and condicion_geologica.get('level') == nivel and condicion_geologica.get('cycle_start') <= contador_ciclos <= condicion_geologica.get('cycle_end') else 0 

                    factor_mod = 1.0 + ((porcentaje_modificacion_tiempo + mod_res_global + mod_res_geol) / 100.0) # Factor total de modificación

                    params_temp = params.copy()
                    # Modificar parámetros de tiempo (loc, scale, value, tiempo_base) por el factor
                    for clave in ['tiempo_base', 'value', 'loc', 'scale']: 
                        if clave in params_temp: params_temp[clave] *= factor_mod

                    # Mantener parámetros de forma (s, c, a) sin modificar por el factor
                    if nombre_dist.lower() == 'lognorm' and 's' in params: params_temp['s'] = params['s'] 
                    elif nombre_dist.lower() in ['weibull', 'fisk'] and 'c' in params: params_temp['c'] = params['c']
                    elif nombre_dist.lower() == 'gamma' and 'a' in params: params_temp['a'] = params['a']

                    tiempo_ciclo += self.muestrear_de_distribucion(nombre_dist, params_temp, 1)[0].item() # Muestrear duración y sumar al ciclo

                if tiempo_sim_total + tiempo_ciclo > limite_tiempo: # Verificar límite de tiempo
                    break # Terminar simulación si se excede el límite

                tiempo_sim_total += tiempo_ciclo # Sumar tiempo de ciclo
                distancia_total += largo_avance_actual # Sumar avance del ciclo
                x_actual += vector_direccion[0] * largo_avance_actual # Actualizar coordenada X
                y_actual += vector_direccion[1] * largo_avance_actual # Actualizar coordenada Y
            # --- FIN DEL BUCLE PRINCIPAL DE SIMULACIÓN ---

            resultados.append({ # Almacenar resultados de la simulación
                'Frentes': info_frente['Frentes'],
                'Nivel': info_frente['Nivel'],
                'planned_distance': distancia_planeada,
                'actual_distance': min(distancia_total, distancia_planeada), # No exceder la distancia planeada
                'total_sim_time': tiempo_sim_total,
                'completed_cycles': contador_ciclos,
                'avg_cycle_time': tiempo_sim_total / contador_ciclos if contador_ciclos > 0 else 0,
                'efficiency': min(distancia_total, distancia_planeada) / distancia_planeada if distancia_planeada > 0 else 0,
                'civil_works_plan': plan_obras_civiles.get('name') if plan_obras_civiles else 'Ninguno',
                'geological_condition': condicion_geologica.get('name') if condicion_geologica else 'Ninguna',
                'restriction_plan': plan_restriccion.get('name') if plan_restriccion else 'Ninguno',
            })
        return pd.DataFrame(resultados) # Retornar resultados en un DataFrame

    def simular_distribuciones_actividad(self, nivel, num_muestras, porcentaje_modificacion_tiempo, ajustes_recursos, tiempos_modificados, actividades_usuario=[], condicion_geologica=None): 
        """Simula las duraciones de cada actividad (Monte Carlo) para análisis estadístico (Boxplot)."""
        lista_actividades_nivel = []
        if self.df_actividades is not None:
            lista_actividades_nivel.extend(self.df_actividades[self.df_actividades['Nivel'] == nivel].to_dict('records'))
        lista_actividades_nivel.extend([ua for ua in actividades_usuario if ua['Nivel'] == nivel]) # Incluir actividades de usuario
        
        df_actividades_nivel = pd.DataFrame(lista_actividades_nivel)

        if df_actividades_nivel.empty:
            return pd.DataFrame(columns=['Actividad', 'Recurso', 'Duracion'])

        datos_actividad = [] #Lista para almacenar las muestras de duración

        for _, actividad in df_actividades_nivel.iterrows():
            id_unico, recurso = actividad['unique_id'], actividad.get('Recurso', 'N/A')

            # Lógica para obtener parámetros modificados o los originales
            params = tiempos_modificados.get(id_unico, {}).get('params', actividad.get('params', {}) or {'tiempo_base': actividad['tiempo_base']}).copy()
            nombre_dist = tiempos_modificados.get(id_unico, {}).get('distribucion', actividad['Distribucion'])

            mod_res_global = ajustes_recursos.get(nivel, {}).get(recurso, 0)
            #Ajuste geológico
            mod_geol = condicion_geologica.get('resource_mods', {}).get(recurso, 0) if condicion_geologica else 0
            
            factor_mod = 1.0 + ((porcentaje_modificacion_tiempo + mod_res_global + mod_geol) / 100.0) #Factor total

            params_temp = params.copy()
            #Modificar parámetros de tiempo por el factor
            for clave in ['tiempo_base', 'value', 'loc', 'scale']: 
                if clave in params_temp: params_temp[clave] *= factor_mod
            
            #Restaurar parámetros de forma si se han modificado por el factor (deben ser los originales o los del UI)
            original_params_activity = actividad.get('params', {})
            modified_params_ui = tiempos_modificados.get(id_unico, {}).get('params', {})

            if nombre_dist.lower() == 'lognorm':
                params_temp['s'] = modified_params_ui.get('s', original_params_activity.get('s', 0.5))
            elif nombre_dist.lower() in ['weibull', 'fisk']:
                params_temp['c'] = modified_params_ui.get('c', original_params_activity.get('c', 1.0))
            elif nombre_dist.lower() == 'gamma':
                params_temp['a'] = modified_params_ui.get('a', original_params_activity.get('a', 1.99))


            muestras = self.muestrear_de_distribucion(nombre_dist, params_temp, size=num_muestras) # Muestrear
            
            for muestra in muestras:
                datos_actividad.append({'Actividad': actividad['Actividad'], 'Recurso': actividad.get('Recurso', 'N/A'), 'Duracion': muestra.item()})

        return pd.DataFrame(datos_actividad)

    def generar_datos_gantt(self, nivel, num_muestras, porcentaje_modificacion_tiempo, 
                          ajustes_recursos, tiempos_modificados,
                          demoras_manuales=[], actividades_usuario=[], plan_obras_civiles=None, condicion_geologica=None,
                          parametros_averias_equipo=None):
        """
        Genera datos (inicio, fin, duración) para el gráfico de Gantt probabilístico 
        (percentiles P10, P50, P90), incluyendo demoras como eventos separados.
        """
        lista_actividades_nivel = []
        if self.df_actividades is not None:
            lista_actividades_nivel.extend(self.df_actividades[self.df_actividades['Nivel'] == nivel].to_dict('records'))
        lista_actividades_nivel.extend([ua for ua in actividades_usuario if ua['Nivel'] == nivel]) # Incluir actividades de usuario
        
        # Crear DataFrame de actividades del nivel y ordenar por secuencia
        df_actividades_nivel = pd.DataFrame(lista_actividades_nivel).sort_values(by='Secuencia').reset_index(drop=True) 
        
        if df_actividades_nivel.empty:
            return pd.DataFrame(), pd.DataFrame()

        todas_las_demoras = self._obtener_demoras_combinadas(nivel, demoras_manuales)

        #Inicializar diccionarios para almacenar los resultados de cada simulación (para cada actividad y demora)
        resultados_actividad = {act['unique_id']: {'starts': [], 'finishes': [], 'durations': []} for _, act in df_actividades_nivel.iterrows()} 
        resultados_demora = {} 
        
        num_actividades = len(df_actividades_nivel)

        for _ in range(num_muestras): #Bucle de simulación Monte Carlo
            tiempo_ciclo_actual = 0.0 #Tiempo acumulado en esta simulación
            
            for idx_ciclo in range(num_actividades): #Iterar sobre las actividades (que representan los ciclos)
                
                #El ciclo actual es idx_ciclo + 1
                ciclo_actual = idx_ciclo + 1

# 1. Aplicar demoras de obras civiles si están en el rango de ciclos
                if plan_obras_civiles:
                    ciclo_inicio_cw = plan_obras_civiles.get('cycle_start', -1)
                    ciclo_fin_cw = plan_obras_civiles.get('cycle_end', -1)
                    if ciclo_inicio_cw <= ciclo_actual <= ciclo_fin_cw: # Si aplica en este ciclo
                        id_demora = f"demora_obras_civiles_{ciclo_inicio_cw}-{ciclo_fin_cw}" # ID único para el evento
                        if id_demora not in resultados_demora:
                            resultados_demora[id_demora] = {'starts': [], 'finishes': [], 'durations': [], 'label': f"Obras Civiles: {plan_obras_civiles.get('name')} (Ciclos {ciclo_inicio_cw}-{ciclo_fin_cw})"}
                        
                        # Muestrear duración
                        params_cw = plan_obras_civiles.get('params', {})
                        dist_cw = plan_obras_civiles.get('distribucion', 'cte')
                        duracion_obras_civiles = self.muestrear_de_distribucion(dist_cw, params_cw, 1)[0].item()

                        # Almacenar y actualizar tiempo
                        resultados_demora[id_demora]['starts'].append(tiempo_ciclo_actual)
                        resultados_demora[id_demora]['durations'].append(duracion_obras_civiles)
                        tiempo_ciclo_actual += duracion_obras_civiles
                        resultados_demora[id_demora]['finishes'].append(tiempo_ciclo_actual)

# 2. Aplicar averías de equipos si están en el rango
                if parametros_averias_equipo:
                    for recurso, params in parametros_averias_equipo.items():
                        prob_averia = params.get('probability', 0) / 100.0
                        ciclo_inicio_averia = params.get('cycle_start', -1)
                        ciclo_fin_averia = params.get('cycle_end', -1)
                        if ciclo_inicio_averia <= ciclo_actual <= ciclo_fin_averia: # Si aplica en este ciclo
                            if np.random.rand() < prob_averia: # Si ocurre la avería
                                id_demora = f"averia_{recurso}_{ciclo_inicio_averia}-{ciclo_fin_averia}"
                                if id_demora not in resultados_demora:
                                    resultados_demora[id_demora] = {'starts': [], 'finishes': [], 'durations': [], 'label': f"Avería: {recurso} (Ciclos {ciclo_inicio_averia}-{ciclo_fin_averia})"}
                                
                                duracion_averia = params.get('duration', 0)
                                # Almacenar y actualizar tiempo
                                resultados_demora[id_demora]['starts'].append(tiempo_ciclo_actual)
                                resultados_demora[id_demora]['durations'].append(duracion_averia)
                                tiempo_ciclo_actual += duracion_averia
                                resultados_demora[id_demora]['finishes'].append(tiempo_ciclo_actual)

# 3. Aplicar demoras manuales si están en el rango de ciclos
                for idx_demora, demora in enumerate(todas_las_demoras):
                    if demora['ciclo_inicio'] <= ciclo_actual <= demora['ciclo_final']:
                         # Usar id manual_delay_X
                         id_demora = f"demora_manual_{idx_demora}" 
                         if id_demora not in resultados_demora:
                             resultados_demora[id_demora] = {'starts': [], 'finishes': [], 'durations': [], 'label': f"Demora: {demora['recurso']} (Ciclos {demora['ciclo_inicio']}-{demora['ciclo_final']})"}
                         
                         tiempo_inicio_demora = tiempo_ciclo_actual
                         duracion_demora = demora['duracion']
                         # Almacenar y actualizar tiempo
                         resultados_demora[id_demora]['starts'].append(tiempo_inicio_demora)
                         resultados_demora[id_demora]['durations'].append(duracion_demora)
                         tiempo_ciclo_actual += duracion_demora
                         resultados_demora[id_demora]['finishes'].append(tiempo_ciclo_actual)

                # INICIO ACTIVIDAD DE AVANCE
                actividad = df_actividades_nivel.iloc[idx_ciclo]
                id_unico, recurso = actividad['unique_id'], actividad.get('Recurso', 'N/A')
                tiempo_inicio = tiempo_ciclo_actual

                # Lógica para obtener parámetros modificados (igual que en simular_distribuciones_actividad)
                params = tiempos_modificados.get(id_unico, {}).get('params', actividad.get('params', {}) or {'tiempo_base': actividad['tiempo_base']}).copy()
                nombre_dist = tiempos_modificados.get(id_unico, {}).get('distribucion', actividad['Distribucion'])

                mod_res_global = ajustes_recursos.get(nivel, {}).get(recurso, 0)
                # Ajuste geológico
                mod_geol = condicion_geologica.get('resource_mods', {}).get(recurso, 0) if condicion_geologica else 0
                
                factor_mod = 1.0 + ((porcentaje_modificacion_tiempo + mod_res_global + mod_geol) / 100.0)

                params_temp = params.copy()
                for clave in ['tiempo_base', 'value', 'loc', 'scale']:
                    if clave in params_temp: params_temp[clave] *= factor_mod

                # Restaurar parámetros de forma si se han modificado por el factor (deben ser los originales o los del UI)
                original_params_activity = actividad.get('params', {})
                modified_params_ui = tiempos_modificados.get(id_unico, {}).get('params', {})

                if nombre_dist.lower() == 'lognorm':
                    params_temp['s'] = modified_params_ui.get('s', original_params_activity.get('s', 0.5))
                elif nombre_dist.lower() in ['weibull', 'fisk']:
                    params_temp['c'] = modified_params_ui.get('c', original_params_activity.get('c', 1.0))
                elif nombre_dist.lower() == 'gamma':
                    params_temp['a'] = modified_params_ui.get('a', original_params_activity.get('a', 1.99))

                duracion = self.muestrear_de_distribucion(nombre_dist, params_temp, size=1)[0].item() # Muestrear duración

                # Almacenar y actualizar tiempo
                resultados_actividad[id_unico]['starts'].append(tiempo_inicio)
                resultados_actividad[id_unico]['durations'].append(duracion)
                tiempo_ciclo_actual += duracion
                resultados_actividad[id_unico]['finishes'].append(tiempo_ciclo_actual)
            # FIN BUCLE DE ACTIVIDADES

# CONSOLIDAR RESULTADOS DE ACTIVIDADES (Cálculo de percentiles)
        filas_gantt = []
        for _, actividad in df_actividades_nivel.iterrows():
            id_unico = actividad['unique_id']
            if id_unico not in resultados_actividad or not resultados_actividad[id_unico]['starts']: continue
            filas_gantt.append({ 
                'Actividad': f"{actividad['Actividad']} ({actividad.get('Recurso', 'N/A')})",
                'Start_p10': np.percentile(resultados_actividad[id_unico]['starts'], 10), 'Finish_p10': np.percentile(resultados_actividad[id_unico]['finishes'], 10),
                'Start_p50': np.percentile(resultados_actividad[id_unico]['starts'], 50), 'Finish_p50': np.percentile(resultados_actividad[id_unico]['finishes'], 50),
                'Start_p90': np.percentile(resultados_actividad[id_unico]['starts'], 90), 'Finish_p90': np.percentile(resultados_actividad[id_unico]['finishes'], 90),
                'Duration_p10': np.percentile(resultados_actividad[id_unico]['durations'], 10),
                'Duration_p50': np.percentile(resultados_actividad[id_unico]['durations'], 50),
                'Duration_p90': np.percentile(resultados_actividad[id_unico]['durations'], 90),
            })

# CONSOLIDAR RESULTADOS DE DEMORAS (Cálculo de percentiles)
        filas_demora = []
        for id_demora, resultados in resultados_demora.items():
            if resultados['starts']:
                filas_demora.append({ 
                    'Actividad': resultados['label'],
                    'Start_p10': np.percentile(resultados['starts'], 10), 'Finish_p10': np.percentile(resultados['finishes'], 10),
                    'Start_p50': np.percentile(resultados['starts'], 50), 'Finish_p50': np.percentile(resultados['finishes'], 50),
                    'Start_p90': np.percentile(resultados['starts'], 90), 'Finish_p90': np.percentile(resultados['finishes'], 90),
                    'Duration_p10': np.percentile(resultados['durations'], 10),
                    'Duration_p50': np.percentile(resultados['durations'], 50),
                    'Duration_p90': np.percentile(resultados['durations'], 90),
                })

        return pd.DataFrame(filas_gantt), pd.DataFrame(filas_demora) # Retornar datos Gantt y demoras
    

#################################MAIN PRINCIPAL PARA STREAMLIT##########################################################

def main():
    st.set_page_config(page_title="Simulador de Avance de Túneles", layout="wide") # Configuración de página
    st.title("Simulador de Avance de Túneles - Monte Carlo") # Título principal

    # Inicialización de la sesión de Streamlit (Estado de la aplicación)
    if 'simulator' not in st.session_state: st.session_state.simulator = SimuladorTunel() # Inicializar clase SimuladorTunel
    if 'modified_times' not in st.session_state: st.session_state.modified_times = {} # Tiempos de actividad modificados
    if 'resource_adjustments' not in st.session_state: st.session_state.resource_adjustments = {} # Ajustes de recursos
    if 'manual_delays' not in st.session_state: st.session_state.manual_delays = [] # Demoras manuales
    if 'user_defined_activities' not in st.session_state: st.session_state.user_defined_activities = [] # Actividades de usuario
    if 'recursos_file_name' not in st.session_state: st.session_state.recursos_file_name = None
    if 'restricciones_file_name' not in st.session_state: st.session_state.restricciones_file_name = None
    if 'tiempo_planificado' not in st.session_state: st.session_state.tiempo_planificado = 160.0 # Tiempo por defecto
    if 'sim_queue' not in st.session_state: st.session_state.sim_queue = [] # Cola de frentes a simular
    if 'sim_results' not in st.session_state: st.session_state.sim_results = pd.DataFrame() # Resultados de simulación consolidados
    if 'current_sim_index' not in st.session_state: st.session_state.current_sim_index = 0
    if 'sim_started' not in st.session_state: st.session_state.sim_started = False
    if 'per_tunnel_data' not in st.session_state: st.session_state.per_tunnel_data = {} # Datos detallados por túnel
    if 'civil_works_plans' not in st.session_state: st.session_state.civil_works_plans = {} # Planes de obras civiles definidos
    if 'geological_conditions' not in st.session_state: st.session_state.geological_conditions = {} # Condiciones geológicas definidas
    if 'equipment_breakdowns' not in st.session_state: st.session_state.equipment_breakdowns = {} # Averías de equipos definidas
    if 'tunnel_interfase_delays' not in st.session_state: st.session_state.tunnel_interfase_delays = {} # Demoras de interfase definidas
    if 'restriction_plans' not in st.session_state: st.session_state.restriction_plans = {} # Planes de restricción definidos


    with st.sidebar: # Barra lateral para configuración de inputs
        st.header("Configuración de Simulación")

#SENS - RECURSOS (Ajuste de rendimiento de recursos por nivel)

        st.markdown("---")
        st.subheader("Sensibilización de Recursos - Variación de Atraso o Adelantamiento del Tiempo")
        if st.session_state.simulator.df_actividades is not None:
            levels = st.session_state.simulator.df_actividades['Nivel'].dropna().unique() # Niveles disponibles
            level_to_config = st.selectbox("Nivel para ajustar recursos", options=levels, key="level_adj_res")
            if level_to_config:
                if level_to_config not in st.session_state.resource_adjustments:
                    st.session_state.resource_adjustments[level_to_config] = {}
                resources = obtener_recursos_por_nivel(st.session_state.simulator.df_actividades, level_to_config) # Obtener recursos por nivel
                with st.expander(f"Ajustes para Nivel '{level_to_config}'", expanded=True):
                    if not resources: st.warning("No hay recursos asignados en este nivel.")
                    else:
                        for r in resources: # Slider para cada recurso (Ajuste en porcentaje)
                            st.session_state.resource_adjustments[level_to_config][r] = st.slider(f"Ajuste para **{r}** (%)", -50, 50, st.session_state.resource_adjustments[level_to_config].get(r, 0), 5, key=f"adj_{level_to_config}_{r}")
        else:
            st.info("Cargue datos para configurar recursos.")

#IMPUTAR DEMORAS POR CICLOS (Demoras manuales)

        st.markdown("---")
        st.subheader("Imputar Demoras Manuales")
        if st.session_state.simulator.df_actividades is not None:
            with st.expander("Añadir nueva demora por evento/cambio"):
                niveles = st.session_state.simulator.df_actividades['Nivel'].dropna().unique()
                nivel_demora = st.selectbox("Nivel de la demora", niveles, key="d_nivel")
                recursos_nivel = obtener_recursos_por_nivel(st.session_state.simulator.df_actividades, nivel_demora)
                recurso_demora = st.selectbox("Recurso afectado (o genérico)", recursos_nivel, key="d_rec")
                col_ciclo_inicio_dm, col_ciclo_final_dm = st.columns(2)
                with col_ciclo_inicio_dm:
                    ciclo_inicio_demora = st.number_input("Ciclo de inicio", min_value=1, value=2, step=1, key="d_ciclo_inicio") # Ciclo donde empieza a aplicar
                with col_ciclo_final_dm:
                    ciclo_final_demora = st.number_input("Ciclo final", min_value=ciclo_inicio_demora, value=3, step=1, key="d_ciclo_final") # Ciclo donde termina de aplicar
                turno_demora = st.slider("Turno de inicio de la demora (hora del día)", 0, 23, 7, key="d_turno")
                horas_demora = st.number_input("Horas de demora", min_value=0.5, value=4.0, step=0.5, key="d_horas")
                if st.button("Añadir Demora Manual"):
                    if ciclo_final_demora >= ciclo_inicio_demora:
                        st.session_state.manual_delays.append({"nivel": nivel_demora, "recurso": recurso_demora, "turno": (turno_demora, turno_demora + 1), "demora": horas_demora, "ciclo_inicio": ciclo_inicio_demora, "ciclo_final": ciclo_final_demora})
                        st.success(f"Demora de {horas_demora}h para {recurso_demora} en ciclos {ciclo_inicio_demora} a {ciclo_final_demora} añadida.")
                        st.rerun()
                    else:
                        st.error("El ciclo final debe ser mayor o igual al ciclo de inicio.")

            if st.session_state.manual_delays:
                st.write("Demoras manuales activas:")
                for i, d in enumerate(st.session_state.manual_delays):
                    st.info(f"{i+1}: {d['recurso']} ({d['nivel']}) - {d['demora']}h en Ciclos {d['ciclo_inicio']} a {d['ciclo_final']}.")
                if st.button("Limpiar Demoras Manuales"):
                    st.session_state.manual_delays = []
                    st.rerun()
        else:
            st.info("Cargue datos para añadir demoras.")

#IMPUTAR NUEVA ACTIVIDAD DADA POR EL USUARIO (Nueva actividad en el ciclo)

        st.markdown("---")
        st.subheader("Imputar Nueva Actividad")
        if st.session_state.simulator.df_actividades is not None:
            with st.expander("Añadir una nueva actividad a los ciclos de avance"):
                niveles = st.session_state.simulator.df_actividades['Nivel'].dropna().unique()
                if not niveles.any():
                    st.warning("No hay niveles definidos en el archivo de actividades.")
                else:
                    nivel_nueva_act = st.selectbox("Nivel de la nueva actividad", niveles, key="n_n_act")
                    nombre_nueva_act = st.text_input("Nombre de la nueva actividad", "Tiempo post tronadura", key="nm_n_act")
                    recurso_nueva_act = st.text_input("Recurso asociado (opcional)", "N/A", key="res_n_act")
                    secuencia_nueva_act = st.number_input("Número de secuencia (para ordenar en la Gantt)", min_value=0, value=99, step=1, key="seq_n_act") # Posición en el ciclo
                    dist_options_new = ['cte', 'norm', 'lognorm', 'weibull', 'gamma', 'fisk', 'rayleigh']
                    new_dist_type = st.selectbox("Tipo de Distribución", dist_options_new, key="nd_t")
                    new_act_params = {}
                    tiempo_base_new = st.number_input("Tiempo Medio/Valor Constante (horas)", 0.01, value=1.0, step=0.1, key="tb_n_act", help="Valor central o constante para la nueva distribución.")
                    new_act_params['tiempo_base'] = tiempo_base_new
                    new_act_params['value'] = tiempo_base_new
                    if new_dist_type != 'cte':
                        variability_new = st.slider("Variabilidad Relativa (scale)", 0.0, 2.0, 0.25, 0.05, key="s_n_act", help="Controla la desviación estándar en relación al tiempo medio.")
                        scale_new = tiempo_base_new * variability_new
                        new_act_params['loc'] = tiempo_base_new
                        new_act_params['scale'] = scale_new
                        if new_dist_type in ['lognorm', 'weibull', 'gamma', 'fisk']: # Parámetros de forma
                            shape_key_new = {'lognorm': 's', 'weibull': 'c', 'gamma': 'a', 'fisk': 'c'}[new_dist_type]
                            shape_label_new = {'lognorm': 'Sigma', 'weibull': 'k', 'gamma': 'alpha', 'fisk': 'c'}[new_dist_type]
                            shape_defaults_new = {'s': 0.5, 'c': 2.0, 'a': 2.0}
                            new_act_params[shape_key_new] = st.number_input(f"Parámetro de Forma ({shape_label_new})", 0.1, 20.0, shape_defaults_new.get(shape_key_new, 1.0), 0.1, key=f"p1_n_act")
                    
                    if st.button("Añadir Nueva Actividad", key="add_new_act_btn"):
                        if nombre_nueva_act:
                            # ID único para la actividad de usuario
                            new_uid = f"user_{nivel_nueva_act}_{nombre_nueva_act}_{recurso_nueva_act}_{pd.Timestamp.now().timestamp()}"
                            st.session_state.user_defined_activities.append({
                                'Nivel': nivel_nueva_act, 'Actividad': nombre_nueva_act, 'Recurso': recurso_nueva_act,
                                'Secuencia': secuencia_nueva_act, 'Distribucion': new_dist_type,
                                'Tiempo': str(new_act_params), 'unique_id': new_uid,
                                'params': new_act_params, 'tiempo_base': tiempo_base_new
                            })
                            st.success(f"Actividad '{nombre_nueva_act}' añadida con éxito.")
                            st.rerun()
                        else: st.warning("El nombre de la actividad no puede estar vacío.")

            if st.session_state.user_defined_activities:
                st.write("Actividades añadidas por el usuario:")
                for i, act in enumerate(st.session_state.user_defined_activities):
                    st.info(f"{i+1}: {act['Actividad']} ({act['Recurso']}) en Nivel {act['Nivel']} - Distribución: {act['Distribucion']}")
                if st.button("Limpiar Actividades Manuales"):
                    st.session_state.user_defined_activities = []
                    st.rerun()
        else:
            st.info("Cargue datos de actividades para añadir nuevas.")
        
# IMPUTAR PLANES DE OBRAS CIVILES DADOS POR SUPUESTOS (Eventos de obras civiles)

        st.markdown("---")
        st.subheader("Planes de Obras Civiles")
        with st.expander("Definir un nuevo plan de obras"):
            plan_name = st.text_input("Nombre del Plan", "Sostenimiento Pesado", key="new_cw_name")
            cw_dist_options = ['cte', 'norm', 'lognorm', 'weibull']
            cw_dist_type = st.selectbox("Tipo de Distribución", cw_dist_options, key="cw_dist")
            cw_time_base = st.number_input("Tiempo Estimado (horas)", 0.1, value=48.0, step=1.0, key="cw_time")
            col_cw_start, col_cw_end = st.columns(2)
            with col_cw_start:
                cw_cycle_start = st.number_input("Ciclo de inicio", min_value=1, value=5, step=1, key="cw_cycle_start") # Rango de ciclos
            with col_cw_end:
                cw_cycle_end = st.number_input("Ciclo final", min_value=cw_cycle_start, value=6, step=1, key="cw_cycle_end") # Rango de ciclos
            cw_params = {'tiempo_base': cw_time_base, 'value': cw_time_base}
            if cw_dist_type != 'cte':
                cw_variability = st.slider("Variabilidad Relativa (scale)", 0.0, 1.0, 0.15, 0.05, key="cw_var")
                cw_scale = cw_time_base * cw_variability
                cw_params['loc'] = cw_time_base
                cw_params['scale'] = cw_scale
                if cw_dist_type == 'lognorm':
                     cw_params['s'] = st.number_input("Parámetro de Forma (Sigma)", 0.1, 5.0, 0.5, 0.1, key="cw_log_s")
                elif cw_dist_type == 'weibull':
                    cw_params['c'] = st.number_input("Parámetro de Forma (k)", 0.1, 5.0, 2.0, 0.1, key="cw_weib_c")
            
            if st.button("Añadir Plan de Obras Civiles"):
                if plan_name and cw_cycle_end >= cw_cycle_start:
                    st.session_state.civil_works_plans[plan_name] = {'distribucion': cw_dist_type, 'params': cw_params, 'name': plan_name, 'cycle_start': cw_cycle_start, 'cycle_end': cw_cycle_end}
                    st.success(f"Plan '{plan_name}' añadido para los ciclos {cw_cycle_start} a {cw_cycle_end}.")
                    st.rerun()
                else: st.warning("El nombre del plan no puede estar vacío o el ciclo final es menor al de inicio.")

        if st.session_state.civil_works_plans:
            st.write("Planes de obras activos:")
            for name, plan in st.session_state.civil_works_plans.items():
                st.info(f"**{name}**: {plan['distribucion']} (Tiempo ~{plan['params']['tiempo_base']:.2f}h) - Inicia en Ciclo {plan['cycle_start']} y termina en {plan['cycle_end']}.")
            if st.button("Limpiar Planes de Obras"):
                st.session_state.civil_works_plans = {}
                st.rerun()
        

        st.markdown("---")
        st.subheader("Planes de Restricción (Geográfica)")
        if st.session_state.simulator.restricciones['front_index'] is not None:
            with st.expander("Definir un nuevo plan de restricción"):
                restriction_plan_name = st.text_input("Nombre del Plan de Restricción", "Restricción por Hundimiento", key="new_res_name")
                # Obtener la lista de nombres únicos de restricciones disponibles
                available_restrictions = st.session_state.simulator.restricciones['front_index']['Nombre'].unique().tolist() if 'front_index' in st.session_state.simulator.restricciones and st.session_state.simulator.restricciones['front_index'] is not None else []
                if not available_restrictions: st.warning("No hay restricciones disponibles para configurar.")
                else:
                    restriction_to_apply = st.selectbox("Restricción geográfica a aplicar", available_restrictions, key="res_to_apply") # La restricción del archivo Excel
                    col_res_start, col_res_end = st.columns(2)
                    with col_res_start:
                        res_cycle_start = st.number_input("Ciclo de inicio de la demora", min_value=1, value=5, step=1, key="res_cycle_start") # Rango de ciclos donde se evalúa la restricción
                    with col_res_end:
                        res_cycle_end = st.number_input("Ciclo final de la demora", min_value=res_cycle_start, value=6, step=1, key="res_cycle_end")
                    
                    if st.button("Añadir Plan de Restricción"):
                        if restriction_plan_name and res_cycle_end >= res_cycle_start and restriction_to_apply:
                            st.session_state.restriction_plans[restriction_plan_name] = {
                                'name': restriction_to_apply,
                                'type': 'front_index', # Tipo de restricción (columna del diccionario self.restricciones)
                                'cycle_start': res_cycle_start,
                                'cycle_end': res_cycle_end
                            }
                            st.success(f"Plan '{restriction_plan_name}' añadido para los ciclos {res_cycle_start} a {res_cycle_end}.")
                            st.rerun()
                        else: st.warning("El nombre del plan no puede estar vacío o los ciclos son inválidos.")
            
            if st.session_state.restriction_plans:
                st.write("Planes de restricción activos:")
                for name, plan in st.session_state.restriction_plans.items():
                    st.info(f"**{name}**: Restricción {plan['name']} - Inicia en Ciclo {plan['cycle_start']} y termina en {plan['cycle_end']}.")
                if st.button("Limpiar Planes de Restricción"):
                    st.session_state.restriction_plans = {}
                    st.rerun()
        else: st.info("Cargue datos de restricciones para añadir planes.")

# CONDICIONES GEOLÓGICAS (Modificación del largo de avance y del tiempo de recursos)

        st.markdown("---")
        st.subheader("Condiciones Geológicas")
        if st.session_state.simulator.df_actividades is not None:
            with st.expander("Definir una nueva condición geológica"):
                niveles_geol = st.session_state.simulator.df_actividades['Nivel'].dropna().unique()
                geol_name = st.text_input("Nombre de la Condición", "Roca Mala", key="new_geol_name")
                nivel_geol = st.selectbox("Nivel de la condición", niveles_geol, key="geol_level") # Nivel afectado
                col_geol_start, col_geol_end = st.columns(2)
                with col_geol_start:
                    geol_cycle_start = st.number_input("Ciclo de inicio", min_value=1, value=2, step=1, key="geol_cycle_start")
                with col_geol_end:
                    geol_cycle_end = st.number_input("Ciclo final", min_value=geol_cycle_start, value=4, step=1, key="geol_cycle_end")

                geol_advance_mod = st.number_input("Modificación del largo de avance por ciclo (%)", -50.0, 100.0, 0.0, 5.0, help="Ajusta el largo de avance del ciclo en este rango. Ej: -20% reduce el avance un 20%.")
                
                geol_resources = obtener_recursos_por_nivel(st.session_state.simulator.df_actividades, nivel_geol)
                geol_mods = {} # Diccionario de ajustes de tiempo por recurso
                st.write("Ajustes de tiempo para recursos en esta condición (en % de aumento):")
                for res in geol_resources:
                    # Ajuste de tiempo individual para cada recurso en esta condición
                    geol_mods[res] = st.slider(f"Ajuste para **{res}** (%)", -50, 100, 20, 5, key=f"geol_{nivel_geol}_{res}") 
                
                if st.button("Añadir Condición Geológica"):
                    if geol_name and geol_cycle_end >= geol_cycle_start:
                        st.session_state.geological_conditions[geol_name] = {
                            'level': nivel_geol,
                            'resource_mods': geol_mods,
                            'name': geol_name,
                            'cycle_start': geol_cycle_start,
                            'cycle_end': geol_cycle_end,
                            'advance_length_mod': geol_advance_mod
                        }
                        st.success(f"Condición '{geol_name}' añadida para los ciclos {geol_cycle_start} a {geol_cycle_end}.")
                        st.rerun()
                    else: st.warning("El nombre de la condición no puede estar vacío o el ciclo final es menor al de inicio.")
            
            if st.session_state.geological_conditions:
                st.write("Condiciones geológicas activas:")
                for name, cond in st.session_state.geological_conditions.items():
                    st.info(f"**{name}**: Nivel {cond['level']} - Ciclos {cond['cycle_start']}-{cond['cycle_end']}. Avance mod: {cond['advance_length_mod']}%. Ajustes: {cond['resource_mods']}")
                if st.button("Limpiar Condiciones Geológicas"):
                    st.session_state.geological_conditions = {}
                    st.rerun()
        else:
            st.info("Cargue datos para añadir condiciones geológicas.")

# SIMULAR AVERÍAS (Probabilidad de falla por ciclo)


        st.markdown("---")
        st.subheader("Averías de Equipos")
        if st.session_state.simulator.df_recursos is not None:
            with st.expander("Definir probabilidad de avería en un ciclo"):
                recursos_con_averia = st.session_state.simulator.df_recursos['Recurso'].unique().tolist()
                recurso_averia = st.selectbox("Recurso con avería", options=recursos_con_averia, key="eb_resource")
                prob_averia = st.slider("Probabilidad de avería (%)", 0.0, 100.0, 1.0, 0.5, key="eb_prob") # Probabilidad en %
                duracion_averia = st.number_input("Duración de la avería (horas)", 0.0, 24.0, 2.0, 0.5, key="eb_duration") # Demora en horas
                col_eb_start, col_eb_end = st.columns(2)
                with col_eb_start:
                    ciclo_inicio_averia = st.number_input("Ciclo de inicio", min_value=1, value=3, step=1, key="eb_cycle_start")
                with col_eb_end:
                    ciclo_final_averia = st.number_input("Ciclo final", min_value=ciclo_inicio_averia, value=5, step=1, key="eb_cycle_end")
                
                if st.button("Añadir Avería de Equipo"):
                    if ciclo_final_averia >= ciclo_inicio_averia:
                        st.session_state.equipment_breakdowns[recurso_averia] = {'probability': prob_averia, 'duration': duracion_averia, 'cycle_start': ciclo_inicio_averia, 'cycle_end': ciclo_final_averia}
                        st.success(f"Avería para {recurso_averia} añadida en los ciclos {ciclo_inicio_averia} a {ciclo_final_averia}.")
                        st.rerun()
                    else:
                        st.error("El ciclo final debe ser mayor o igual al ciclo de inicio.")
            
            if st.session_state.equipment_breakdowns:
                st.write("Averías de equipos activas:")
                for res, params in st.session_state.equipment_breakdowns.items():
                    st.info(f"**{res}**: {params['probability']}% de probabilidad, duración de {params['duration']}h en ciclos {params['cycle_start']} a {params['cycle_end']}.")
                if st.button("Limpiar Averías"):
                    st.session_state.equipment_breakdowns = {}
                    st.rerun()
        else:
            st.info("Cargue datos de recursos para definir averías.")
            
#DEMORAS (Demoras de interfase al inicio del frente)
        st.markdown("---")
        st.subheader("Demoras de Interfase (Inicio de Avance)")
        if st.session_state.simulator.df_frentes is not None:
            with st.expander("Añadir demoras de interfase"):
                frentes_interfase = st.session_state.simulator.df_frentes['Frentes'].unique().tolist()
                frente_interfase_sel = st.selectbox("Frente para demora de interfase", options=frentes_interfase, key="interfase_frente")
                demora_interfase_tiempo = st.number_input("Demora de Interfase (horas)", 0.0, 48.0, 0.0, 1.0, key="interfase_tiempo") # Demora que se aplica al tiempo total al inicio
                if st.button("Aplicar Demora de Interfase"):
                    st.session_state.tunnel_interfase_delays[frente_interfase_sel] = demora_interfase_tiempo
                    st.success(f"Demora de interfase de {demora_interfase_tiempo}h aplicada a {frente_interfase_sel}.")
                    st.rerun()

            if st.session_state.tunnel_interfase_delays:
                st.write("Demoras de interfase activas:")
                for frente, demora in st.session_state.tunnel_interfase_delays.items():
                    st.info(f"**{frente}**: {demora}h.")
                if st.button("Limpiar Demoras de Interfase"):
                    st.session_state.tunnel_interfase_delays = {}
                    st.rerun()
        else:
            st.info("Cargue datos de frentes para definir demoras de interfase.")

#################################CARGA DE DATOS Y VISUALIZACIÓN INICIAL##########################################################

    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Cargar Datos")
        # Uploader para archivo principal
        recursos_file = st.file_uploader("Archivo de Actividades/Recursos/Frentes", type=['xlsx', 'xls'], key="uploader1")
        if recursos_file and recursos_file.name != st.session_state.recursos_file_name:
            if st.session_state.simulator.cargar_datos_desde_excel(recursos_file): # Carga de datos principales
                st.session_state.recursos_file_name = recursos_file.name
                st.success(f"Datos '{recursos_file.name}' cargados.")
                st.rerun()
        # Uploader para archivo de restricciones
        restricciones_file = st.file_uploader("Archivo de Restricciones", type=['xlsx', 'xls'], key="uploader2")
        if restricciones_file and restricciones_file.name != st.session_state.restricciones_file_name:
            if st.session_state.simulator.cargar_restricciones_desde_excel(restricciones_file): # Carga de restricciones
                st.session_state.restricciones_file_name = restricciones_file.name
                st.success(f"Restricciones '{restricciones_file.name}' cargadas.")
                st.rerun()
        if st.session_state.recursos_file_name: st.write(f"Datos cargados: **{st.session_state.recursos_file_name}**")
        if st.session_state.restricciones_file_name: st.write(f"Restricciones cargadas: **{st.session_state.restricciones_file_name}**")

    with col2:
        st.subheader("Visualización de Frentes y Restricciones")
        if st.session_state.simulator.df_frentes is not None:
            niveles_viz = st.session_state.simulator.df_frentes['Nivel'].unique()
            nivel_selected_viz = st.selectbox("Nivel para visualización", niveles_viz, key="nivel_viz")
            if nivel_selected_viz:
                # Mostrar el gráfico de frentes y restricciones
                fig_viz = generar_visualizacion_frentes(st.session_state.simulator.df_frentes, st.session_state.simulator.restricciones, 50, nivel_selected_viz) 
                st.plotly_chart(fig_viz, use_container_width=True)

#################################MODIFICACIÓN DE DISTRIBUCIONES DE ACTIVIDADES##########################################################

    if st.session_state.simulator.df_actividades is not None:
        st.markdown("---")
        st.subheader("Modificar Distribuciones de Actividades")
        
        # Lógica de conciliación de actividades (Excel + Usuario)
        all_activities_df = st.session_state.simulator.df_actividades.copy()
        if st.session_state.user_defined_activities:
            user_df = pd.DataFrame(st.session_state.user_defined_activities)
            all_activities_df = pd.concat([all_activities_df, user_df], ignore_index=True)
            
        with st.expander("Modificación Masiva por Nivel", expanded=True):
            niveles_mod_masiva = all_activities_df['Nivel'].dropna().unique()
            nivel_masivo_selected = st.selectbox("Seleccionar Nivel para modificación masiva", niveles_mod_masiva, key="nivel_masivo_sel")
            
            if nivel_masivo_selected:
                activities_in_level = all_activities_df[all_activities_df['Nivel'] == nivel_masivo_selected]
                # Lista de actividades para mostrar en el multiselect
                all_activities_options = activities_in_level.apply(lambda x: f"{x['Actividad']} ({x['Recurso']})", axis=1).tolist()
                
                select_all_activities = st.checkbox("Seleccionar todas las actividades en este nivel", key="select_all_activities_mass")
                
                if select_all_activities: 
                    activities_to_modify_uids_display = all_activities_options
                else:
                    activities_to_modify_uids_display = st.multiselect(
                        "Seleccionar Actividades a Modificar", options=all_activities_options, default=[], key="activities_masive_sel")
                
                # Obtener los unique_id de las actividades seleccionadas
                filtered_uids = []
                for act_str in activities_to_modify_uids_display:
                    act_name = act_str.split(' (')[0]
                    res_name = act_str.split(' (')[1][:-1]
                    matching_rows = activities_in_level[(activities_in_level['Actividad'] == act_name) & (activities_in_level['Recurso'] == res_name)]
                    if not matching_rows.empty: filtered_uids.extend(matching_rows['unique_id'].tolist())
                    
                st.markdown("#### Configurar Nueva Distribución (Aplicará a las seleccionadas)")
                dist_options_masiva = ['cte', 'norm', 'lognorm', 'weibull', 'gamma', 'fisk', 'rayleigh']
                new_dist_masiva = st.selectbox("Tipo de Distribución", dist_options_masiva, key="new_dist_masiva")
                new_params_masiva = {}
                tiempo_base_masiva = st.number_input("Tiempo Medio/Valor Constante (horas)", 0.01, value=1.0, step=0.1, key="tb_masiva", help="Valor central o constante para la nueva distribución.")
                new_params_masiva['tiempo_base'] = tiempo_base_masiva; new_params_masiva['value'] = tiempo_base_masiva
                
                if new_dist_masiva != 'cte':
                    variability_masiva = st.slider("Variabilidad Relativa (scale)", 0.0, 2.0, 0.25, 0.05, key="s_masiva", help="Controla la desviación estándar en relación al tiempo medio.")
                    scale_masiva = tiempo_base_masiva * variability_masiva
                    new_params_masiva['loc'] = tiempo_base_masiva; new_params_masiva['scale'] = scale_masiva
                    if new_dist_masiva in ['lognorm', 'weibull', 'gamma', 'fisk']: # Parámetros de forma para modificación masiva
                        shape_key_masiva = {'lognorm': 's', 'weibull': 'c', 'gamma': 'a', 'fisk': 'c'}[new_dist_masiva]
                        shape_label_masiva = {'lognorm': 'Sigma', 'weibull': 'k', 'gamma': 'alpha', 'fisk': 'c'}[new_dist_masiva]
                        shape_defaults_masiva = {'s': 0.5, 'c': 2.0, 'a': 2.0}
                        new_params_masiva[shape_key_masiva] = st.number_input(f"Parámetro de Forma ({shape_label_masiva})", 0.1, 20.0, shape_defaults_masiva.get(shape_key_masiva, 1.0), 0.1, key=f"p1_masiva")
                        
                st.markdown("#### Vista Previa de la Distribución Propuesta")
                # Lógica para graficar la distribución propuesta
                original_params_for_viz = {'tiempo_base': tiempo_base_masiva}; original_dist_name_for_viz = 'cte' if new_dist_masiva == 'cte' else 'norm'
                if new_dist_masiva != 'cte': 
                    original_params_for_viz = {'loc': tiempo_base_masiva, 'scale': tiempo_base_masiva * 0.25} # Base para la comparación visual
                
                fig_mass_comp = st.session_state.simulator.graficar_comparacion_distribucion( 
                    nombre_dist_original=original_dist_name_for_viz, tupla_params_original=tuple(sorted(original_params_for_viz.items())),
                    nombre_dist_modificada=new_dist_masiva, tupla_params_modificada=tuple(sorted(new_params_masiva.items())),
                    titulo=f"Comparación de la Nueva Distribución ({new_dist_masiva})")
                if fig_mass_comp: st.plotly_chart(fig_mass_comp, use_container_width=True)
                
                col_mass_save, col_mass_reset = st.columns(2)
                with col_mass_save:
                    if st.button("Aplicar Cambios Masivos", key="save_mass_mod", use_container_width=True):
                        if filtered_uids:
                            # Guardar las modificaciones en el estado de la sesión
                            for uid in filtered_uids:
                                st.session_state.modified_times[uid] = {'params': new_params_masiva.copy(), 'distribucion': new_dist_masiva}
                            st.success(f"Aplicados cambios a {len(filtered_uids)} actividades del nivel {nivel_masivo_selected}.")
                            st.rerun()
                        else: st.warning("No hay actividades seleccionadas para aplicar cambios.")
                with col_mass_reset:
                    if st.button("Restablecer todas las actividades de este nivel a su original", key="reset_mass_mod", use_container_width=True):
                        # Resetear las modificaciones en el estado de la sesión
                        count_reset = 0
                        for uid in activities_in_level['unique_id'].tolist():
                            if uid in st.session_state.modified_times: del st.session_state.modified_times[uid]; count_reset += 1
                        st.info(f"Restablecidas {count_reset} actividades del nivel {nivel_masivo_selected} a su configuración original.")
                        st.rerun()

        st.markdown("---")
        with st.expander("Configurar Tiempos de Actividades Individualmente", expanded=False):
            # Lógica de conciliación de actividades (Excel + Usuario)
            activities = all_activities_df[['Nivel', 'Actividad', 'Recurso', 'unique_id']].drop_duplicates().apply(lambda x: f"{x[0]} - {x[1]} ({x[2]}) - {x[3]}", axis=1)
            activity_to_modify_str = st.selectbox("Actividad para modificar", options=activities, key="single_act_mod_sel")
            
            if activity_to_modify_str:
                parts = activity_to_modify_str.split(' - ')
                nivel_mod = parts[0]; act_res_part = parts[1]; uid = parts[2]
                act_rows = all_activities_df[all_activities_df['unique_id'] == uid]
                
                if not act_rows.empty:
                    current_act = act_rows.iloc[0]; params_before = current_act.get('params', {}) or {'tiempo_base': current_act['tiempo_base']}
                    st.markdown("#### Comparación de Distribuciones")
                    original_params_tuple = tuple(sorted(params_before.items()))
                    
                    if uid in st.session_state.modified_times:
                        # Graficar Original vs. Modificada
                        mod_data = st.session_state.modified_times[uid]
                        fig_comp = st.session_state.simulator.graficar_comparacion_distribucion( 
                            nombre_dist_original=current_act.get('Distribucion', 'cte'), tupla_params_original=original_params_tuple,
                            nombre_dist_modificada=mod_data['distribucion'], tupla_params_modificada=tuple(sorted(mod_data['params'].items())),
                            titulo=f"Comparación para: {act_res_part}")
                    else:
                        # Graficar solo Original
                        fig_comp = st.session_state.simulator.graficar_comparacion_distribucion( 
                            nombre_dist_original=current_act.get('Distribucion', 'cte'), tupla_params_original=original_params_tuple,
                            titulo=f"Distribución Original para: {act_res_part}")
                    
                    if fig_comp: st.plotly_chart(fig_comp, use_container_width=True)
                    st.markdown("---")
                    
                    st.markdown("#### Controles de Modificación")
                    dist_options = ['cte', 'norm', 'lognorm', 'weibull', 'gamma', 'fisk', 'rayleigh']
                    current_dist = st.session_state.modified_times.get(uid, {'distribucion': current_act.get('Distribucion', 'cte')})['distribucion'].lower()
                    current_dist_index = dist_options.index(current_dist) if current_dist in dist_options else 0
                    
                    new_dist = st.selectbox("Distribución", dist_options, index=current_dist_index, key=f"d_{uid}")
                    tiempo_base = st.number_input("Tiempo Medio (loc)", 0.01, value=float(current_act.get('tiempo_base', 1.0)), step=0.1, key=f"t_{uid}")
                    variability = st.slider("Variabilidad Relativa (scale)", 0.0, 2.0, 0.25, 0.05, key=f"s_{uid}")
                    
                    scale = tiempo_base * variability; new_params = {'loc': tiempo_base, 'scale': scale, 'tiempo_base': tiempo_base}
                    
                    if new_dist == 'cte': 
                        new_params = {'tiempo_base': tiempo_base, 'value': tiempo_base}
                    elif new_dist in ['lognorm', 'weibull', 'gamma', 'fisk']: # Parámetros de forma para modificación individual
                        shape_key = {'lognorm': 's', 'weibull': 'c', 'gamma': 'a', 'fisk': 'c'}[new_dist]; shape_label = {'lognorm': 'Sigma', 'weibull': 'k', 'gamma': 'alpha', 'fisk': 'c'}[new_dist]
                        shape_defaults = {'s': 0.5, 'c': 2.0, 'a': 2.0}; default_shape_val = shape_defaults.get(shape_key, 1.0)
                        # Usar el valor modificado si existe, sino el valor por defecto
                        if uid in st.session_state.modified_times and shape_key in st.session_state.modified_times[uid]['params']: 
                            default_shape_val = st.session_state.modified_times[uid]['params'][shape_key]
                        new_params[shape_key] = st.number_input(f"Parámetro de Forma ({shape_label})", 0.1, 20.0, default_shape_val, 0.1, key=f"p1_{uid}")
                        
                    col_save, col_reset = st.columns(2)
                    with col_save:
                        if st.button("Guardar Cambios", key=f"save_{uid}", use_container_width=True):
                            st.session_state.modified_times[uid] = {'params': new_params, 'distribucion': new_dist}
                            st.success("Cambios guardados."); st.rerun()
                    with col_reset:
                        if uid in st.session_state.modified_times and st.button("Restaurar Original", key=f"del_{uid}", use_container_width=True):
                            del st.session_state.modified_times[uid]; st.info("Modificación eliminada."); st.rerun()

#################################SELECCIÓN Y EJECUCIÓN DE SIMULACIÓN##########################################################

    st.markdown("---")
    st.subheader("Selección de Frentes y Ejecución")
    
    if st.session_state.simulator.df_frentes is not None:
        niveles_sim = st.session_state.simulator.df_frentes['Nivel'].unique()
        nivel_sel = st.selectbox("Nivel para Simular", niveles_sim, key="nivel_sim")
        
        if nivel_sel:
            frentes = st.session_state.simulator.df_frentes[st.session_state.simulator.df_frentes['Nivel'] == nivel_sel]['Frentes'].unique()
            

            st.markdown("### Parámetros específicos para esta simulación")
            col_params1, col_params2 = st.columns(2)
            with col_params1:
                advance_length = st.number_input("Largo de avance por ciclo (m)", 0.1, 10.0, 2.5, 0.1, key="per_sim_advance")
                num_simulations = st.number_input("Número de simulaciones de avance", 100, 10000, 1000, 100, key="per_sim_num")
            with col_params2:
                time_limit = st.number_input("Tiempo Límite de Ejecución de Obras por Simulación (horas)", 1.0, 8760.0, 160.0, 1.0, key="per_sim_time_limit")
                time_modification_percent = st.number_input("Modificación de tiempo global (%)", -50, 100, 0, 5, help="Afecta a TODAS las actividades de los túneles seleccionados.", key="per_sim_mod_time")
            
            st.markdown("### Configuración de Restricciones y Planes")
            col_plans1, col_plans2 = st.columns(2)
            with col_plans1:
                # Selector de Plan de Obras Civiles
                cw_options = ["Ninguno"] + list(st.session_state.civil_works_plans.keys())
                civil_works_plan_name = st.selectbox("Plan de Obras Civiles para este túnel", options=cw_options, key="cw_plan_sel")
                # Selector de Condición Geológica (filtrado por nivel)
                geol_options = ["Ninguna"] + [name for name, cond in st.session_state.geological_conditions.items() if cond['level'] == nivel_sel]
                geological_condition_name = st.selectbox("Condición Geológica para este túnel", options=geol_options, key="geol_cond_sel")
            with col_plans2:
                # Selector de Plan de Restricción Geográfica
                res_options = ["Ninguno"] + list(st.session_state.restriction_plans.keys())
                restriction_plan_name = st.selectbox("Plan de Restricción Geográfica para este túnel", options=res_options, key="res_plan_sel")
                restriction_radius = st.number_input("Radio de restricciones (m)", 10.0, 200.0, 50.0, 10.0, key="per_sim_res_radius")
                restriction_delay_time = st.number_input("Demora por restricción (horas)", 0.0, 24.0, 4.0, 0.5, key="per_sim_res_delay")

            # Selector de frentes a simular
            frentes_sel = st.multiselect("Frentes a Simular", frentes, default=st.session_state.sim_queue, key="frentes_sel")

            if st.button("Iniciar Simulación", type="primary", use_container_width=True, disabled=not frentes_sel or st.session_state.sim_started):
                st.session_state.sim_queue = list(frentes_sel)
                st.session_state.sim_started = True
                st.session_state.current_sim_index = 0
                st.info(f"Simulación iniciada. Cola de túneles a simular: {', '.join(st.session_state.sim_queue)}.")
                st.rerun()

            if st.session_state.sim_started and st.session_state.current_sim_index < len(st.session_state.sim_queue):
                current_front_name = st.session_state.sim_queue[st.session_state.current_sim_index]
                st.info(f"Listo para simular el siguiente túnel: **{current_front_name}** ({st.session_state.current_sim_index + 1} de {len(st.session_state.sim_queue)})")

                if st.button("Simular Siguiente Túnel", type="primary", use_container_width=True):
                    with st.spinner(f"Ejecutando simulación para el frente {current_front_name}..."):
                        # Obtener información del frente actual
                        front_info = st.session_state.simulator.df_frentes[(st.session_state.simulator.df_frentes['Frentes'] == current_front_name) & (st.session_state.simulator.df_frentes['Nivel'] == nivel_sel)].iloc[0]
                        
                        # Obtener los planes y condiciones seleccionadas (o None)
                        civil_works_plan = st.session_state.civil_works_plans.get(civil_works_plan_name)
                        geological_condition = st.session_state.geological_conditions.get(geological_condition_name)
                        restriction_plan = st.session_state.restriction_plans.get(restriction_plan_name)

                        # 1. Ejecutar la simulación de avance (Monte Carlo)
                        results_df_front = st.session_state.simulator.simular_avance_tunel( 
                            info_frente=front_info, largo_avance=advance_length, num_simulaciones=num_simulations,
                            limite_tiempo=time_limit, radio_restriccion=restriction_radius, tiempo_demora_restriccion=restriction_delay_time,
                            porcentaje_modificacion_tiempo=time_modification_percent, tiempos_modificados=st.session_state.modified_times,
                            ajustes_recursos=st.session_state.resource_adjustments, demoras_manuales=st.session_state.manual_delays,
                            actividades_usuario=st.session_state.user_defined_activities, plan_obras_civiles=civil_works_plan,
                            condicion_geologica=geological_condition, parametros_averias_equipo=st.session_state.equipment_breakdowns,
                            demora_interfase_tunel=st.session_state.tunnel_interfase_delays.get(current_front_name, 0.0),
                            plan_restriccion=restriction_plan
                        )

                        # 2. Simular distribuciones de actividad (para boxplot)
                        per_tunnel_data_key = f"{nivel_sel}-{current_front_name}"
                        activity_dist_df_new = st.session_state.simulator.simular_distribuciones_actividad( 
                            nivel=nivel_sel, num_muestras=num_simulations, porcentaje_modificacion_tiempo=time_modification_percent, 
                            ajustes_recursos=st.session_state.resource_adjustments, tiempos_modificados=st.session_state.modified_times,
                            actividades_usuario=st.session_state.user_defined_activities, condicion_geologica=geological_condition
                        )
                        
                        # 3. Generar datos para Gantt probabilística
                        gantt_df_new, delay_df_new = st.session_state.simulator.generar_datos_gantt( 
                            nivel=nivel_sel, num_muestras=num_simulations, porcentaje_modificacion_tiempo=time_modification_percent,
                            ajustes_recursos=st.session_state.resource_adjustments, tiempos_modificados=st.session_state.modified_times,
                            demoras_manuales=st.session_state.manual_delays, actividades_usuario=st.session_state.user_defined_activities,
                            plan_obras_civiles=civil_works_plan, condicion_geologica=geological_condition,
                            parametros_averias_equipo=st.session_state.equipment_breakdowns
                        )
                        
                        # Almacenar datos detallados por túnel
                        st.session_state.per_tunnel_data[per_tunnel_data_key] = {
                            'results': results_df_front,
                            'activity_dist_df': activity_dist_df_new,
                            'gantt_df': gantt_df_new,
                            'delay_df': delay_df_new
                        }
                        
                        # Consolidar resultados globales
                        st.session_state.sim_results = pd.concat([st.session_state.sim_results, results_df_front], ignore_index=True)
                        st.session_state.current_sim_index += 1
                        st.success(f"Simulación para el túnel '{current_front_name}' completada.")
                        st.rerun()

            elif st.session_state.sim_started and st.session_state.current_sim_index >= len(st.session_state.sim_queue):
                st.success("¡Todos los túneles han sido simulados!")
                st.session_state.sim_started = False
                st.session_state.sim_queue = []
                st.rerun()
            else:
                st.info("Seleccione los frentes y presione 'Iniciar Simulación' para comenzar.")

            if st.session_state.sim_started and st.button("Reiniciar Simulación", use_container_width=True):
                # Limpiar todos los resultados
                st.session_state.sim_queue = []
                st.session_state.sim_results = pd.DataFrame()
                st.session_state.current_sim_index = 0
                st.session_state.sim_started = False
                st.session_state.per_tunnel_data = {}
                st.success("Simulación reiniciada. Puede configurar nuevos parámetros.")
                st.rerun()

#################################ANÁLISIS DE RESULTADOS##########################################################

    if not st.session_state.sim_results.empty:
        st.markdown("---")
        st.header("Resumen de Resultados de la Simulación de Avance")

        # Agregación de resultados (Avance y Tiempos)
        summary = st.session_state.sim_results.groupby(['Nivel', 'Frentes']).agg(
            Avance_Promedio_m=('actual_distance', 'mean'), # Avance promedio
            Avance_P10_m=('actual_distance', lambda x: x.quantile(0.1)), # Avance P10
            Avance_P50_m=('actual_distance', lambda x: x.quantile(0.5)), # Avance P50
            Avance_P90_m=('actual_distance', lambda x: x.quantile(0.9)), # Avance P90
            Eficiencia_Media_pct=('efficiency', lambda x: x.mean() * 100), # Eficiencia media
            Ciclos_Completados_Promedio=('completed_cycles', 'mean'), # Ciclos completados promedio
            Tiempo_Ciclo_Promedio_hrs=('avg_cycle_time', 'mean'), # Tiempo de ciclo promedio
            Tiempo_Total_Promedio_hrs=('total_sim_time', 'mean') # Tiempo total promedio
        ).reset_index()

        # Unir con la información de frentes para mantener la consistencia
        summary = pd.merge(st.session_state.simulator.df_frentes, summary, on=['Nivel', 'Frentes'], how='right')


        st.dataframe(summary[[
            'Nivel', 'Frentes', 'Avance_Promedio_m',
            'Avance_P10_m', 'Avance_P50_m', 'Avance_P90_m',
            'Eficiencia_Media_pct', 'Ciclos_Completados_Promedio', 'Tiempo_Ciclo_Promedio_hrs',
        ]].round(2), use_container_width=True) # Mostrar tabla de resumen

        res_col1, res_col2 = st.columns(2)
        with res_col1:
            fig_hist = px.histogram(st.session_state.sim_results, x='actual_distance', color='Frentes',
                                    barmode='overlay', title='Distribución de Distancias Finales por Frente',
                                    labels={'actual_distance': 'Distancia Final (m)'})
            st.plotly_chart(fig_hist, use_container_width=True) # Histograma de avance
        with res_col2:
            fig_scatter = px.scatter(st.session_state.sim_results, x='avg_cycle_time', y='actual_distance', color='Frentes',
                                     title='Distancia Lograda vs. Tiempo de Ciclo Promedio',
                                     labels={'actual_distance': 'Distancia Final (m)',
                                             'avg_cycle_time': 'Tiempo Promedio Ciclo (hrs)'})
            st.plotly_chart(fig_scatter, use_container_width=True) # Scatter de tiempo vs distancia


        st.markdown("---")
        st.header("Análisis Detallado por Túnel")
        simulated_tunnels = list(st.session_state.per_tunnel_data.keys())
        if simulated_tunnels:
            # Selector para ver el detalle de un túnel específico
            selected_tunnel_key = st.selectbox("Seleccione un túnel para ver el detalle", options=simulated_tunnels, index=len(simulated_tunnels)-1)
            
            if selected_tunnel_key in st.session_state.per_tunnel_data:
                tunnel_data = st.session_state.per_tunnel_data[selected_tunnel_key]

                st.subheader(f"Resultados del ciclo para: {selected_tunnel_key}")
                
                activity_dist_df = tunnel_data.get('activity_dist_df')
                if not activity_dist_df.empty:
                    # Resumen estadístico de las actividades del ciclo
                    summary_act = activity_dist_df.groupby(['Actividad', 'Recurso'])['Duracion'].agg(
                        Optimista_P10=lambda x: x.quantile(0.1),
                        Mas_Probable_P50=lambda x: x.quantile(0.5),
                        Pesimista_P90=lambda x: x.quantile(0.9),
                        Esperanza_Media='mean'
                    ).reset_index()
                    st.dataframe(summary_act.round(2), use_container_width=True) # Resumen de actividades

                    fig_dist = graficar_distribuciones_actividad(activity_dist_df) 
                    st.plotly_chart(fig_dist, use_container_width=True) # Boxplot de distribuciones
                
                gantt_df = tunnel_data.get('gantt_df')
                delay_df = tunnel_data.get('delay_df')
                if not gantt_df.empty:
                    st.subheader("Carta Gantt Probabilística del Ciclo")
                    fig_gantt = graficar_gantt_probabilistica(gantt_df, delay_df) 
                    st.plotly_chart(fig_gantt, use_container_width=True) # Gráfico Gantt



if __name__ == "__main__":
    main()
