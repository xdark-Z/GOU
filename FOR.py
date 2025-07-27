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

warnings.filterwarnings('ignore')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CUDA - STREAMLIT CACHÉ

@st.cache_data
def get_resources_for_level(_df, level):
    """Función cacheada para obtener recursos únicos para un nivel."""
    if _df is None or level is None:
        return []
    resources = _df[_df['Nivel'] == level]['Recurso'].dropna().unique()
    return [r for r in resources if r and pd.notna(r)]

@st.cache_data
def generate_fronts_visualization(_fronts_df, _restrictions, restriction_radius, nivel_selected):
    """Genera la figura de visualización de frentes. Cacheada para no recalcular."""
    fronts_for_viz = _fronts_df[_fronts_df['Nivel'] == nivel_selected]
    fig = go.Figure()
    for _, front_info in fronts_for_viz.iterrows():
        fig.add_trace(go.Scatter(x=[front_info['Xi'], front_info['Xf']], y=[front_info['Yi'], front_info['Yf']],
                                 mode='lines+markers', name=f"Frente {front_info['Frentes']}",
                                 line=dict(width=4), marker=dict(size=10)))

    for restriction_type, df in _restrictions.items():
        if df is not None:
            for _, restriction in df.iterrows():
                fig.add_shape(type="circle", xref="x", yref="y",
                              x0=restriction['X'] - restriction_radius, y0=restriction['Y'] - restriction_radius,
                              x1=restriction['X'] + restriction_radius, y1=restriction['Y'] + restriction_radius,
                              line_color="red", line_dash="dash", opacity=0.5)
                restriction_name = f"{restriction_type}: {restriction.get('Nombre', 'Sin Nombre')}"
                fig.add_trace(go.Scatter(x=[restriction['X']], y=[restriction['Y']],
                                         mode='markers', name=restriction_name,
                                         marker=dict(color='red', size=8, symbol='x')))

    fig.update_layout(title=f'Ubicación del Frente y Restricciones para Nivel {nivel_selected}',
                      xaxis_title='Coordenada X', yaxis_title='Coordenada Y', height=600, showlegend=True)
    return fig

#PLOTLY GRAFICOS + GANTT 

def plot_activity_distributions(df):
    """Crea un box plot para visualizar la distribución de tiempos de cada actividad."""
    fig = px.box(df, x="Duracion", y="Actividad", color="Recurso",
                 title="Distribución de Tiempos de Actividad por Ciclo",
                 labels={"Duracion": "Duración (horas)", "Actividad": "Actividad"},
                 orientation='h')
    fig.update_traces(quartilemethod="linear") # Percentiles P10 y P90 para los bigotes [CRITERIOS]
    fig.update_layout(template='plotly_white', yaxis={'categoryorder':'total ascending'})
    return fig

def plot_probabilistic_gantt(gantt_df, delay_df=None):
    """
    Crea una carta Gantt probabilística. Ahora acepta un dataframe adicional
    para visualizar las demoras por cambios de equipo en rojo.
    """
    if gantt_df.empty:
        return go.Figure()

    display_dfs = [gantt_df.assign(type='activity')]
    if delay_df is not None and not delay_df.empty:
        display_dfs.append(delay_df.assign(type='delay'))

    combined_df = pd.concat(display_dfs, ignore_index=True)
    combined_df = combined_df.sort_values(by='Start_p50', ascending=False)
    y_axis_labels = combined_df['Actividad']

    fig = go.Figure()

    activity_df = combined_df[combined_df['type'] == 'activity']
    fig.add_trace(go.Bar(
        y=activity_df['Actividad'],
        x=activity_df['Duration_p90'], base=activity_df['Start_p10'],
        orientation='h', name='Rango Actividad (P10-P90)',
        marker=dict(color='rgba(255, 165, 0, 0.5)', line_width=0),
        hoverinfo='none'
    ))
    fig.add_trace(go.Bar(
        y=activity_df['Actividad'],
        x=activity_df['Duration_p50'], base=activity_df['Start_p50'],
        orientation='h', name='Duración Probable (P50)',
        marker=dict(color='rgba(26, 118, 255, 0.8)'),
        text=activity_df['Duration_p50'].apply(lambda x: f'{x:.2f}h'),
        textposition='inside', insidetextanchor='middle'
    ))

    if delay_df is not None and not delay_df.empty:
        delay_plot_df = combined_df[combined_df['type'] == 'delay']
        fig.add_trace(go.Bar(
            y=delay_plot_df['Actividad'],
            x=delay_plot_df['Duration_p50'], base=delay_plot_df['Start_p50'],
            orientation='h', name='Demora por Cambio (P50)',
            marker=dict(color='rgba(255, 0, 0, 0.8)'),
            text=delay_plot_df['Duration_p50'].apply(lambda x: f'{x:.2f}h'),
            textposition='inside', insidetextanchor='middle'
        ))

    total_p90 = combined_df['Finish_p90'].max()
    fig.update_layout(
        title='Carta Gantt Probabilística de un Ciclo (con Demoras por Cambios)',
        xaxis_title=f'Tiempo Acumulado (horas) - Fin Pesimista (P90): {total_p90:.2f}h',
        yaxis_title='Actividades y Demoras',
        barmode='stack', template='plotly_white',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=len(y_axis_labels) * 35 + 200,
        yaxis=dict(categoryorder='array', categoryarray=y_axis_labels.tolist()), # Usar el orden combinado
        xaxis=dict(range=[0, total_p90 * 1.05])
    )
    return fig


class TunnelSimulator:
    def __init__(self):
        self.activities_df = None
        self.resources_df = None
        self.fronts_df = None
        self.restrictions = {
            'frente_hundimiento': None,
            'front_index': None,
            'obras_civiles': None
        }

    def load_data_from_excel(self, excel_file):
        """Cargar datos desde archivo Excel"""
        try:
            self.activities_df = pd.read_excel(excel_file, sheet_name=0)
            self.resources_df = pd.read_excel(excel_file, sheet_name=1)
            self.fronts_df = pd.read_excel(excel_file, sheet_name=2)
            self.activities_df = self.activities_df.fillna('')
            self.process_activity_data()
            self._parse_resource_data()
            return True
        except Exception as e:
            st.error(f"Error al cargar archivo de datos: {str(e)}")
            return False

    def _parse_resource_data(self):
        """
        Procesa el DataFrame de recursos para interpretar turnos, cantidades y
        la nueva columna 'Demora_Cambio'.
        """
        if self.resources_df is None:
            return

        def parse_range(range_str):
            """Función interna para convertir 'inicio-fin' en (inicio, fin)."""
            if isinstance(range_str, str) and '-' in range_str:
                try:
                    parts = [float(p.strip().replace(',', '.')) for p in range_str.split('-')]
                    return min(parts), max(parts)
                except (ValueError, TypeError):
                    return 0, 0
            try:
                val = float(str(range_str).strip().replace(',', '.'))
                return val, val
            except (ValueError, TypeError):
                return 0, 0

        if 'Turno_Cambio' in self.resources_df.columns:
            self.resources_df[['turno_start', 'turno_end']] = self.resources_df['Turno_Cambio'].apply(lambda x: pd.Series(parse_range(x)))
        else:
            self.resources_df['turno_start'] = 0
            self.resources_df['turno_end'] = 0

        if 'Demora_Cambio' in self.resources_df.columns:
            self.resources_df['Demora_Cambio'] = pd.to_numeric(self.resources_df['Demora_Cambio'], errors='coerce').fillna(0)
        else:
            self.resources_df['Demora_Cambio'] = 0

        if 'Cambio' in self.resources_df.columns:
            self.resources_df['Cambio'] = self.resources_df['Cambio'].fillna(False).astype(bool)
        else:
            self.resources_df['Cambio'] = False

        if 'Cantidad' in self.resources_df.columns:
            self.resources_df['Cantidad'] = pd.to_numeric(self.resources_df['Cantidad'].astype(str).str.replace(',', '.'), errors='coerce').fillna(1)
        else:
            self.resources_df['Cantidad'] = 1


    def process_activity_data(self):
        """Procesar datos de actividades para asegurar formato consistente"""
        for idx, row in self.activities_df.iterrows():
            unique_id = f"{row['Nivel']}_{row.get('Secuencia', idx)}_{row['Actividad']}_{row.get('Recurso', 'default')}_{idx}"
            self.activities_df.at[idx, 'unique_id'] = unique_id
            if pd.isna(row['Distribucion']) or row['Distribucion'] == '':
                self.activities_df.at[idx, 'Distribucion'] = 'cte'
            if isinstance(row['Tiempo'], str) and '{' in str(row['Tiempo']):
                try:
                    params = eval(row['Tiempo'])
                    self.activities_df.at[idx, 'params'] = params
                    if 'loc' in params: self.activities_df.at[idx, 'tiempo_base'] = abs(params['loc'])
                    elif 'scale' in params: self.activities_df.at[idx, 'tiempo_base'] = abs(params['scale'])
                    else: self.activities_df.at[idx, 'tiempo_base'] = 1.0
                except:
                    self.activities_df.at[idx, 'params'], self.activities_df.at[idx, 'tiempo_base'] = {}, 1.0
            else:
                self.activities_df.at[idx, 'params'] = {}
                try: self.activities_df.at[idx, 'tiempo_base'] = float(row['Tiempo']) if pd.notna(row['Tiempo']) else 1.0
                except: self.activities_df.at[idx, 'tiempo_base'] = 1.0

    def load_restrictions_from_excel(self, excel_file):
        """Cargar restricciones desde archivo Excel"""
        try:
            self.restrictions['frente_hundimiento'] = pd.read_excel(excel_file, sheet_name='FrenteHundimiento')
            self.restrictions['front_index'] = pd.read_excel(excel_file, sheet_name='Restriccion')
            self.restrictions['obras_civiles'] = pd.read_excel(excel_file, sheet_name='ObrasCiviles')
            return True
        except Exception as e:
            st.error(f"Error al cargar archivo de restricciones: {str(e)}")
            return False

    def sample_from_distribution(self, dist_name, params, size=1):
        """Generar muestras de distribuciones específicas."""
        try:
            dist_name = dist_name.lower()
            if dist_name in ['constant', 'cte', 'constante']:
                return torch.full((int(size),), params.get('value', 1.0), device=device)

            dist_map = {
                'norm': (stats.norm.rvs, {}), 'lognorm': (stats.lognorm.rvs, {'s': params.get('s', 0.5)}),
                'weibull': (stats.weibull_min.rvs, {'c': params.get('c', 1.0)}), 'gamma': (stats.gamma.rvs, {'a': params.get('a', 1.99)}),
                'fisk': (stats.fisk.rvs, {'c': params.get('c', 1.0)}), 'rayleigh': (stats.rayleigh.rvs, {}),
                'foldcauchy': (stats.foldnorm.rvs, {'c': params.get('c', 1.0)}), 'foldnorm': (stats.foldnorm.rvs, {'c': params.get('c', 1.0)}),
                'ncx2': (stats.ncx2.rvs, {'df': params.get('df', 1), 'nc': params.get('nc', 1)})
            }

            if dist_name in dist_map:
                dist_func, shape_params = dist_map[dist_name]
                kwargs = {'loc': params.get('loc', 0), 'scale': params.get('scale', 1), 'size': int(size), **shape_params}
                samples = np.abs(dist_func(**kwargs))
                return torch.tensor(samples, device=device, dtype=torch.float32).clamp(min=0)
            else:
                dist = torch.distributions.Normal(torch.tensor(params.get('loc', 0.0), device=device), torch.tensor(params.get('scale', 1.0), device=device))
                return dist.sample((int(size),)).clamp(min=0)
        except Exception as e:
            st.warning(f"Error en distribución {dist_name}: {e}. Usando valor constante.")
            return torch.full((int(size),), params.get('tiempo_base', 1.0), device=device)

    @st.cache_data
    def plot_comparison_distribution(_self, original_dist_name, original_params_tuple,
                                   modified_dist_name=None, modified_params_tuple=None, title="Distribución"):
        """Crea un gráfico comparando la distribución original con la modificada usando Plotly."""
        fig = go.Figure()
        max_x_range = 0

        def _plot_single_dist_go(fig_obj, dist_name, params_tuple, label, color, dash):
            params = dict(params_tuple)
            dist_name = dist_name.lower()
            scale = params.get('scale', 0) if dist_name != 'cte' else 0
            loc = params.get('loc', params.get('value', params.get('tiempo_base', 1)))
            current_max_x = max(10, loc + 5 * scale)

            if dist_name in ['constant', 'cte', 'constante']:
                const_val = params.get('value', loc)
                fig_obj.add_vline(x=const_val, line_width=2.5, line_dash=dash, line_color=color,
                                  annotation_text=f"{label} (Valor={const_val:.2f})", annotation_position="top right")
                return current_max_x

            x = np.linspace(0, current_max_x, 1000)
            dist_map_pdf = {
                'norm': (stats.norm.pdf, {}), 'lognorm': (stats.lognorm.pdf, {'s': params.get('s', 0.5)}),
                'weibull': (stats.weibull_min.pdf, {'c': params.get('c', 1.0)}), 'gamma': (stats.gamma.pdf, {'a': params.get('a', 1.99)}),
                'fisk': (stats.fisk.pdf, {'c': params.get('c', 1.0)}), 'rayleigh': (stats.rayleigh.pdf, {}),
                'foldcauchy': (stats.foldnorm.pdf, {'c': params.get('c', 1.0)}), 'foldnorm': (stats.foldnorm.pdf, {'c': params.get('c', 1.0)}),
                'ncx2': (stats.ncx2.pdf, {'df': params.get('df', 1), 'nc': params.get('nc', 1)})
            }

            y = np.zeros_like(x)
            if dist_name in dist_map_pdf:
                pdf_func, shape_params = dist_map_pdf[dist_name]
                pdf_scale = params.get('scale', 1e-9)
                if pdf_scale <= 0: pdf_scale = 1e-9
                kwargs = {'loc': params.get('loc', 0), 'scale': pdf_scale, **shape_params}
                y = pdf_func(x, **kwargs)

            fig_obj.add_trace(go.Scatter(
                x=x, y=y, mode='lines', name=f"{label} ({dist_name})",
                line=dict(color=color, width=2, dash=dash)
            ))
            fig_obj.add_trace(go.Scatter(
                x=x, y=y, fill='tozeroy', mode='none',
                fillcolor=color, opacity=0.2,
                showlegend=False,
                hoverinfo='none'
            ))
            return current_max_x

        max_x1 = _plot_single_dist_go(fig, original_dist_name, original_params_tuple, "Original", "blue", "solid")
        max_x_range = max(max_x_range, max_x1)

        if modified_dist_name and modified_params_tuple:
            max_x2 = _plot_single_dist_go(fig, modified_dist_name, modified_params_tuple, "Modificada", "red", "dash")
            max_x_range = max(max_x_range, max_x2)

        fig.update_layout(
            title=title,
            xaxis_title='Tiempo (horas)',
            yaxis_title='Densidad de Probabilidad',
            xaxis=dict(range=[0, max_x_range]),
            yaxis=dict(range=[0, None]),
            legend_title_text='Distribución',
            template='plotly_white'
        )
        return fig

    def check_restrictions(self, x, y, restriction_radius):
        """Verificar si una posición viola restricciones operacionales."""
        total_delay = 0
        for df in self.restrictions.values():
            if df is not None:
                for _, restriction in df.iterrows():
                    if np.sqrt((x - restriction['X'])**2 + (y - restriction['Y'])**2) < restriction_radius:
                        total_delay += restriction.get('Demora', 1)
        return total_delay

    def check_potential_collisions(self, fronts_df, restrictions, radius):
        """Revisa si los segmentos de línea de los frentes chocan con los radios de las restricciones."""
        colliding_fronts = {}
        total_collisions = 0
        all_restrictions = [r for df in restrictions.values() if df is not None for _, r in df.iterrows()]
        if not all_restrictions: return {}, 0

        for _, front in fronts_df.iterrows():
            p1, p2 = np.array([front['Xi'], front['Yi']]), np.array([front['Xf'], front['Yf']])
            line_vec, line_len_sq = p2 - p1, np.dot(p2 - p1, p2 - p1)
            for r in all_restrictions:
                c = np.array([r['X'], r['Y']])
                dist_sq = np.sum((p1 - c)**2) if line_len_sq == 0 else np.sum((p1 + max(0, min(1, np.dot(c - p1, line_vec) / line_len_sq)) * line_vec - c)**2)
                if dist_sq < radius**2:
                    front_name = front['Frentes']
                    colliding_fronts.setdefault(front_name, []).append(r.get('Nombre', 'Sin Nombre'))
                    total_collisions += 1
        return colliding_fronts, total_collisions

    def _get_combined_delays(self, nivel, manual_delays):
        """Helper para unificar demoras de Excel y manuales."""
        all_delays = []
        excel_delays = self.resources_df[(self.resources_df['Nivel'] == nivel) & (self.resources_df['Cambio'] == True) & (self.resources_df['Demora_Cambio'] > 0)]
        for _, delay_info in excel_delays.iterrows():
            all_delays.append({
                "resource": delay_info['Recurso'],
                "start_hour": delay_info['turno_start'],
                "duration": delay_info['Demora_Cambio']
            })
        for manual_delay in manual_delays:
            if manual_delay.get('nivel') == nivel:
                 all_delays.append({
                    "resource": manual_delay['recurso'],
                    "start_hour": manual_delay['turno'][0],
                    "duration": manual_delay['demora']
                })
        return all_delays

    def simulate_tunnel_advance(self, front_info, advance_length, num_simulations,
                              time_limit, restriction_radius, time_modification_percent,
                              restriction_delay_time, modified_times, resource_adjustments,
                              manual_delays=[]):
        """
        Simulación Monte Carlo del avance de un frente específico.
        AHORA DEVUELVE MÁS MÉTRICAS para un análisis más claro.
        """
        nivel = front_info['Nivel']
        planned_distance = np.sqrt((front_info['Xf'] - front_info['Xi'])**2 + (front_info['Yf'] - front_info['Yi'])**2)
        direction_vector = np.array([front_info['Xf'] - front_info['Xi'], front_info['Yf'] - front_info['Yi']]) / (planned_distance or 1)
        results, nivel_activities = [], self.activities_df[self.activities_df['Nivel'] == nivel]

        all_delays = self._get_combined_delays(nivel, manual_delays)

        for sim in range(num_simulations):
            total_sim_time, total_distance, cycle_count = 0.0, 0.0, 0
            current_x, current_y = front_info['Xi'], front_info['Yi']
            applied_delays_this_run = set()

            while total_sim_time < time_limit:
                cycle_delay = 0.0
                for delay_idx, delay in enumerate(all_delays):
                    delay_id = f"delay_{delay_idx}"
                    # Check if current time has passed the delay start and it hasn't been applied in this 24h cycle
                    # The modulo 24 ensures we check for delays within each day
                    if (total_sim_time % 24) >= delay['start_hour'] and delay_id not in applied_delays_this_run:
                        cycle_delay += delay['duration']
                        applied_delays_this_run.add(delay_id)
                # Reset applied_delays_this_run for the next 24-hour cycle
                if (total_sim_time + cycle_delay) // 24 > total_sim_time // 24: # If new day starts
                    applied_delays_this_run = set()

                restriction_delay = self.check_restrictions(current_x, current_y, restriction_radius) * restriction_delay_time
                cycle_time = cycle_delay + restriction_delay

    # Suma tiempo por ciclo #
                for _, activity in nivel_activities.iterrows():
                    uid, resource = activity['unique_id'], activity['Recurso']
                    if uid in modified_times:
                        params, dist_name = modified_times[uid]['params'].copy(), modified_times[uid]['distribucion']
                    else:
                        params, dist_name = (activity.get('params', {}).copy() or {'tiempo_base': activity['tiempo_base']}), activity['Distribucion']

                    res_mod = resource_adjustments.get(nivel, {}).get(resource, 0)
                    mod_factor = 1.0 + ((time_modification_percent + res_mod) / 100.0)

                    temp_params = params.copy()
                    for key in ['tiempo_base', 'value', 'loc', 'scale']:
                        if key in temp_params: temp_params[key] *= mod_factor
                    # Special handling for shape parameters, they should not be scaled by mod_factor
                    # 's' for lognorm, 'c' for weibull/fisk, 'a' for gamma
                    if dist_name.lower() == 'lognorm' and 's' in params:
                        temp_params['s'] = params['s']
                    elif dist_name.lower() in ['weibull', 'fisk'] and 'c' in params:
                        temp_params['c'] = params['c']
                    elif dist_name.lower() == 'gamma' and 'a' in params:
                        temp_params['a'] = params['a']

                    cycle_time += self.sample_from_distribution(dist_name, temp_params, 1)[0].item()

                if total_sim_time + cycle_time > time_limit:
                    break

                total_sim_time += cycle_time
                total_distance += advance_length
                cycle_count += 1
                current_x += direction_vector[0] * advance_length
                current_y += direction_vector[1] * advance_length

            
            results.append({     #SAVE RESULTADOS [EVITAR PERDER INFORMACIÓN]
                'planned_distance': planned_distance,
                'actual_distance': min(total_distance, planned_distance),
                'total_sim_time': total_sim_time,
                'completed_cycles': cycle_count,
                'avg_cycle_time': total_sim_time / cycle_count if cycle_count > 0 else 0,
                'efficiency': min(total_distance, planned_distance) / planned_distance if planned_distance > 0 else 0
            })
        return pd.DataFrame(results)

    def simulate_activity_distributions(self, nivel, num_samples, time_modification_percent, resource_adjustments, modified_times):
        """Simula las duraciones de cada actividad para análisis estadístico."""
        nivel_activities = self.activities_df[self.activities_df['Nivel'] == nivel]
        activity_data = []

        for _, activity in nivel_activities.iterrows():
            uid, resource = activity['unique_id'], activity['Recurso']

            if uid in modified_times:
                params, dist_name = modified_times[uid]['params'].copy(), modified_times[uid]['distribucion']
            else:
                params, dist_name = (activity.get('params', {}).copy() or {'tiempo_base': activity['tiempo_base']}), activity['Distribucion']

            res_mod = resource_adjustments.get(nivel, {}).get(resource, 0)
            mod_factor = 1.0 + ((time_modification_percent + res_mod) / 100.0)
            for key in ['tiempo_base', 'value', 'loc', 'scale']:
                if key in params: params[key] *= mod_factor
            # Special handling for shape parameters, they should not be scaled by mod_factor
            if dist_name.lower() == 'lognorm' and 's' in params:
                params['s'] = activity.get('params', {}).get('s', 0.5) if uid not in modified_times else modified_times[uid]['params'].get('s', 0.5)
            elif dist_name.lower() in ['weibull', 'fisk'] and 'c' in params:
                params['c'] = activity.get('params', {}).get('c', 1.0) if uid not in modified_times else modified_times[uid]['params'].get('c', 1.0)
            elif dist_name.lower() == 'gamma' and 'a' in params:
                params['a'] = activity.get('params', {}).get('a', 1.99) if uid not in modified_times else modified_times[uid]['params'].get('a', 1.99)


            samples = self.sample_from_distribution(dist_name, params, size=num_samples)

            for sample in samples:
                activity_data.append({'Actividad': activity['Actividad'], 'Recurso': activity['Recurso'], 'Duracion': sample.item()})

        return pd.DataFrame(activity_data)

    def generate_gantt_data(self, nivel, num_samples, time_modification_percent,
                          resource_adjustments, modified_times,
                          manual_delays=[]):
        """
        Genera datos para la Gantt, ahora identificando y almacenando demoras como eventos separados.
        """
        nivel_activities = self.activities_df[self.activities_df['Nivel'] == nivel].sort_values(by='Secuencia').reset_index()
        all_delays = self._get_combined_delays(nivel, manual_delays)

        activity_results = {act['unique_id']: {'starts': [], 'finishes': [], 'durations': []} for _, act in nivel_activities.iterrows()}
        delay_results = {f"delay_{i}": {'starts': [], 'finishes': [], 'durations': [], 'label': f"Cambio: {d['resource']}"} for i, d in enumerate(all_delays)}

        for _ in range(num_samples):
            current_cycle_time = 0.0
            applied_delays_this_run = set()

            for i in range(len(nivel_activities) + 1): # Bucle extra para chequear demoras al final [PORSILASMOSCAS :)]
                # --- Aplicar Demoras Programadas ---
                for delay_idx, delay in enumerate(all_delays):
                    delay_id = f"delay_{delay_idx}"
                    if current_cycle_time % 24 >= delay['start_hour'] and delay_id not in applied_delays_this_run:
                        delay_start_time = current_cycle_time
                        delay_duration = delay['duration']

                        delay_results[delay_id]['starts'].append(delay_start_time)
                        delay_results[delay_id]['durations'].append(delay_duration)
                        current_cycle_time += delay_duration
                        delay_results[delay_id]['finishes'].append(current_cycle_time)
                        applied_delays_this_run.add(delay_id)
                
                # Reset applied_delays_this_run for the next 24-hour cycle
                if current_cycle_time // 24 > (current_cycle_time - (delay_duration if 'delay_duration' in locals() else 0)) // 24:
                    applied_delays_this_run = set()


                if i < len(nivel_activities):
                    activity = nivel_activities.iloc[i]
                    uid, resource = activity['unique_id'], activity['Recurso']
                    start_time = current_cycle_time

                    if uid in modified_times:
                        params, dist_name = modified_times[uid]['params'].copy(), modified_times[uid]['distribucion']
                    else:
                        params, dist_name = (activity.get('params', {}).copy() or {'tiempo_base': activity['tiempo_base']}), activity['Distribucion']

                    res_mod = resource_adjustments.get(nivel, {}).get(resource, 0)
                    mod_factor = 1.0 + ((time_modification_percent + res_mod) / 100.0)

                    temp_params = params.copy()
                    for key in ['tiempo_base', 'value', 'loc', 'scale']:
                        if key in temp_params: temp_params[key] *= mod_factor
                    # Special handling for shape parameters, they should not be scaled by mod_factor
                    if dist_name.lower() == 'lognorm' and 's' in params:
                        temp_params['s'] = params['s']
                    elif dist_name.lower() in ['weibull', 'fisk'] and 'c' in params:
                        temp_params['c'] = params['c']
                    elif dist_name.lower() == 'gamma' and 'a' in params:
                        temp_params['a'] = params['a']

                    duration = self.sample_from_distribution(dist_name, temp_params, size=1)[0].item()

                    activity_results[uid]['starts'].append(start_time)
                    activity_results[uid]['durations'].append(duration)
                    current_cycle_time += duration
                    activity_results[uid]['finishes'].append(current_cycle_time)

        gantt_rows = []
        for _, activity in nivel_activities.iterrows():
            uid = activity['unique_id']
            if not activity_results[uid]['starts']: continue
            gantt_rows.append({
                'Actividad': f"{activity['Actividad']} ({activity['Recurso'] or 'N/A'})",
                'Start_p10': np.percentile(activity_results[uid]['starts'], 10), 'Finish_p10': np.percentile(activity_results[uid]['finishes'], 10),
                'Start_p50': np.percentile(activity_results[uid]['starts'], 50), 'Finish_p50': np.percentile(activity_results[uid]['finishes'], 50),
                'Start_p90': np.percentile(activity_results[uid]['starts'], 90), 'Finish_p90': np.percentile(activity_results[uid]['finishes'], 90),
                'Duration_p10': np.percentile(activity_results[uid]['durations'], 10),
                'Duration_p50': np.percentile(activity_results[uid]['durations'], 50),
                'Duration_p90': np.percentile(activity_results[uid]['durations'], 90),
            })

        delay_rows = []
        for delay_id, results in delay_results.items():
            if results['starts']:
                delay_rows.append({
                    'Actividad': results['label'],
                    'Start_p10': np.percentile(results['starts'], 10), 'Finish_p10': np.percentile(results['finishes'], 10),
                    'Start_p50': np.percentile(results['starts'], 50), 'Finish_p50': np.percentile(results['finishes'], 50),
                    'Start_p90': np.percentile(results['starts'], 90), 'Finish_p90': np.percentile(results['finishes'], 90),
                    'Duration_p10': np.percentile(results['durations'], 10),
                    'Duration_p50': np.percentile(results['durations'], 50),
                    'Duration_p90': np.percentile(results['durations'], 90),
                })

        return pd.DataFrame(gantt_rows), pd.DataFrame(delay_rows)

def main():
    st.set_page_config(page_title="Simulador de Avance de Túneles", layout="wide")
    st.title("🚇 Simulador de Avance de Túneles - Monte Carlo")

    if 'simulator' not in st.session_state: st.session_state.simulator = TunnelSimulator()
    if 'modified_times' not in st.session_state: st.session_state.modified_times = {}
    if 'resource_adjustments' not in st.session_state: st.session_state.resource_adjustments = {}
    if 'manual_delays' not in st.session_state: st.session_state.manual_delays = []
    if 'recursos_file_name' not in st.session_state: st.session_state.recursos_file_name = None
    if 'restricciones_file_name' not in st.session_state: st.session_state.restricciones_file_name = None

    with st.sidebar:
        st.header("⚙️ Configuración Global")
        st.subheader("Parámetros de Simulación")
        advance_length = st.number_input("Largo de avance por ciclo (m)", 0.1, 10.0, 2.5, 0.1)
        num_simulations = st.number_input("Número de simulaciones de avance", 100, 10000, 1000, 100)
        time_limit = st.number_input("Tiempo límite total (horas)", 1.0, 8760.0, 160.0, 1.0) 
        restriction_radius = st.number_input("Radio de restricciones (m)", 10.0, 200.0, 50.0, 10.0)
        restriction_delay_time = st.number_input("Demora por restricción (horas)", 0.0, 24.0, 4.0, 0.5)
        time_modification_percent = st.number_input("Modificación de tiempo global (%)", -50, 100, 0, 5, help="Afecta a TODAS las actividades.")

        st.markdown("---")
        st.subheader("🔧 Sensibilización de Recursos")
        if st.session_state.simulator.activities_df is not None:
            levels = st.session_state.simulator.activities_df['Nivel'].dropna().unique()
            level_to_config = st.selectbox("Nivel para ajustar recursos", options=levels, key="level_adj_res")
            if level_to_config:
                if level_to_config not in st.session_state.resource_adjustments:
                    st.session_state.resource_adjustments[level_to_config] = {}
                resources = get_resources_for_level(st.session_state.simulator.activities_df, level_to_config)
                with st.expander(f"Ajustes para Nivel '{level_to_config}'", expanded=True):
                    if not resources: st.warning("No hay recursos asignados en este nivel.")
                    else:
                        for r in resources:
                            st.session_state.resource_adjustments[level_to_config][r] = st.slider(f"Ajuste para **{r}** (%)", -50, 50, st.session_state.resource_adjustments[level_to_config].get(r, 0), 5, key=f"adj_{level_to_config}_{r}")
        else:
            st.info("Cargue datos para configurar recursos.")

        st.markdown("---")
        st.subheader("⏱️ Imputar Demoras Manuales")
        if st.session_state.simulator.activities_df is not None:
            with st.expander("Añadir nueva demora por cambio"):
                niveles = st.session_state.simulator.activities_df['Nivel'].dropna().unique()
                nivel_demora = st.selectbox("Nivel de la demora", niveles, key="d_nivel")

                recursos_nivel = get_resources_for_level(st.session_state.simulator.activities_df, nivel_demora)
                recurso_demora = st.selectbox("Recurso afectado", recursos_nivel, key="d_rec")

                turno_demora = st.slider("Turno de inicio de la demora (hora del día)", 0, 23, 7, key="d_turno")
                horas_demora = st.number_input("Horas de demora", min_value=0.5, value=4.0, step=0.5, key="d_horas")

                if st.button("➕ Añadir Demora Manual"):
                    st.session_state.manual_delays.append({
                        "nivel": nivel_demora,
                        "recurso": recurso_demora,
                        "turno": (turno_demora, turno_demora + 1), # Se asume una ventana de 1h para el trigger
                        "demora": horas_demora
                    })
                    st.success(f"Demora de {horas_demora}h para {recurso_demora} añadida.")

            if st.session_state.manual_delays:
                st.write("Demoras manuales activas:")
                for i, d in enumerate(st.session_state.manual_delays):
                    st.info(f"{i+1}: {d['recurso']} ({d['nivel']}) - {d['demora']}h a partir de la hora {d['turno'][0]}.")
                if st.button("Limpiar Demoras Manuales"):
                    st.session_state.manual_delays = []
                    st.rerun()
        else:
            st.info("Cargue datos para añadir demoras.")


    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("📁 Cargar Datos")
        recursos_file = st.file_uploader("Archivo de Actividades/Recursos/Frentes", type=['xlsx', 'xls'], key="uploader1")
        if recursos_file and recursos_file.name != st.session_state.recursos_file_name:
            if st.session_state.simulator.load_data_from_excel(recursos_file):
                st.session_state.recursos_file_name = recursos_file.name
                st.success(f"✅ Datos '{recursos_file.name}' cargados.")
                st.rerun()
        restricciones_file = st.file_uploader("Archivo de Restricciones", type=['xlsx', 'xls'], key="uploader2")
        if restricciones_file and restricciones_file.name != st.session_state.restricciones_file_name:
            if st.session_state.simulator.load_restrictions_from_excel(restricciones_file):
                st.session_state.restricciones_file_name = restricciones_file.name
                st.success(f"✅ Restricciones '{restricciones_file.name}' cargadas.")
                st.rerun()
        if st.session_state.recursos_file_name: st.write(f"✔️ Datos cargados: **{st.session_state.recursos_file_name}**")
        if st.session_state.restricciones_file_name: st.write(f"✔️ Restricciones cargadas: **{st.session_state.restricciones_file_name}**")

    with col2:
        st.subheader("🎯 Visualización de Frentes")
        if st.session_state.simulator.fronts_df is not None:
            niveles_viz = st.session_state.simulator.fronts_df['Nivel'].unique()
            nivel_selected_viz = st.selectbox("Nivel para visualización", niveles_viz, key="nivel_viz")
            if nivel_selected_viz:
                fig_viz = generate_fronts_visualization(st.session_state.simulator.fronts_df, st.session_state.simulator.restrictions, restriction_radius, nivel_selected_viz)
                st.plotly_chart(fig_viz, use_container_width=True)

    if st.session_state.simulator.activities_df is not None:
        st.markdown("---")
        st.subheader("📊 Modificar Distribuciones de Actividades")

        # --- Nueva sección para modificación masiva por nivel ---
        with st.expander("Modificación Masiva por Nivel", expanded=True):
            niveles_mod_masiva = st.session_state.simulator.activities_df['Nivel'].dropna().unique()
            nivel_masivo_selected = st.selectbox("Seleccionar Nivel para modificación masiva", niveles_mod_masiva, key="nivel_masivo_sel")

            if nivel_masivo_selected:
                activities_in_level = st.session_state.simulator.activities_df[st.session_state.simulator.activities_df['Nivel'] == nivel_masivo_selected]
                activities_to_modify_uids = st.multiselect(
                    "Seleccionar Actividades a Modificar (o dejar vacío para todas)",
                    options=activities_in_level.apply(lambda x: f"{x['Actividad']} ({x['Recurso']})", axis=1).unique(),
                    key="activities_masive_sel"
                )
                
                # Filter to get actual UIDs
                if activities_to_modify_uids:
                    filtered_uids = []
                    for act_str in activities_to_modify_uids:
                        # Extract Activity and Resource from string "Activity (Resource)"
                        act_name = act_str.split(' (')[0]
                        res_name = act_str.split(' (')[1][:-1]
                        # Find the corresponding unique_id
                        matching_rows = activities_in_level[(activities_in_level['Actividad'] == act_name) & (activities_in_level['Recurso'] == res_name)]
                        if not matching_rows.empty:
                            filtered_uids.extend(matching_rows['unique_id'].tolist())
                else:
                    filtered_uids = activities_in_level['unique_id'].tolist()


                st.markdown("#### Configurar Nueva Distribución (Aplicará a las seleccionadas)")
                dist_options_masiva = ['cte', 'norm', 'lognorm', 'weibull', 'gamma', 'fisk', 'rayleigh']
                new_dist_masiva = st.selectbox("Tipo de Distribución", dist_options_masiva, key="new_dist_masiva")

                new_params_masiva = {}
                tiempo_base_masiva = st.number_input("Tiempo Medio/Valor Constante (horas)", 0.01, value=1.0, step=0.1, key="tb_masiva", help="Valor central o constante para la nueva distribución.")
                new_params_masiva['tiempo_base'] = tiempo_base_masiva
                new_params_masiva['value'] = tiempo_base_masiva # For 'cte' distribution

                if new_dist_masiva != 'cte':
                    variability_masiva = st.slider("Variabilidad Relativa (scale)", 0.0, 2.0, 0.25, 0.05, key="s_masiva", help="Controla la desviación estándar en relación al tiempo medio.")
                    scale_masiva = tiempo_base_masiva * variability_masiva
                    new_params_masiva['loc'] = tiempo_base_masiva
                    new_params_masiva['scale'] = scale_masiva

                    if new_dist_masiva in ['lognorm', 'weibull', 'gamma', 'fisk']:
                        shape_key_masiva = {'lognorm': 's', 'weibull': 'c', 'gamma': 'a', 'fisk': 'c'}[new_dist_masiva]
                        shape_label_masiva = {'lognorm': 'Sigma', 'weibull': 'k', 'gamma': 'alpha', 'fisk': 'c'}[new_dist_masiva]
                        shape_defaults_masiva = {'s': 0.5, 'c': 2.0, 'a': 2.0}
                        new_params_masiva[shape_key_masiva] = st.number_input(f"Parámetro de Forma ({shape_label_masiva})", 0.1, 20.0, shape_defaults_masiva.get(shape_key_masiva, 1.0), 0.1, key=f"p1_masiva")

                # Visualización de la distribución propuesta
                st.markdown("#### Vista Previa de la Distribución Propuesta")
                # Create a dummy activity for visualization purposes
                original_params_for_viz = {'tiempo_base': tiempo_base_masiva} # Just a placeholder
                if new_dist_masiva == 'cte':
                     original_dist_name_for_viz = 'cte'
                     original_params_for_viz = {'value': tiempo_base_masiva}
                else:
                    original_dist_name_for_viz = 'norm' # Using norm as a base for comparison if no specific original is loaded
                    original_params_for_viz = {'loc': tiempo_base_masiva, 'scale': tiempo_base_masiva * 0.25} # Arbitrary initial variability for comparison

                fig_mass_comp = st.session_state.simulator.plot_comparison_distribution(
                    original_dist_name=original_dist_name_for_viz, original_params_tuple=tuple(sorted(original_params_for_viz.items())),
                    modified_dist_name=new_dist_masiva, modified_params_tuple=tuple(sorted(new_params_masiva.items())),
                    title=f"Comparación de la Nueva Distribución ({new_dist_masiva})"
                )
                if fig_mass_comp:
                    st.plotly_chart(fig_mass_comp, use_container_width=True)

                col_mass_save, col_mass_reset = st.columns(2)
                with col_mass_save:
                    if st.button("💾 Aplicar Cambios Masivos", key="save_mass_mod", use_container_width=True):
                        if filtered_uids:
                            for uid in filtered_uids:
                                st.session_state.modified_times[uid] = {'params': new_params_masiva.copy(), 'distribucion': new_dist_masiva}
                            st.success(f"Aplicados cambios a {len(filtered_uids)} actividades del nivel {nivel_masivo_selected}.")
                            st.rerun()
                        else:
                            st.warning("No hay actividades seleccionadas para aplicar cambios.")
                with col_mass_reset:
                    if st.button("🔄 Restablecer todas las actividades de este nivel a su original", key="reset_mass_mod", use_container_width=True):
                        count_reset = 0
                        for uid in activities_in_level['unique_id'].tolist():
                            if uid in st.session_state.modified_times:
                                del st.session_state.modified_times[uid]
                                count_reset += 1
                        st.info(f"Restablecidas {count_reset} actividades del nivel {nivel_masivo_selected} a su configuración original.")
                        st.rerun()

        st.markdown("---")
        # --- Sección para modificación individual (existente) ---
        with st.expander("Configurar Tiempos de Actividades Individualmente", expanded=False):
            activities = st.session_state.simulator.activities_df[['Nivel', 'Actividad', 'Recurso', 'unique_id']].drop_duplicates().apply(lambda x: f"{x[0]} - {x[1]} ({x[2]}) - {x[3]}", axis=1)
            activity_to_modify_str = st.selectbox("Actividad para modificar", options=activities, key="single_act_mod_sel")

            if activity_to_modify_str:
                parts = activity_to_modify_str.split(' - ')
                nivel_mod = parts[0]
                act_res_part = parts[1]
                uid = parts[2]
                
                # Find the original row for the selected unique_id
                act_rows = st.session_state.simulator.activities_df[st.session_state.simulator.activities_df['unique_id'] == uid]
                
                if not act_rows.empty:
                    current_act = act_rows.iloc[0]
                    params_before = current_act.get('params', {}) or {'tiempo_base': current_act['tiempo_base']}

                    st.markdown("#### Comparación de Distribuciones")
                    original_params_tuple = tuple(sorted(params_before.items()))

                    if uid in st.session_state.modified_times:
                        mod_data = st.session_state.modified_times[uid]
                        fig_comp = st.session_state.simulator.plot_comparison_distribution(
                            original_dist_name=current_act['Distribucion'], original_params_tuple=original_params_tuple,
                            modified_dist_name=mod_data['distribucion'], modified_params_tuple=tuple(sorted(mod_data['params'].items())),
                            title=f"Comparación para: {act_res_part}")
                    else:
                        fig_comp = st.session_state.simulator.plot_comparison_distribution(
                            original_dist_name=current_act['Distribucion'], original_params_tuple=original_params_tuple,
                            title=f"Distribución Original para: {act_res_part}")
                    if fig_comp:
                        st.plotly_chart(fig_comp, use_container_width=True)

                    st.markdown("---")
                    st.markdown("#### Controles de Modificación")
                    dist_options = ['cte', 'norm', 'lognorm', 'weibull', 'gamma', 'fisk', 'rayleigh']
                    current_dist = st.session_state.modified_times.get(uid, {'distribucion': current_act['Distribucion']})['distribucion'].lower()
                    current_dist_index = dist_options.index(current_dist) if current_dist in dist_options else 0
                    new_dist = st.selectbox("Distribución", dist_options, index=current_dist_index, key=f"d_{uid}")

                    tiempo_base = st.number_input("Tiempo Medio (loc)", 0.01, value=float(current_act['tiempo_base']), step=0.1, key=f"t_{uid}", help="Valor central o de localización.")
                    variability = st.slider("Variabilidad Relativa (scale)", 0.0, 2.0, 0.25, 0.05, key=f"s_{uid}", help="Controla la desviación estándar en relación al tiempo medio.")

                    scale = tiempo_base * variability
                    new_params = {'loc': tiempo_base, 'scale': scale, 'tiempo_base': tiempo_base}

                    if new_dist == 'cte': new_params = {'tiempo_base': tiempo_base, 'value': tiempo_base}
                    elif new_dist in ['lognorm', 'weibull', 'gamma', 'fisk']:
                        shape_key = {'lognorm': 's', 'weibull': 'c', 'gamma': 'a', 'fisk': 'c'}[new_dist]
                        shape_label = {'lognorm': 'Sigma', 'weibull': 'k', 'gamma': 'alpha', 'fisk': 'c'}[new_dist]
                        shape_defaults = {'s': 0.5, 'c': 2.0, 'a': 2.0}
                        # If already modified and has the shape param, use its value, else use default
                        default_shape_val = shape_defaults.get(shape_key, 1.0)
                        if uid in st.session_state.modified_times and shape_key in st.session_state.modified_times[uid]['params']:
                            default_shape_val = st.session_state.modified_times[uid]['params'][shape_key]
                        new_params[shape_key] = st.number_input(f"Parámetro de Forma ({shape_label})", 0.1, 20.0, default_shape_val, 0.1, key=f"p1_{uid}")


                    col_save, col_reset = st.columns(2)
                    with col_save:
                        if st.button("💾 Guardar Cambios", key=f"save_{uid}", use_container_width=True):
                            st.session_state.modified_times[uid] = {'params': new_params, 'distribucion': new_dist}
                            st.success("Cambios guardados.")
                            st.rerun()
                    with col_reset:
                        if uid in st.session_state.modified_times and st.button("🔄 Restaurar Original", key=f"del_{uid}", use_container_width=True):
                            del st.session_state.modified_times[uid]
                            st.info("Modificación eliminada.")
                            st.rerun()

    st.markdown("---")
    st.subheader("🚀 Selección de Frentes y Ejecución")
    if st.session_state.simulator.fronts_df is not None:
        niveles_sim = st.session_state.simulator.fronts_df['Nivel'].unique()
        nivel_sel = st.selectbox("Nivel para Simular", niveles_sim, key="nivel_sim")
        if nivel_sel:
            frentes = st.session_state.simulator.fronts_df[st.session_state.simulator.fronts_df['Nivel'] == nivel_sel]['Frentes'].unique()
            frentes_sel = st.multiselect("Frentes a Simular", frentes, default=frentes, key="frentes_sel")

            if frentes_sel and st.session_state.simulator.restrictions:
                fronts_to_check = st.session_state.simulator.fronts_df[st.session_state.simulator.fronts_df['Frentes'].isin(frentes_sel)]
                colliding_fronts_info, total_collisions = st.session_state.simulator.check_potential_collisions(
                    fronts_to_check, st.session_state.simulator.restrictions, restriction_radius)
                if total_collisions > 0:
                    colliding_names = list(colliding_fronts_info.keys())
                    st.warning(f"**⚠️ Alerta de Colisión Potencial:**\n- **{len(colliding_names)} frente(s)** ({', '.join(colliding_names)}) intersecta(n) con zonas de restricción, lo que puede añadir demoras.")

            if st.button("Ejecutar Simulación", type="primary", use_container_width=True):
                if frentes_sel:
                    with st.spinner("Ejecutando simulación de avance..."):
                        fronts_to_run = st.session_state.simulator.fronts_df[st.session_state.simulator.fronts_df['Frentes'].isin(frentes_sel)]
                        all_results = [st.session_state.simulator.simulate_tunnel_advance(
                                            info, advance_length, num_simulations, time_limit, restriction_radius,
                                            time_modification_percent, restriction_delay_time,
                                            st.session_state.modified_times, st.session_state.resource_adjustments,
                                            manual_delays=st.session_state.manual_delays
                                        ).assign(Frentes=info['Frentes'], Nivel=info['Nivel'])
                                       for _, info in fronts_to_run.iterrows()]
                        if all_results:
                            st.session_state.all_results_df = pd.concat(all_results, ignore_index=True)

                    with st.spinner("Generando análisis detallado y Carta Gantt..."):
                        selected_level_for_analysis = nivel_sel # Use the selected level for simulation
                        num_samples_analysis = 1000

                        st.session_state.activity_dist_df = st.session_state.simulator.simulate_activity_distributions(
                            nivel=selected_level_for_analysis, num_samples=num_samples_analysis,
                            time_modification_percent=time_modification_percent, resource_adjustments=st.session_state.resource_adjustments,
                            modified_times=st.session_state.modified_times)

                        gantt_df, delay_df = st.session_state.simulator.generate_gantt_data(
                            nivel=selected_level_for_analysis, num_samples=num_samples_analysis,
                            time_modification_percent=time_modification_percent,
                            resource_adjustments=st.session_state.resource_adjustments,
                            modified_times=st.session_state.modified_times,
                            manual_delays=st.session_state.manual_delays
                        )
                        st.session_state.gantt_df = gantt_df
                        st.session_state.delay_df = delay_df

                    st.success("✅ Simulación y análisis completados!")
                    st.rerun()
    else:
        st.warning("⚠️ Carga datos para seleccionar frentes.")

    # --- SECCIÓN DE RESULTADOS  ---
    if 'all_results_df' in st.session_state and not st.session_state.all_results_df.empty:
        st.markdown("---")
        st.header("📊 Resultados de la Simulación de Avance")
        results_df = st.session_state.all_results_df

        summary = results_df.groupby(['Nivel', 'Frentes']).agg(
            Distancia_Promedio_m=('actual_distance', 'mean'),
            Eficiencia_Media_pct=('efficiency', lambda x: x.mean() * 100),
            Ciclos_Completados_Promedio=('completed_cycles', 'mean'),
            Tiempo_Ciclo_Promedio_hrs=('avg_cycle_time', 'mean'),
            Tiempo_Total_Promedio_hrs=('total_sim_time', 'mean')
        ).reset_index()

        summary = pd.merge(st.session_state.simulator.fronts_df, summary, on=['Nivel', 'Frentes'], how='right')
        summary['Largo_Planeado_m'] = np.sqrt((summary['Xf'] - summary['Xi'])**2 + (summary['Yf'] - summary['Yi'])**2)
        
        st.info("Esta tabla muestra la conexión directa entre demoras, tiempo de ciclo, ciclos completados y el avance final.")
        
        st.dataframe(summary[[
            'Nivel', 'Frentes', 'Largo_Planeado_m', 'Distancia_Promedio_m',
            'Eficiencia_Media_pct', 'Ciclos_Completados_Promedio', 'Tiempo_Ciclo_Promedio_hrs'
        ]].round(2), use_container_width=True)


        res_col1, res_col2 = st.columns(2)
        with res_col1:
            fig_hist = px.histogram(results_df, x='actual_distance', color='Frentes',
                                    barmode='overlay', title='Distribución de Distancias Finales por Frente',
                                    labels={'actual_distance': 'Distancia Final (m)'})
            st.plotly_chart(fig_hist, use_container_width=True)
        with res_col2:
            fig_scatter = px.scatter(results_df, x='avg_cycle_time', y='actual_distance', color='Frentes',
                                     title='Distancia Lograda vs. Tiempo de Ciclo Promedio',
                                     labels={'actual_distance': 'Distancia Final (m)',
                                             'avg_cycle_time': 'Tiempo Promedio Ciclo (hrs)'})
            st.plotly_chart(fig_scatter, use_container_width=True)

    if 'activity_dist_df' in st.session_state and not st.session_state.activity_dist_df.empty:
        st.markdown("---")
        st.header("🔬 Análisis Detallado de Actividades del Ciclo")

        st.subheader("Análisis de Tiempos (Optimista-Pesimista-Probable)")
        st.info("Calculado sobre 1000 ciclos simulados con los parámetros actuales. Optimista=P10, Probable=P50 (Mediana), Pesimista=P90.")

        summary_act = st.session_state.activity_dist_df.groupby(['Actividad', 'Recurso'])['Duracion'].agg(
            Optimista_P10=lambda x: x.quantile(0.1),
            Mas_Probable_P50=lambda x: x.quantile(0.5),
            Pesimista_P90=lambda x: x.quantile(0.9)
        ).reset_index()
        st.dataframe(summary_act.round(2), use_container_width=True)

        fig_dist = plot_activity_distributions(st.session_state.activity_dist_df)
        st.plotly_chart(fig_dist, use_container_width=True)

    if 'gantt_df' in st.session_state and not st.session_state.gantt_df.empty:
        st.subheader("Carta Gantt Probabilística del Ciclo")
        st.info("La barra roja representa una demora fija por cambio de equipo. Su impacto se propaga a todas las actividades posteriores.")

        fig_gantt = plot_probabilistic_gantt(st.session_state.gantt_df, st.session_state.get('delay_df'))
        st.plotly_chart(fig_gantt, use_container_width=True)

if __name__ == "__main__":
    main()
