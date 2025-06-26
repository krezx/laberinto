# Laberinto Interactivo con Q-Learning

Este proyecto implementa una plataforma interactiva para experimentar con laberintos y agentes inteligentes entrenados mediante Q-Learning. Incluye una interfaz web, entrenamiento visual y automático, análisis comparativo y visualización de métricas y Q-tables.

## Funcionalidades principales

- **Tres laberintos predefinidos**: Tamaños 15x15, 20x20 y 25x25, cada uno con su propio diseño y nivel de dificultad.
- **Obstáculos dinámicos**: Aparecen durante la simulación y el entrenamiento, cambiando la ruta óptima y aumentando el desafío.
- **Entrenamiento de agentes Q-Learning**:
  - Entrenamiento visual paso a paso para el laberinto 25x25 desde la web.
  - Entrenamiento rápido (sin visualización) para el laberinto 25x25.
  - Entrenamiento offline para los tres laberintos mediante scripts (`qlearning_trainer.py` y `qlearning_trainer2.py`).
- **Simulación**:
  - Simulación automática del agente en cualquier laberinto.
  - Simulación con el modelo entrenado para el laberinto 25x25.
- **Visualización interactiva**:
  - Interfaz web con animación del agente, obstáculos, meta y paredes.
  - Estadísticas en tiempo real: movimientos, recompensa, barra de progreso.
  - Leyenda visual de los elementos del laberinto.
- **Gestión de modelos**:
  - Guardado y carga automática de modelos entrenados.
  - Reinicio completo del laberinto 25x25 (incluye borrado del modelo).
- **Análisis y reportes**:
  - Generación de métricas de entrenamiento y gráficos comparativos.
  - Visualización de Q-tables y mapas de calor.
  - Reporte completo de entrenamiento (`training_report.txt`).
  - Imágenes generadas: `training_metrics.png`, `comparative_analysis.png`, `performance_summary.png`, `q_tables_max_only.png`.

## Flujo de ejecución

### 1. Instalación

```bash
git clone <url-del-repositorio>
cd laberinto
pip install -r requirements.txt
```

### 2. Uso desde la interfaz web

1. Ejecuta la aplicación:
   ```bash
   python app.py
   ```
2. Abre tu navegador en [http://localhost:5000](http://localhost:5000).
3. Selecciona el laberinto que deseas explorar:
   - **15x15 y 20x20**: Simulación directa con el agente entrenado.
   - **25x25**: Puedes entrenar al agente visualmente, entrenar rápido sin visualización, simular con el modelo entrenado o volver a entrenar para mejorar el modelo.
4. Observa la animación, estadísticas y evolución del agente en tiempo real.
5. Reinicia el laberinto o el modelo cuando lo desees.

### 3. Entrenamiento y análisis offline

- Ejecuta los scripts de entrenamiento para generar modelos y reportes:
  ```bash
  python qlearning_trainer.py
  # o
  python qlearning_trainer2.py
  ```
- Se generarán archivos de métricas, análisis comparativos y reportes en la carpeta principal.

### 4. Visualización de Q-tables

- Ejecuta el visualizador:
  ```bash
  python q_tables_visualizer.py
  ```
- Se generarán imágenes de las Q-tables y estadísticas de los valores aprendidos.

## Archivos y estructura del proyecto

- `app.py`: Servidor Flask y lógica de la interfaz web.
- `maze.py`: Lógica y reglas del laberinto, obstáculos y recompensas.
- `qlearning_trainer.py` y `qlearning_trainer2.py`: Entrenamiento y análisis de agentes Q-Learning.
- `q_tables_visualizer.py`: Visualización avanzada de Q-tables.
- `trained_agent.py`: Clase para cargar y usar agentes entrenados.
- `trained_agents.pkl`: Modelos entrenados guardados.
- `training_report.txt`: Reporte detallado de entrenamiento y resultados.
- `training_metrics.png`, `comparative_analysis.png`, `performance_summary.png`, `q_tables_max_only.png`: Imágenes de análisis y visualización.
- `templates/index.html`: Plantilla HTML principal.
- `static/agent.png`: Imagen del agente.
- `requirements.txt`: Dependencias del proyecto.

## Requisitos

- Python 3.7 o superior
- Flask
- NumPy
- Matplotlib (para análisis y visualización)

## Créditos

Desarrollado para experimentación y docencia en Sistemas Inteligentes.
