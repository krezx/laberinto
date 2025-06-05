# Laberinto Interactivo

Este proyecto implementa una visualización interactiva de laberintos con obstáculos dinámicos. El proyecto está desarrollado en Python utilizando Flask para la interfaz web y NumPy para los cálculos.

## Características

- Tres laberintos de diferentes tamaños (15x15, 20x20, 25x25)
- Obstáculos dinámicos que aparecen cada 10 movimientos
- Interfaz web interactiva para visualizar el laberinto
- Movimiento aleatorio del agente
- Visualización en tiempo real del progreso

## Requisitos

- Python 3.7 o superior
- Flask
- NumPy

## Instalación

1. Clonar el repositorio:
```bash
git clone <url-del-repositorio>
cd laberinto
```

2. Instalar las dependencias:
```bash
pip install -r requirements.txt
```

## Uso

1. Ejecutar la aplicación:
```bash
python app.py
```

2. Abrir un navegador web y acceder a:
```
http://localhost:5000
```

3. Seleccionar uno de los tres laberintos disponibles (15x15, 20x20, 25x25) y observar cómo el agente se mueve aleatoriamente.

## Estructura del Proyecto

- `app.py`: Aplicación principal de Flask
- `maze.py`: Implementación de la clase Maze
- `templates/`: Directorio con las plantillas HTML
- `requirements.txt`: Dependencias del proyecto
