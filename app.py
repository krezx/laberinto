from flask import Flask, render_template, jsonify, request
import numpy as np
from maze import Maze
from trained_agent import TrainedAgent
from qlearning_trainer import QLearningAgent
import types
import os
import pickle

app = Flask(__name__)

# Variables globales para el estado del modelo 25x25
modelo_25x25_entrenado = False
exitos_25x25 = 0  # Contador de éxitos para el entrenamiento Q-learning
# Inicializar mejor_pasos_25x25 leyendo del modelo guardado si existe
mejor_pasos_25x25 = None
if os.path.exists('trained_agents.pkl'):
    try:
        with open('trained_agents.pkl', 'rb') as f:
            models = pickle.load(f)
            if 'agent_25x25' in models and 'q_table' in models['agent_25x25']:
                # Buscar el mejor número de pasos guardado (si está en hyperparameters)
                if 'hyperparameters' in models['agent_25x25'] and 'mejor_pasos' in models['agent_25x25']['hyperparameters']:
                    mejor_pasos_25x25 = models['agent_25x25']['hyperparameters']['mejor_pasos']
    except Exception as e:
        print(f"No se pudo leer mejor_pasos_25x25 del modelo guardado: {e}")

def convert_numpy_types(obj):
    """Convierte tipos de NumPy a tipos nativos de Python"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    return obj

# Crear instancias de los laberintos y agentes
maze1 = Maze(15, 15)  # Laberinto 15x15
maze2 = Maze(20, 20)  # Laberinto 20x20
maze3 = Maze(25, 25)  # Laberinto 25x25

agent1 = TrainedAgent(15)
agent2 = TrainedAgent(20)
agent3 = QLearningAgent(learning_rate=0.1, discount_factor=0.99, epsilon=1.0)  # Agente 25x25 sin entrenamiento previo

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_maze/<int:maze_id>')
def get_maze(maze_id):
    if maze_id == 1:
        return jsonify(convert_numpy_types(maze1.get_full_state()))
    elif maze_id == 2:
        return jsonify(convert_numpy_types(maze2.get_full_state()))
    elif maze_id == 3:
        return jsonify(convert_numpy_types(maze3.get_full_state()))
    return jsonify({"error": "Laberinto no encontrado"}), 404

@app.route('/move/<int:maze_id>')
def move(maze_id):
    if maze_id == 1:
        maze = maze1
        agent = agent1
    elif maze_id == 2:
        maze = maze2
        agent = agent2
    elif maze_id == 3:
        maze = maze3
        agent = agent3
    else:
        return jsonify({"error": "Laberinto no encontrado"}), 404

    # Obtener acción del agente
    state = maze.get_state()
    action = agent.get_action(state)
    reward, done = maze.move(action)
    
    # Actualizar estado de los obstáculos primero
    maze.update_obstacles()
    
    # Agregar obstáculos según el tamaño del laberinto
    if maze.moves_count > 0:
        if maze.width == 15:  # Laberinto 15x15
            if maze.moves_count % 5 == 0:  # Cada 5 movimientos
                maze.add_random_obstacle(1)
        elif maze.width == 20:  # Laberinto 20x20
            if maze.moves_count % 10 == 0:  # Cada 10 movimientos
                maze.add_random_obstacle(2)
        elif maze.width == 25:  # Laberinto 25x25
            if maze.moves_count % 15 == 0:  # Cada 15 movimientos
                maze.add_random_obstacle(1)
    
    # Si es el laberinto 25x25 y el agente es QLearningAgent, actualizar Q-table
    if maze_id == 3 and isinstance(agent, QLearningAgent):
        next_state = maze.get_state()
        agent.update_q_table(state, action, reward, next_state, done)
        agent.decay_epsilon()
        epsilon = agent.epsilon
    else:
        epsilon = None
    
    return jsonify(convert_numpy_types({
        "state": maze.get_full_state(),
        "reward": reward,
        "done": done,
        "action": action,
        "epsilon": epsilon
    }))

@app.route('/train_25x25')
def train_25x25():
    """Entrena el agente 25x25 en vivo (un solo paso por llamada, visualización lenta) y guarda el modelo solo si mejora la ruta a la meta."""
    global maze3, agent3, exitos_25x25, mejor_pasos_25x25
    state = maze3.get_state()
    action = agent3.get_action(state)
    reward, done = maze3.move(action)
    next_state = maze3.get_state()
    agent3.update_q_table(state, action, reward, next_state, done)
    maze3.update_obstacles()
    if maze3.moves_count % 15 == 0:
        maze3.add_random_obstacle(1)
    agent3.decay_epsilon()
    # Lógica de éxito: si reward es 100 (meta alcanzada), sumar éxito y guardar modelo solo si mejora la ruta
    if reward == 100:
        exitos_25x25 += 1
        pasos_actuales = maze3.moves_count
        if mejor_pasos_25x25 is None or pasos_actuales < mejor_pasos_25x25:
            mejor_pasos_25x25 = pasos_actuales
            from qlearning_trainer import save_model_25x25
            # Guardar mejor_pasos en hyperparameters
           
            save_model_25x25(agent3, mejor_pasos_25x25)
    # Considerar entrenado si hay al menos 3 éxitos
    entrenado = exitos_25x25 >= 3
    return jsonify(convert_numpy_types({
        "state": maze3.get_full_state(),
        "reward": reward,
        "done": done,
        "action": action,
        "epsilon": agent3.epsilon,
        "exitos": exitos_25x25,
        "entrenado": entrenado,
        "mejor_pasos": mejor_pasos_25x25
    }))

@app.route('/skip_training_25x25', methods=['POST'])
def skip_training_25x25():
    """Entrena rápidamente el modelo 25x25 sin mostrar paso a paso y devuelve la simulación animada"""
    global maze3, agent3, modelo_25x25_entrenado
    from qlearning_trainer import train_agent_25x25, save_model_25x25
    agent, _, _, _ = train_agent_25x25(episodes=15000)
    agent3 = agent
    maze3 = Maze(25, 25)
    modelo_25x25_entrenado = True
    # Guardar solo el modelo entrenado actual de 25x25 (al final del entrenamiento)
    save_model_25x25(agent3)
    # Simular el recorrido con el nuevo modelo entrenado (sin guardar durante la simulación)
    state = maze3.get_state()
    done = False
    total_reward = 0
    steps = 0
    ruta = [list(maze3.agent_pos)]
    estados = [convert_numpy_types(maze3.get_full_state())]
    while not done and steps < maze3.max_moves:
        action = agent.get_action(state)
        reward, done = maze3.move(action)
        maze3.update_obstacles()
        if maze3.moves_count % 15 == 0:
            maze3.add_random_obstacle(1)
        state = maze3.get_state()
        ruta.append(list(maze3.agent_pos))
        estados.append(convert_numpy_types(maze3.get_full_state()))
        total_reward += reward
        steps += 1
    return jsonify({'success': True, 'message': 'Entrenamiento rápido completado.', 'reward': total_reward, 'steps': steps, 'done': done, 'state': convert_numpy_types(maze3.get_full_state()), 'ruta': ruta, 'estados': estados})

@app.route('/simulate_trained_25x25', methods=['POST'])
def simulate_trained_25x25():
    """Simula el laberinto 25x25 usando siempre el modelo entrenado guardado. Devuelve la ruta recorrida."""
    global maze3
    # Usar siempre el modelo guardado
    from trained_agent import TrainedAgent
    agent = TrainedAgent(25)
    # Reiniciar el laberinto
    maze3 = Maze(25, 25)
    # Simulación paso a paso
    state = maze3.get_state()
    done = False
    total_reward = 0
    steps = 0
    ruta = [list(maze3.agent_pos)]  # Guardar la posición inicial
    estados = [convert_numpy_types(maze3.get_full_state())]  # Guardar el estado inicial
    while not done and steps < maze3.max_moves:
        action = agent.get_action(state)
        reward, done = maze3.move(action)
        maze3.update_obstacles()
        if maze3.moves_count % 15 == 0:
            maze3.add_random_obstacle(1)
        state = maze3.get_state()
        ruta.append(list(maze3.agent_pos))  # Guardar cada posición
        estados.append(convert_numpy_types(maze3.get_full_state()))  # Guardar el estado tras cada paso
        total_reward += reward
        steps += 1
    return jsonify({'success': True, 'message': 'Simulación completada.', 'reward': total_reward, 'steps': steps, 'done': done, 'state': convert_numpy_types(maze3.get_full_state()), 'ruta': ruta, 'estados': estados})

@app.route('/continue_training_25x25', methods=['POST'])
def continue_training_25x25():
    """Continúa el entrenamiento del modelo 25x25 acumulando conocimiento y devuelve la simulación animada"""
    global maze3, agent3, modelo_25x25_entrenado
    from qlearning_trainer import train_agent_25x25
    # Entrenar usando el agente actual (acumula conocimiento)
    agent, _, _, _ = train_agent_25x25(episodes=5000, agente_existente=agent3)
    agent3 = agent
    modelo_25x25_entrenado = True
    # Simular el recorrido con el nuevo modelo entrenado
    maze3 = Maze(25, 25)
    state = maze3.get_state()
    done = False
    total_reward = 0
    steps = 0
    ruta = [list(maze3.agent_pos)]
    estados = [convert_numpy_types(maze3.get_full_state())]
    while not done and steps < maze3.max_moves:
        action = agent.get_action(state)
        reward, done = maze3.move(action)
        maze3.update_obstacles()
        if maze3.moves_count % 15 == 0:
            maze3.add_random_obstacle(1)
        state = maze3.get_state()
        ruta.append(list(maze3.agent_pos))
        estados.append(convert_numpy_types(maze3.get_full_state()))
        total_reward += reward
        steps += 1
    return jsonify({'success': True, 'message': 'Entrenamiento adicional completado.', 'reward': total_reward, 'steps': steps, 'done': done, 'state': convert_numpy_types(maze3.get_full_state()), 'ruta': ruta, 'estados': estados})

@app.route('/reset_maze/<int:maze_id>')
def reset_maze(maze_id):
    global maze1, maze2, maze3, agent1, agent2, agent3, modelo_25x25_entrenado
    if maze_id == 1:
        maze1 = Maze(15, 15)
        agent1 = TrainedAgent(15)
        return jsonify({"message": "Laberinto 1 reiniciado"})
    elif maze_id == 2:
        maze2 = Maze(20, 20)
        agent2 = TrainedAgent(20)
        return jsonify({"message": "Laberinto 2 reiniciado"})
    elif maze_id == 3:
        maze3 = Maze(25, 25)
        agent3 = QLearningAgent(learning_rate=0.1, discount_factor=0.99, epsilon=1.0)
        modelo_25x25_entrenado = False
        return jsonify({"message": "Laberinto 3 reiniciado"})
    return jsonify({"error": "Laberinto no encontrado"}), 404

if __name__ == '__main__':
    app.run(debug=True) 