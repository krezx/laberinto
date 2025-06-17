from flask import Flask, render_template, jsonify, request
import numpy as np
from maze import Maze
from trained_agent import TrainedAgent
from qlearning_trainer import QLearningAgent

app = Flask(__name__)

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
    
    # Si es el laberinto 25x25, actualizar Q-table
    if maze_id == 3:
        next_state = maze.get_state()
        agent.update_q_table(state, action, reward, next_state, done)
        agent.decay_epsilon()
    
    return jsonify(convert_numpy_types({
        "state": maze.get_full_state(),
        "reward": reward,
        "done": done,
        "action": action,
        "epsilon": agent.epsilon if maze_id == 3 else None
    }))

@app.route('/train_25x25')
def train_25x25():
    """Ruta para entrenar el agente 25x25 en vivo"""
    global maze3, agent3
    
    # Reiniciar el laberinto y el agente
    maze3 = Maze(25, 25)
    agent3 = QLearningAgent(learning_rate=0.1, discount_factor=0.99, epsilon=1.0)
    
    # Realizar un paso de entrenamiento
    state = maze3.get_state()
    action = agent3.get_action(state)
    reward, done = maze3.move(action)
    next_state = maze3.get_state()
    
    # Actualizar Q-table
    agent3.update_q_table(state, action, reward, next_state, done)
    
    # Actualizar obstáculos
    maze3.update_obstacles()
    if maze3.moves_count % 15 == 0:
        maze3.add_random_obstacle(1)
    
    # Reducir epsilon
    agent3.decay_epsilon()
    
    return jsonify(convert_numpy_types({
        "state": maze3.get_full_state(),
        "reward": reward,
        "done": done,
        "action": action,
        "epsilon": agent3.epsilon
    }))

@app.route('/reset_maze/<int:maze_id>')
def reset_maze(maze_id):
    global maze1, maze2, maze3, agent1, agent2, agent3
    
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
        return jsonify({"message": "Laberinto 3 reiniciado"})
    return jsonify({"error": "Laberinto no encontrado"}), 404

if __name__ == '__main__':
    app.run(debug=True) 