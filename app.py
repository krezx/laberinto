from flask import Flask, render_template, jsonify, request
import numpy as np
from maze import Maze
from trained_agent import TrainedAgent

app = Flask(__name__)

# Crear instancias de los laberintos y agentes
maze1 = Maze(15, 15)  # Laberinto 15x15
maze2 = Maze(20, 20)  # Laberinto 20x20
maze3 = Maze(25, 25)  # Laberinto 25x25

agent1 = TrainedAgent(15)
agent2 = TrainedAgent(20)
agent3 = TrainedAgent(25)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_maze/<int:maze_id>')
def get_maze(maze_id):
    if maze_id == 1:
        return jsonify(maze1.get_state())
    elif maze_id == 2:
        return jsonify(maze2.get_state())
    elif maze_id == 3:
        return jsonify(maze3.get_state())
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

    # Obtener acción del agente entrenado
    state = maze.get_state()
    action = agent.get_action(state)
    reward, done = maze.move(action)
    
    # Actualizar estado de los obstáculos primero
    maze.update_obstacles()
    
    # Agregar obstáculos cada 10 movimientos según el tamaño del laberinto
    if maze.moves_count > 0:
        if maze.width == 15:  # Laberinto 15x15
            if maze.moves_count % 5 == 0:  # Cada 5 movimientos
                maze.add_random_obstacle(1)
        elif maze.width == 20:  # Laberinto 20x20
            if maze.moves_count % 10 == 0:  # Cada 10 movimientos
                maze.add_random_obstacle(2)
        elif maze.width == 25:  # Laberinto 25x25
            if maze.moves_count % 10 == 0:  # Cada 10 movimientos
                maze.add_random_obstacle(3)
    
    return jsonify({
        "state": maze.get_state(),
        "reward": reward,
        "done": done
    })

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
        agent3 = TrainedAgent(25)
        return jsonify({"message": "Laberinto 3 reiniciado"})
    return jsonify({"error": "Laberinto no encontrado"}), 404

if __name__ == '__main__':
    app.run(debug=True) 