import pickle
import numpy as np
from collections import defaultdict

class TrainedAgent:
    def __init__(self, maze_size):
        self.maze_size = maze_size
        self.q_table = None
        self.load_model()
    
    def load_model(self):
        try:
            with open('trained_agents.pkl', 'rb') as f:
                models = pickle.load(f)
                if self.maze_size == 15:
                    model_data = models['agent_15x15']
                elif self.maze_size == 20:
                    model_data = models['agent_20x20']
                elif self.maze_size == 25:
                    # Para el laberinto 25x25, usamos el modelo del 20x20
                    print("Usando modelo del laberinto 20x20 para el laberinto 25x25")
                    model_data = models['agent_20x20']
                else:
                    raise ValueError(f"No hay modelo entrenado para laberinto {self.maze_size}x{self.maze_size}")
                
                self.q_table = defaultdict(lambda: np.zeros(4), model_data['q_table'])
        except FileNotFoundError:
            print(f"Error: No se encontró el archivo trained_agents.pkl")
            self.q_table = defaultdict(lambda: np.zeros(4))
    
    def get_action(self, state):
        """Selecciona la mejor acción según la Q-table"""
        state_tuple = tuple(state['agent_pos'])
        return np.argmax(self.q_table[state_tuple]) 