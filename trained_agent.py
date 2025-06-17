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
                    model_data = models['agent_25x25']
                else:
                    raise ValueError(f"No hay modelo entrenado para laberinto {self.maze_size}x{self.maze_size}")
                
                self.q_table = defaultdict(lambda: np.zeros(4), model_data['q_table'])
        except FileNotFoundError:
            print(f"Error: No se encontró el archivo trained_agents.pkl")
            self.q_table = defaultdict(lambda: np.zeros(4))
    
    def get_action(self, state):
        """Selecciona la mejor acción según la Q-table"""
        # Si el estado es un diccionario, extraer la posición del agente
        if isinstance(state, dict):
            state_tuple = tuple(state['agent_pos'])
        else:
            state_tuple = state  # Si ya es una tupla, usarla directamente
            
        return np.argmax(self.q_table[state_tuple]) 