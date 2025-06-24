import numpy as np
import math

class Maze:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.grid = np.ones((height, width))  # Inicializar todo como paredes
        self.agent_pos = [1, 1]  # Posición inicial del agente
        self.goal_pos = [height-2, width-2]  # Posición de la meta
        self.moves_count = 0
        # Límite de movimientos más generoso para 25x25
        if width == 25 and height == 25:
            self.max_moves = (width + height) * 20
        else:
            self.max_moves = (width + height) * 3
        self.dynamic_obstacles = {}  # Diccionario para almacenar obstáculos dinámicos y sus tiempos de vida
        self.visited = set()  # Para penalizar retrocesos
        self.last_distance = self.manhattan(self.agent_pos, self.goal_pos)  # Para premiar acercarse
        # Inicializar para penalización de bucles
        self.visit_count = {}
        self.recent_positions = []
        
        # Generar laberinto predefinido según el tamaño
        self.generate_predefined_maze()
        
    def generate_predefined_maze(self):
        # Laberinto 20x20 fijo
        if self.width == 20 and self.height == 20:
            laberinto_20x20 = [
                [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
                [1,0,0,0,1,0,1,0,0,0,1,0,1,0,1,0,0,0,0,1],
                [1,1,1,0,1,0,1,1,1,0,1,0,1,0,1,1,1,1,0,1],
                [1,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,1,0,1],
                [1,0,1,1,1,1,1,0,1,1,1,1,1,1,1,1,0,1,0,1],
                [1,0,1,0,0,0,1,0,0,0,0,0,0,0,0,1,0,1,0,1],
                [1,0,1,0,1,0,1,1,1,1,1,1,1,1,0,1,0,1,0,1],
                [1,0,1,0,1,0,0,0,0,0,0,0,0,1,0,1,0,1,0,1],
                [1,0,1,0,1,1,1,1,1,1,1,1,0,1,0,1,0,1,0,1],
                [1,0,1,0,0,0,0,0,0,1,0,0,0,1,0,1,0,1,0,1],
                [1,0,1,1,1,1,1,1,0,1,0,1,1,1,0,1,0,1,0,1],
                [1,0,0,0,0,0,0,1,0,1,0,0,0,0,0,1,0,1,0,1],
                [1,1,0,1,1,1,0,1,0,1,1,1,1,1,1,1,0,1,0,1],
                [1,0,0,0,0,1,0,1,0,0,0,0,0,1,0,0,0,1,0,1],
                [1,0,1,1,0,1,0,1,0,1,1,1,0,1,1,1,1,1,0,1],
                [1,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
                [1,0,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1],
                [1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
                [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1],
                [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
            ]
            self.grid = np.array(laberinto_20x20)
            self.agent_pos = [1, 1]
            self.goal_pos = [18, 18]
            self.grid[self.goal_pos[0], self.goal_pos[1]] = 0
        # Laberinto 15x15 fijo y completo
        elif self.width == 15 and self.height == 15:
            laberinto_15x15 = [
                [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
                [1,0,0,0,1,0,0,0,1,0,0,0,0,0,1],
                [1,1,1,0,1,0,1,0,1,1,1,1,1,0,1],
                [1,0,0,0,0,0,1,0,0,0,0,0,1,0,1],
                [1,0,1,1,1,1,1,1,1,1,1,0,1,0,1],
                [1,0,1,0,0,0,0,0,0,0,1,0,1,0,1],
                [1,0,1,0,1,1,1,1,0,1,1,0,1,0,1],
                [1,0,1,0,1,0,0,1,0,0,0,0,1,0,1],
                [1,0,1,0,1,0,1,1,1,1,1,1,1,0,1],
                [1,0,1,0,0,0,0,0,0,0,0,0,0,0,1],
                [1,0,1,1,1,1,1,1,1,1,1,1,1,1,1],
                [1,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
                [1,1,1,1,1,1,1,1,1,1,1,1,1,0,1],
                [1,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
                [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
            ]
            self.grid = np.array(laberinto_15x15)
            self.agent_pos = [1, 1]
            self.goal_pos = [13, 13]
            self.grid[self.goal_pos[0], self.goal_pos[1]] = 0
        # Laberinto 25x25 fijo y completo
        elif self.width == 25 and self.height == 25:
            laberinto_25x25 = [
                [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
                [1,0,0,0,1,0,1,0,0,0,1,0,1,0,1,0,0,0,1,0,0,0,0,0,1],
                [1,1,1,0,1,0,1,1,1,0,1,0,1,0,1,1,1,0,1,1,1,1,1,0,1],
                [1,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,1],
                [1,0,1,1,1,1,1,0,1,1,1,1,1,1,1,1,0,1,1,1,1,0,1,0,1],
                [1,0,1,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,1,0,1],
                [1,0,1,0,1,0,1,1,1,1,1,1,1,1,0,1,1,1,1,0,1,0,1,0,1],
                [1,0,1,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,1,0,1],
                [1,0,1,0,1,1,1,1,1,1,1,1,0,1,1,1,1,1,0,1,1,0,1,0,1],
                [1,0,1,0,0,0,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,0,1,0,1],
                [1,0,1,1,1,1,1,1,0,1,0,1,1,1,0,1,1,1,0,1,1,1,1,0,1],
                [1,0,0,0,0,0,0,1,0,1,0,0,0,0,0,1,0,0,0,1,0,0,0,0,1],
                [1,1,1,1,1,1,0,1,0,1,1,1,1,1,1,1,0,1,0,1,1,1,1,1,1],
                [1,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1],
                [1,0,1,1,0,1,0,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,0,1],
                [1,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1],
                [1,0,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,0,1],
                [1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,1],
                [1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,0,1,0,1],
                [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,1],
                [1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,0,1],
                [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1],
                [1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1],
                [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
                [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
            ]
            self.grid = np.array(laberinto_25x25)
            self.agent_pos = [1, 1]
            self.goal_pos = [23, 23]
            self.grid[self.goal_pos[0], self.goal_pos[1]] = 0
        # Asegurar que la meta sea accesible
        self.grid[self.goal_pos[0], self.goal_pos[1]] = 0
        self.grid[self.goal_pos[0]-1, self.goal_pos[1]] = 0
        self.grid[self.goal_pos[0], self.goal_pos[1]-1] = 0
    
    def get_valid_paths(self):
        """Obtiene las coordenadas de las rutas válidas hacia la meta"""
        valid_paths = []
        for y in range(self.height):
            for x in range(self.width):
                if self.grid[y, x] == 0:  # Si es un camino válido
                    valid_paths.append((y, x))
        return valid_paths

    def add_random_obstacle(self, num_obstacles=1):
        """Agrega obstáculos aleatorios en las rutas válidas del laberinto"""
        valid_paths = self.get_valid_paths()
        for _ in range(num_obstacles):
            attempts = 0
            while attempts < 100:  # Límite de intentos para evitar bucles infinitos
                # Seleccionar una posición aleatoria de las rutas válidas
                pos = valid_paths[np.random.randint(0, len(valid_paths))]
                y, x = pos
                # Verificar que el obstáculo esté a al menos 2 casillas del agente
                if (abs(y - self.agent_pos[0]) >= 2 or abs(x - self.agent_pos[1]) >= 2) and \
                   [y, x] != self.goal_pos and self.grid[y, x] == 0:
                    self.grid[y, x] = 2  # 2 representa un obstáculo dinámico
                    self.dynamic_obstacles[(y, x)] = 20  # 20 movimientos de vida
                    break
                attempts += 1
    
    def update_obstacles(self):
        """Actualiza el estado de los obstáculos dinámicos"""
        obstacles_to_remove = []
        for pos, lifetime in self.dynamic_obstacles.items():
            self.dynamic_obstacles[pos] = lifetime - 1
            if lifetime <= 1:
                obstacles_to_remove.append(pos)
                self.grid[pos] = 0  # Eliminar el obstáculo del grid
        
        # Eliminar obstáculos que han expirado
        for pos in obstacles_to_remove:
            del self.dynamic_obstacles[pos]
    
    def manhattan(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def move(self, action):
        # 0: arriba, 1: derecha, 2: abajo, 3: izquierda
        new_pos = self.agent_pos.copy()
        if action == 0:  # arriba
            new_pos[0] -= 1
        elif action == 1:  # derecha
            new_pos[1] += 1
        elif action == 2:  # abajo
            new_pos[0] += 1
        elif action == 3:  # izquierda
            new_pos[1] -= 1
        self.moves_count += 1
        # Verificar si el movimiento es válido
        if (0 <= new_pos[0] < self.height and 
            0 <= new_pos[1] < self.width and 
            self.grid[new_pos[0], new_pos[1]] != 1):  # No es pared
            # Verificar si es un obstáculo dinámico
            if self.grid[new_pos[0], new_pos[1]] == 2:
                return -10, False  # Penalización por intentar moverse a un obstáculo
            self.agent_pos = new_pos
        # Calcular recompensa acumulativa
        reward = -1  # Penalización base por moverse
        pos_tuple = tuple(self.agent_pos)
        self.visit_count[pos_tuple] = self.visit_count.get(pos_tuple, 0) + 1
        if self.visit_count[pos_tuple] == 1:
            reward += 0.5  # Recompensa por visitar una celda nueva
        elif self.visit_count[pos_tuple] > 1:
            reward -= 1  # Penalización por retroceso
        # Penalización por moverse en círculo (repetir secuencia reciente)
        self.recent_positions.append(pos_tuple)
        if len(self.recent_positions) > 8:
            self.recent_positions.pop(0)
        if len(self.recent_positions) == 8 and len(set(self.recent_positions)) <= 4:
            reward -= 5  # Penalización por bucle
        # Premiar acercarse a la meta
        new_distance = self.manhattan(self.agent_pos, self.goal_pos)
        if new_distance < self.last_distance:
            reward += 2  # Recompensa por acercarse
        self.last_distance = new_distance
        # Recompensa extra si bate el récord de pasos
        from app import mejor_pasos_25x25
        if self.agent_pos == self.goal_pos:
            extra = 0
            if mejor_pasos_25x25 is not None and self.moves_count < mejor_pasos_25x25:
                extra = 10
            return 100 + extra, True  # Recompensa alta por llegar a la meta
        if self.moves_count >= self.max_moves:
            return -100, True  # Penalización por exceder el límite
        return reward, False  # Retornar la recompensa calculada
    
    def get_state(self):
        """Devuelve el estado actual como tupla"""
        return tuple(self.agent_pos)  # Retornar solo la posición del agente como tupla

    def get_full_state(self):
        """Devuelve el estado completo para la visualización"""
        return {
            'grid': self.grid.tolist(),
            'agent_pos': self.agent_pos,
            'goal_pos': self.goal_pos,
            'moves_count': self.moves_count,
            'max_moves': self.max_moves
        }


    def reset(self):
        """Reinicia el laberinto al estado inicial"""
        self.agent_pos = [1, 1]
        self.moves_count = 0
        self.dynamic_obstacles = {}  # Limpiar obstáculos dinámicos
        self.grid = np.ones((self.height, self.width))  # Reiniciar el grid
        self.generate_predefined_maze()  # Regenerar el laberinto
        self.visited = set()
        self.last_distance = self.manhattan(self.agent_pos, self.goal_pos)
        self.visit_count = {}
        self.recent_positions = []
        return tuple(self.agent_pos)  # Retornar el estado inicial 