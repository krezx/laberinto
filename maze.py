import numpy as np

class Maze:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.grid = np.ones((height, width))  # Inicializar todo como paredes
        self.agent_pos = [1, 1]  # Posición inicial del agente
        self.goal_pos = [height-2, width-2]  # Posición de la meta
        self.moves_count = 0
        # Aumentar el límite de movimientos para el laberinto 25x25
        if width == 25 and height == 25:
            self.max_moves = (width + height) * 10  # 6 veces el tamaño del laberinto para 25x25
        else:
            self.max_moves = (width + height) * 3  # 3 veces el tamaño del laberinto para otros tamaños
        self.dynamic_obstacles = {}  # Diccionario para almacenar obstáculos dinámicos y sus tiempos de vida
        
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
            
        # Incrementar contador de movimientos antes de verificar si el movimiento es válido
        self.moves_count += 1
            
        # Verificar si el movimiento es válido
        if (0 <= new_pos[0] < self.height and 
            0 <= new_pos[1] < self.width and 
            self.grid[new_pos[0], new_pos[1]] != 1):  # No es pared
            
            # Verificar si es un obstáculo dinámico
            if self.grid[new_pos[0], new_pos[1]] == 2:
                return -10, False  # Penalización por intentar moverse a un obstáculo
            
            self.agent_pos = new_pos
            
        # Verificar si llegó a la meta
        if self.agent_pos == self.goal_pos:
            return 100, True  # Recompensa alta por llegar a la meta
            
        # Verificar si se excedió el límite de movimientos
        if self.moves_count >= self.max_moves:
            return -100, True  # Penalización por exceder el límite
            
        return -1, False  # Pequeña penalización por cada movimiento
    
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
        return tuple(self.agent_pos)  # Retornar el estado inicial 