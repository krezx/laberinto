import pickle
import numpy as np
import matplotlib.pyplot as plt
from qlearning_trainer2 import QLearningAgent

def load_trained_agents():
    """Carga los agentes entrenados desde el archivo pickle"""
    try:
        with open('trained_agents.pkl', 'rb') as f:
            models = pickle.load(f)
        
        agents = {}
        for size in ['15x15', '20x20', '25x25']:
            agent_key = f'agent_{size}'
            if agent_key in models:
                # Crear nuevo agente
                agent = QLearningAgent()
                
                # Cargar hiperparámetros
                hyperparams = models[agent_key]['hyperparameters']
                agent.learning_rate = hyperparams['learning_rate']
                agent.discount_factor = hyperparams['discount_factor']
                agent.epsilon = hyperparams['epsilon']
                agent.epsilon_min = hyperparams['epsilon_min']
                agent.epsilon_decay = hyperparams['epsilon_decay']
                
                # Cargar Q-table
                agent.q_table = models[agent_key]['q_table']
                
                agents[size] = agent
                print(f"Agente {size} cargado exitosamente")
            else:
                print(f"Agente {size} no encontrado en el archivo")
        
        return agents
    except FileNotFoundError:
        print("Archivo 'trained_agents.pkl' no encontrado")
        return None
    except Exception as e:
        print(f"Error al cargar agentes: {e}")
        return None

def visualize_q_tables(agents):
    """Visualiza las Q-tables de los agentes entrenados"""
    if not agents:
        print("No hay agentes para visualizar")
        return
    
    fig, axes = plt.subplots(len(agents), 4, figsize=(20, 5*len(agents)))
    if len(agents) == 1:
        axes = axes.reshape(1, -1)
    
    sizes = list(agents.keys())
    maze_sizes = {'15x15': 15, '20x20': 20, '25x25': 25}
    action_names = ['Arriba', 'Derecha', 'Abajo', 'Izquierda']
    
    for i, size in enumerate(sizes):
        agent = agents[size]
        maze_size = maze_sizes[size]
        print(f"\nProcesando Q-table para laberinto {size}...")
        
        # Crear matrices para cada acción
        q_matrices = {0: np.zeros((maze_size, maze_size)),  # Arriba
                     1: np.zeros((maze_size, maze_size)),  # Derecha
                     2: np.zeros((maze_size, maze_size)),  # Abajo
                     3: np.zeros((maze_size, maze_size))}  # Izquierda
        
        # Llenar las matrices con valores Q
        for state, q_values in agent.q_table.items():
            if isinstance(state, tuple) and len(state) == 2:
                y, x = state
                if 0 <= y < maze_size and 0 <= x < maze_size:
                    for action in range(4):
                        q_matrices[action][y, x] = get_q_value(q_values, action)
        
        # Visualizar cada acción
        for action in range(4):
            ax = axes[i, action]
            q_matrix = q_matrices[action]
            
            # Crear mapa de calor
            im = ax.imshow(q_matrix, cmap='RdYlBu_r', aspect='equal')
            ax.set_title(f'{size} - {action_names[action]}')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            
            # Agregar valores en las celdas (solo para laberintos pequeños)
            if maze_size <= 15:
                for y in range(maze_size):
                    for x in range(maze_size):
                        value = q_matrix[y, x]
                        if value != 0:  # Solo mostrar valores no cero
                            text = ax.text(x, y, f'{value:.1f}', 
                                         ha="center", va="center", 
                                         color="black" if abs(value) < 50 else "white",
                                         fontsize=6)
            
            # Agregar barra de color
            cbar = plt.colorbar(im, ax=ax, shrink=0.8)
            cbar.set_label('Valor Q')
            
            # Configurar ticks
            ax.set_xticks(range(maze_size))
            ax.set_yticks(range(maze_size))
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('q_tables_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Visualización de Q-tables guardada en 'q_tables_visualization.png'")

def create_q_table_heatmap(agent, maze_size):
    """Crea un mapa de calor de la Q-table para un agente específico"""
    # Crear matriz de valores Q promedio por estado
    q_avg_matrix = np.zeros((maze_size, maze_size))
    q_max_matrix = np.zeros((maze_size, maze_size))
    
    for state, q_values in agent.q_table.items():
        if isinstance(state, tuple) and len(state) == 2:
            y, x = state
            if 0 <= y < maze_size and 0 <= x < maze_size:
                # Obtener todos los valores Q para este estado
                state_values = []
                for action in range(4):
                    state_values.append(get_q_value(q_values, action))
                
                if state_values:
                    q_avg_matrix[y, x] = np.mean(state_values)
                    q_max_matrix[y, x] = max(state_values)
    
    return q_avg_matrix, q_max_matrix

def visualize_q_table_summary(agents):
    """Crea un resumen visual de las Q-tables mostrando valores promedio y máximos"""
    if not agents:
        print("No hay agentes para visualizar")
        return
    
    fig, axes = plt.subplots(2, len(agents), figsize=(6*len(agents), 12))
    if len(agents) == 1:
        axes = axes.reshape(2, 1)
    
    sizes = list(agents.keys())
    maze_sizes = {'15x15': 15, '20x20': 20, '25x25': 25}
    
    for i, size in enumerate(sizes):
        agent = agents[size]
        maze_size = maze_sizes[size]
        q_avg, q_max = create_q_table_heatmap(agent, maze_size)
        
        # Valor Q promedio
        im1 = axes[0, i].imshow(q_avg, cmap='RdYlBu_r', aspect='equal')
        axes[0, i].set_title(f'{size} - Valor Q Promedio')
        axes[0, i].set_xlabel('X')
        axes[0, i].set_ylabel('Y')
        plt.colorbar(im1, ax=axes[0, i], shrink=0.8)
        
        # Valor Q máximo
        im2 = axes[1, i].imshow(q_max, cmap='RdYlBu_r', aspect='equal')
        axes[1, i].set_title(f'{size} - Valor Q Máximo')
        axes[1, i].set_xlabel('X')
        axes[1, i].set_ylabel('Y')
        plt.colorbar(im2, ax=axes[1, i], shrink=0.8)
        
        # Agregar valores en las celdas para el laberinto más pequeño
        if maze_size <= 15:
            for y in range(maze_size):
                for x in range(maze_size):
                    if q_avg[y, x] != 0:
                        axes[0, i].text(x, y, f'{q_avg[y, x]:.1f}', 
                                       ha="center", va="center", 
                                       color="black" if abs(q_avg[y, x]) < 50 else "white",
                                       fontsize=8)
                    if q_max[y, x] != 0:
                        axes[1, i].text(x, y, f'{q_max[y, x]:.1f}', 
                                       ha="center", va="center", 
                                       color="black" if abs(q_max[y, x]) < 50 else "white",
                                       fontsize=8)
    
    plt.tight_layout()
    plt.savefig('q_tables_summary.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Resumen de Q-tables guardado en 'q_tables_summary.png'")

def print_q_table_statistics(agents):
    """Imprime estadísticas detalladas de las Q-tables"""
    print("\n=== ESTADÍSTICAS DE LAS Q-TABLES ===")
    
    for size, agent in agents.items():
        q_values = []
        for state_q in agent.q_table.values():
            # Manejar diferentes formatos de datos
            if isinstance(state_q, dict):
                q_values.extend(state_q.values())
            elif isinstance(state_q, np.ndarray):
                q_values.extend(state_q.tolist())
            else:
                # Si es un valor directo
                q_values.append(state_q)
        
        if q_values:
            print(f"\nLaberinto {size}:")
            print(f"  - Estados únicos: {len(agent.q_table)}")
            print(f"  - Valores Q totales: {len(q_values)}")
            print(f"  - Valor Q máximo: {max(q_values):.2f}")
            print(f"  - Valor Q mínimo: {min(q_values):.2f}")
            print(f"  - Valor Q promedio: {np.mean(q_values):.2f}")
            print(f"  - Desviación estándar: {np.std(q_values):.2f}")
            
            # Contar valores positivos vs negativos
            positive_q = sum(1 for q in q_values if q > 0)
            negative_q = sum(1 for q in q_values if q < 0)
            zero_q = sum(1 for q in q_values if q == 0)
            
            print(f"  - Valores Q positivos: {positive_q} ({positive_q/len(q_values)*100:.1f}%)")
            print(f"  - Valores Q negativos: {negative_q} ({negative_q/len(q_values)*100:.1f}%)")
            print(f"  - Valores Q cero: {zero_q} ({zero_q/len(q_values)*100:.1f}%)")

def get_q_value(state_q, action):
    """Obtiene el valor Q para una acción específica, manejando diferentes formatos"""
    if isinstance(state_q, dict):
        return state_q.get(action, 0)
    elif isinstance(state_q, np.ndarray):
        if action < len(state_q):
            return float(state_q[action])
        return 0
    else:
        return 0

def visualize_q_tables_max_only(agents):
    """Visualiza SOLO una Q-table por laberinto (valor Q máximo por estado)"""
    if not agents:
        print("No hay agentes para visualizar")
        return
    
    fig, axes = plt.subplots(1, len(agents), figsize=(6*len(agents), 6))
    if len(agents) == 1:
        axes = [axes]
    
    sizes = list(agents.keys())
    maze_sizes = {'15x15': 15, '20x20': 20, '25x25': 25}
    
    for i, size in enumerate(sizes):
        agent = agents[size]
        maze_size = maze_sizes[size]
        # Crear matriz de valor Q máximo por estado
        q_max_matrix = np.zeros((maze_size, maze_size))
        for state, q_values in agent.q_table.items():
            if isinstance(state, tuple) and len(state) == 2:
                y, x = state
                if 0 <= y < maze_size and 0 <= x < maze_size:
                    # Obtener el valor Q máximo para este estado
                    if isinstance(q_values, dict):
                        q_max = max(q_values.values()) if q_values else 0
                    elif isinstance(q_values, np.ndarray):
                        q_max = np.max(q_values) if len(q_values) > 0 else 0
                    else:
                        q_max = float(q_values)
                    q_max_matrix[y, x] = q_max
        # Visualizar
        ax = axes[i]
        im = ax.imshow(q_max_matrix, cmap='RdYlBu_r', aspect='equal')
        ax.set_title(f'{size} - Valor Q Máximo')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        plt.colorbar(im, ax=ax, shrink=0.8)
        # Agregar valores en las celdas para el laberinto más pequeño
        if maze_size <= 15:
            for y in range(maze_size):
                for x in range(maze_size):
                    if q_max_matrix[y, x] != 0:
                        ax.text(x, y, f'{q_max_matrix[y, x]:.1f}', 
                                ha="center", va="center", 
                                color="black" if abs(q_max_matrix[y, x]) < 50 else "white",
                                fontsize=8)
        ax.set_xticks(range(maze_size))
        ax.set_yticks(range(maze_size))
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('q_tables_max_only.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Visualización de Q-tables (máximo por estado) guardada en 'q_tables_max_only.png'")

def main():
    print("=== VISUALIZADOR DE Q-TABLES ===")
    agents = load_trained_agents()
    if agents:
        print_q_table_statistics(agents)
        visualize_q_tables_max_only(agents)
        print("\n=== Visualización completada ===")
        print("Archivo generado:")
        print("- q_tables_max_only.png: Q-table máxima por laberinto")
    else:
        print("No se pudieron cargar los agentes entrenados")

if __name__ == "__main__":
    main() 