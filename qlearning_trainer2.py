import numpy as np
import pickle
import matplotlib.pyplot as plt
from collections import defaultdict
import random
from tqdm import tqdm
from maze import Maze  # Importar la clase Maze desde maze.py
import datetime

class QLearningAgent:
    def __init__(self, n_actions=4, learning_rate=0.1, discount_factor=0.95, epsilon=1.0):
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        
        # Q-table usando defaultdict para manejar estados no visitados
        self.q_table = defaultdict(lambda: np.zeros(n_actions))
        
    def get_action(self, state):
        """Selecciona una acción usando epsilon-greedy"""
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        else:
            return np.argmax(self.q_table[state])
    
    def update_q_table(self, state, action, reward, next_state, done):
        """Actualiza la Q-table usando la ecuación de Bellman"""
        current_q = self.q_table[state][action]
        
        if done:
            target_q = reward
        else:
            target_q = reward + self.discount_factor * np.max(self.q_table[next_state])
        
        # Actualización Q-learning
        self.q_table[state][action] = current_q + self.learning_rate * (target_q - current_q)
    
    def decay_epsilon(self):
        """Reduce epsilon para disminuir la exploración con el tiempo"""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


def train_agent(maze_size, episodes=5000):
    """Entrena el agente en un laberinto específico"""
    print(f"\n=== Entrenando en laberinto {maze_size}x{maze_size} ===")
    
    # Crear ambiente y agente
    maze = Maze(maze_size, maze_size)
    agent = QLearningAgent()
    
    # Métricas de entrenamiento
    episode_rewards = []
    episode_steps = []
    success_rate = []
    avg_rewards = []
    avg_steps = []
    exploration_rate = []
    
    # Variables para tracking
    recent_successes = []
    recent_rewards = []
    recent_steps = []
    
    for episode in tqdm(range(episodes), desc=f"Entrenando {maze_size}x{maze_size}"):
        state = maze.reset()
        total_reward = 0
        steps = 0
        done = False
        
        while not done:
            # Seleccionar acción
            action = agent.get_action(state)
            
            # Ejecutar acción
            reward, done = maze.move(action)
            next_state = maze.get_state()
            
            # Actualizar Q-table
            agent.update_q_table(state, action, reward, next_state, done)
            
            # Actualizar estado
            state = next_state
            total_reward += reward
            steps += 1
        
        # Reducir epsilon
        agent.decay_epsilon()
        
        # Guardar métricas
        episode_rewards.append(total_reward)
        episode_steps.append(steps)
        
        # Calcular tasa de éxito (llegar a la meta)
        success = maze.agent_pos == maze.goal_pos
        recent_successes.append(success)
        if len(recent_successes) > 100:
            recent_successes.pop(0)
        
        # Calcular métricas móviles
        recent_rewards.append(total_reward)
        recent_steps.append(steps)
        if len(recent_rewards) > 100:
            recent_rewards.pop(0)
            recent_steps.pop(0)
        
        success_rate.append(np.mean(recent_successes) * 100)
        avg_rewards.append(np.mean(recent_rewards))
        avg_steps.append(np.mean(recent_steps))
        exploration_rate.append(agent.epsilon * 100)
        
        # Mostrar progreso cada 1000 episodios
        if (episode + 1) % 1000 == 0:
            print(f"Episodio {episode + 1}: Reward promedio: {avg_rewards[-1]:.2f}, "
                  f"Pasos promedio: {avg_steps[-1]:.2f}, Epsilon: {agent.epsilon:.3f}, "
                  f"Tasa de éxito: {success_rate[-1]:.1f}%")
    
    return agent, episode_rewards, episode_steps, success_rate, avg_rewards, avg_steps, exploration_rate


def test_agent(agent, maze_size, test_episodes=100):
    """Evalúa el rendimiento del agente entrenado"""
    print(f"\n=== Evaluando agente en laberinto {maze_size}x{maze_size} ===")
    
    maze = Maze(maze_size, maze_size)
    successes = 0
    total_steps = 0
    
    # Desactivar exploración para testing
    original_epsilon = agent.epsilon
    agent.epsilon = 0
    
    for episode in range(test_episodes):
        state = maze.reset()
        steps = 0
        done = False
        
        while not done:
            action = agent.get_action(state)
            reward, done = maze.move(action)
            state = maze.get_state()
            steps += 1
            
            if steps > maze.max_moves:  # Evitar bucles infinitos
                break
        
        if reward == 100:
            successes += 1
        total_steps += steps
    
    # Restaurar epsilon
    agent.epsilon = original_epsilon
    
    success_rate = (successes / test_episodes) * 100
    avg_steps = total_steps / test_episodes
    
    print(f"Tasa de éxito: {success_rate:.1f}% ({successes}/{test_episodes})")
    print(f"Pasos promedio: {avg_steps:.1f}")
    
    return success_rate, avg_steps


def train_agent_25x25(episodes=15000, agente_existente=None):
    """Entrena el agente en el laberinto 25x25 con obstáculos dinámicos. Early stopping si la tasa de éxito >= 90% en los últimos 100 episodios."""
    print("\n=== Entrenando en laberinto 25x25 con obstáculos dinámicos ===")
    maze = Maze(25, 25)
    if agente_existente is not None:
        agent = agente_existente
    else:
        agent = QLearningAgent(learning_rate=0.2, discount_factor=0.95, epsilon=1.0)
        agent.epsilon_min = 0.001
        agent.epsilon_decay = 0.999
    
    episode_rewards = []
    episode_steps = []
    success_rate = []
    avg_rewards = []
    avg_steps = []
    exploration_rate = []
    
    recent_successes = []
    recent_rewards = []
    recent_steps = []
    
    for episode in range(episodes):
        state = maze.reset()
        total_reward = 0
        steps = 0
        done = False
        maze.add_random_obstacle(1)
        
        while not done:
            action = agent.get_action(state)
            reward, done = maze.move(action)
            next_state = maze.get_state()
            if reward == 100:  # Llegar a la meta
                reward = 1000
            elif reward == -100:  # Exceder límite de movimientos
                reward = -200
            elif reward == -10:  # Chocar con obstáculo
                reward = -50
            agent.update_q_table(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            steps += 1
            # Menos obstáculos dinámicos durante entrenamiento rápido
            if steps % 50 == 0:
                maze.add_random_obstacle(1)
            maze.update_obstacles()
        
        agent.decay_epsilon()
        episode_rewards.append(total_reward)
        episode_steps.append(steps)
        
        # Calcular tasa de éxito (llegar a la meta)
        success = maze.agent_pos == maze.goal_pos
        recent_successes.append(success)
        if len(recent_successes) > 100:
            recent_successes.pop(0)
        
        # Calcular métricas móviles
        recent_rewards.append(total_reward)
        recent_steps.append(steps)
        if len(recent_rewards) > 100:
            recent_rewards.pop(0)
            recent_steps.pop(0)
        
        success_rate.append(np.mean(recent_successes) * 100)
        avg_rewards.append(np.mean(recent_rewards))
        avg_steps.append(np.mean(recent_steps))
        exploration_rate.append(agent.epsilon * 100)
        
        if (episode + 1) % 1000 == 0:
            print(f"Episodio {episode + 1}: Reward promedio: {avg_rewards[-1]:.2f}, "
                  f"Pasos promedio: {avg_steps[-1]:.2f}, Epsilon: {agent.epsilon:.3f}, "
                  f"Tasa de éxito: {success_rate[-1]:.1f}%")
        
        # Early stopping: si la tasa de éxito es >= 90% en los últimos 100 episodios
        if len(recent_successes) == 100 and success_rate[-1] >= 90.0:
            print(f"Entrenamiento detenido anticipadamente en el episodio {episode+1} por alta tasa de éxito ({success_rate[-1]:.1f}%)")
            break
    
    return agent, episode_rewards, episode_steps, success_rate, avg_rewards, avg_steps, exploration_rate


def generate_training_statistics(metrics_15x15, metrics_20x20, metrics_25x25):
    """Genera estadísticas detalladas del entrenamiento"""
    print("\n=== ESTADÍSTICAS DETALLADAS DEL ENTRENAMIENTO ===")
    
    sizes = ['15x15', '20x20', '25x25']
    metrics = [metrics_15x15, metrics_20x20, metrics_25x25]
    
    for i, (size, (rewards, steps, success, avg_rewards, avg_steps, exploration)) in enumerate(zip(sizes, metrics)):
        print(f"\n--- Laberinto {size} ---")
        
        # Estadísticas de recompensas
        final_avg_reward = avg_rewards[-1] if avg_rewards else 0
        max_reward = max(rewards) if rewards else 0
        min_reward = min(rewards) if rewards else 0
        
        print(f"Recompensa promedio final: {final_avg_reward:.2f}")
        print(f"Recompensa máxima: {max_reward:.2f}")
        print(f"Recompensa mínima: {min_reward:.2f}")
        
        # Estadísticas de pasos
        final_avg_steps = avg_steps[-1] if avg_steps else 0
        min_steps = min(steps) if steps else 0
        max_steps = max(steps) if steps else 0
        
        print(f"Pasos promedio final: {final_avg_steps:.2f}")
        print(f"Menor número de pasos: {min_steps}")
        print(f"Mayor número de pasos: {max_steps}")
        
        # Estadísticas de éxito
        final_success_rate = success[-1] if success else 0
        max_success_rate = max(success) if success else 0
        
        print(f"Tasa de éxito final: {final_success_rate:.1f}%")
        print(f"Tasa de éxito máxima: {max_success_rate:.1f}%")
        
        # Estadísticas de exploración
        final_exploration = exploration[-1] if exploration else 0
        initial_exploration = exploration[0] if exploration else 0
        
        print(f"Tasa de exploración inicial: {initial_exploration:.1f}%")
        print(f"Tasa de exploración final: {final_exploration:.1f}%")
        
        # Análisis de convergencia
        if len(success) > 100:
            last_100_success = success[-100:]
            convergence_rate = np.mean(last_100_success)
            print(f"Tasa de convergencia (últimos 100 episodios): {convergence_rate:.1f}%")
            
            if convergence_rate >= 80:
                print("✅ El agente ha convergido exitosamente")
            elif convergence_rate >= 50:
                print("⚠️ El agente muestra progreso pero necesita más entrenamiento")
            else:
                print("❌ El agente necesita más entrenamiento")


def plot_training_metrics(metrics_15x15, metrics_20x20, metrics_25x25):
    """Visualiza las métricas de entrenamiento con gráficas mejoradas"""
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    
    sizes = ['15x15', '20x20', '25x25']
    metrics = [metrics_15x15, metrics_20x20, metrics_25x25]
    
    for i, (size, (rewards, steps, success, avg_rewards, avg_steps, exploration)) in enumerate(zip(sizes, metrics)):
        # Reward por episodio con promedio móvil
        axes[i, 0].plot(rewards, alpha=0.3, color='blue', linewidth=0.5)
        axes[i, 0].plot(avg_rewards, 'r-', linewidth=2, label='Promedio móvil (100 episodios)')
        axes[i, 0].set_title(f'Rewards - Laberinto {size}')
        axes[i, 0].set_xlabel('Episodio')
        axes[i, 0].set_ylabel('Reward')
        axes[i, 0].grid(True, alpha=0.3)
        axes[i, 0].legend()
        
        # Pasos por episodio con promedio móvil
        axes[i, 1].plot(steps, alpha=0.3, color='green', linewidth=0.5)
        axes[i, 1].plot(avg_steps, 'r-', linewidth=2, label='Promedio móvil (100 episodios)')
        axes[i, 1].set_title(f'Pasos por Episodio - Laberinto {size}')
        axes[i, 1].set_xlabel('Episodio')
        axes[i, 1].set_ylabel('Pasos')
        axes[i, 1].grid(True, alpha=0.3)
        axes[i, 1].legend()
        
        # Tasa de éxito
        axes[i, 2].plot(success, 'g-', linewidth=2)
        axes[i, 2].axhline(y=90, color='red', linestyle='--', alpha=0.7, label='Objetivo 90%')
        axes[i, 2].axhline(y=80, color='orange', linestyle='--', alpha=0.5, label='Objetivo 80%')
        axes[i, 2].set_title(f'Tasa de Éxito - Laberinto {size}')
        axes[i, 2].set_xlabel('Episodio')
        axes[i, 2].set_ylabel('Tasa de Éxito (%)')
        axes[i, 2].set_ylim(0, 100)
        axes[i, 2].grid(True, alpha=0.3)
        axes[i, 2].legend()
        
        # Tasa de exploración (epsilon)
        axes[i, 3].plot(exploration, 'purple', linewidth=2)
        axes[i, 3].set_title(f'Tasa de Exploración - Laberinto {size}')
        axes[i, 3].set_xlabel('Episodio')
        axes[i, 3].set_ylabel('Epsilon (%)')
        axes[i, 3].set_ylim(0, 100)
        axes[i, 3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_metrics.png', dpi=300, bbox_inches='tight')
    plt.show()


def save_models(agent_15x15, agent_20x20, agent_25x25):
    """Guarda los modelos entrenados"""
    models = {
        'agent_15x15': {
            'q_table': dict(agent_15x15.q_table),
            'hyperparameters': {
                'learning_rate': agent_15x15.learning_rate,
                'discount_factor': agent_15x15.discount_factor,
                'epsilon': agent_15x15.epsilon,
                'epsilon_min': agent_15x15.epsilon_min,
                'epsilon_decay': agent_15x15.epsilon_decay
            }
        },
        'agent_20x20': {
            'q_table': dict(agent_20x20.q_table),
            'hyperparameters': {
                'learning_rate': agent_20x20.learning_rate,
                'discount_factor': agent_20x20.discount_factor,
                'epsilon': agent_20x20.epsilon,
                'epsilon_min': agent_20x20.epsilon_min,
                'epsilon_decay': agent_20x20.epsilon_decay
            }
        },
        'agent_25x25': {
            'q_table': dict(agent_25x25.q_table),
            'hyperparameters': {
                'learning_rate': agent_25x25.learning_rate,
                'discount_factor': agent_25x25.discount_factor,
                'epsilon': agent_25x25.epsilon,
                'epsilon_min': agent_25x25.epsilon_min,
                'epsilon_decay': agent_25x25.epsilon_decay
            }
        }
    }
    
    with open('trained_agents.pkl', 'wb') as f:
        pickle.dump(models, f)
    
    print("\n=== Modelos guardados ===")
    print("Archivo: trained_agents.pkl")
    print(f"Tamaño Q-table 15x15: {len(models['agent_15x15']['q_table'])} estados")
    print(f"Tamaño Q-table 20x20: {len(models['agent_20x20']['q_table'])} estados")
    print(f"Tamaño Q-table 25x25: {len(models['agent_25x25']['q_table'])} estados")


def save_model_25x25(agent_25x25, mejor_pasos=None):
    """Guarda solo el modelo entrenado para 25x25 en trained_agents.pkl, incluyendo mejor_pasos si se proporciona."""
    import pickle
    models = {}
    try:
        with open('trained_agents.pkl', 'rb') as f:
            models = pickle.load(f)
    except FileNotFoundError:
        pass
    hyperparams = {
        'learning_rate': agent_25x25.learning_rate,
        'discount_factor': agent_25x25.discount_factor,
        'epsilon': agent_25x25.epsilon,
        'epsilon_min': agent_25x25.epsilon_min,
        'epsilon_decay': agent_25x25.epsilon_decay
    }
    if mejor_pasos is not None:
        hyperparams['mejor_pasos'] = mejor_pasos
    models['agent_25x25'] = {
        'q_table': dict(agent_25x25.q_table),
        'hyperparameters': hyperparams
    }
    with open('trained_agents.pkl', 'wb') as f:
        pickle.dump(models, f)
    print("\n=== Modelo 25x25 guardado ===")


def create_comparative_analysis(metrics_15x15, metrics_20x20, metrics_25x25):
    """Crea análisis comparativo entre los diferentes tamaños de laberinto"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    sizes = ['15x15', '20x20', '25x25']
    metrics = [metrics_15x15, metrics_20x20, metrics_25x25]
    colors = ['blue', 'green', 'red']
    
    # Comparación de tasas de éxito
    for i, (size, (rewards, steps, success, avg_rewards, avg_steps, exploration), color) in enumerate(zip(sizes, metrics, colors)):
        axes[0, 0].plot(success, color=color, linewidth=2, label=f'Laberinto {size}')
    
    axes[0, 0].set_title('Comparación de Tasas de Éxito')
    axes[0, 0].set_xlabel('Episodio')
    axes[0, 0].set_ylabel('Tasa de Éxito (%)')
    axes[0, 0].set_ylim(0, 100)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    axes[0, 0].axhline(y=90, color='red', linestyle='--', alpha=0.7, label='Objetivo 90%')
    
    # Comparación de recompensas promedio
    for i, (size, (rewards, steps, success, avg_rewards, avg_steps, exploration), color) in enumerate(zip(sizes, metrics, colors)):
        axes[0, 1].plot(avg_rewards, color=color, linewidth=2, label=f'Laberinto {size}')
    
    axes[0, 1].set_title('Comparación de Recompensas Promedio')
    axes[0, 1].set_xlabel('Episodio')
    axes[0, 1].set_ylabel('Recompensa Promedio')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    
    # Comparación de pasos promedio
    for i, (size, (rewards, steps, success, avg_rewards, avg_steps, exploration), color) in enumerate(zip(sizes, metrics, colors)):
        axes[1, 0].plot(avg_steps, color=color, linewidth=2, label=f'Laberinto {size}')
    
    axes[1, 0].set_title('Comparación de Pasos Promedio')
    axes[1, 0].set_xlabel('Episodio')
    axes[1, 0].set_ylabel('Pasos Promedio')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    
    # Comparación de tasas de exploración
    for i, (size, (rewards, steps, success, avg_rewards, avg_steps, exploration), color) in enumerate(zip(sizes, metrics, colors)):
        axes[1, 1].plot(exploration, color=color, linewidth=2, label=f'Laberinto {size}')
    
    axes[1, 1].set_title('Comparación de Tasas de Exploración')
    axes[1, 1].set_xlabel('Episodio')
    axes[1, 1].set_ylabel('Epsilon (%)')
    axes[1, 1].set_ylim(0, 100)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig('comparative_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()


def create_performance_summary(metrics_15x15, metrics_20x20, metrics_25x25):
    """Crea un resumen visual del rendimiento final de todos los agentes"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    sizes = ['15x15', '20x20', '25x25']
    metrics = [metrics_15x15, metrics_20x20, metrics_25x25]
    colors = ['skyblue', 'lightgreen', 'lightcoral']
    
    # Datos para el resumen
    final_success_rates = []
    final_avg_rewards = []
    final_avg_steps = []
    
    for rewards, steps, success, avg_rewards, avg_steps, exploration in metrics:
        final_success_rates.append(success[-1] if success else 0)
        final_avg_rewards.append(avg_rewards[-1] if avg_rewards else 0)
        final_avg_steps.append(avg_steps[-1] if avg_steps else 0)
    
    # Gráfico de barras para tasas de éxito finales
    bars1 = axes[0].bar(sizes, final_success_rates, color=colors, alpha=0.8)
    axes[0].set_title('Tasa de Éxito Final por Laberinto')
    axes[0].set_ylabel('Tasa de Éxito (%)')
    axes[0].set_ylim(0, 100)
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Agregar valores en las barras
    for bar, value in zip(bars1, final_success_rates):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                    f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # Gráfico de barras para recompensas promedio finales
    bars2 = axes[1].bar(sizes, final_avg_rewards, color=colors, alpha=0.8)
    axes[1].set_title('Recompensa Promedio Final por Laberinto')
    axes[1].set_ylabel('Recompensa Promedio')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # Agregar valores en las barras
    for bar, value in zip(bars2, final_avg_rewards):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + (max(final_avg_rewards) * 0.01), 
                    f'{value:.1f}', ha='center', va='bottom', fontweight='bold')
    
    # Gráfico de barras para pasos promedio finales
    bars3 = axes[2].bar(sizes, final_avg_steps, color=colors, alpha=0.8)
    axes[2].set_title('Pasos Promedio Final por Laberinto')
    axes[2].set_ylabel('Pasos Promedio')
    axes[2].grid(True, alpha=0.3, axis='y')
    
    # Agregar valores en las barras
    for bar, value in zip(bars3, final_avg_steps):
        axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + (max(final_avg_steps) * 0.01), 
                    f'{value:.1f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('performance_summary.png', dpi=300, bbox_inches='tight')
    plt.show()


def generate_training_report(metrics_15x15, metrics_20x20, metrics_25x25, agent_15x15, agent_20x20, agent_25x25):
    """Genera un reporte completo de entrenamiento en formato texto"""
    report = []
    report.append("=" * 60)
    report.append("REPORTE COMPLETO DE ENTRENAMIENTO Q-LEARNING")
    report.append("=" * 60)
    report.append(f"Fecha de generación: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    sizes = ['15x15', '20x20', '25x25']
    metrics = [metrics_15x15, metrics_20x20, metrics_25x25]
    agents = [agent_15x15, agent_20x20, agent_25x25]
    
    for i, (size, (rewards, steps, success, avg_rewards, avg_steps, exploration), agent) in enumerate(zip(sizes, metrics, agents)):
        report.append(f"\n{'='*20} LABERINTO {size} {'='*20}")
        
        # Información del agente
        report.append(f"\n--- CONFIGURACIÓN DEL AGENTE ---")
        report.append(f"Learning Rate: {agent.learning_rate}")
        report.append(f"Discount Factor: {agent.discount_factor}")
        report.append(f"Epsilon inicial: {1.0}")
        report.append(f"Epsilon final: {agent.epsilon:.4f}")
        report.append(f"Epsilon mínimo: {agent.epsilon_min}")
        report.append(f"Epsilon decay: {agent.epsilon_decay}")
        report.append(f"Estados en Q-table: {len(agent.q_table)}")
        
        # Estadísticas de entrenamiento
        report.append(f"\n--- ESTADÍSTICAS DE ENTRENAMIENTO ---")
        report.append(f"Episodios totales: {len(rewards)}")
        
        # Recompensas
        final_avg_reward = avg_rewards[-1] if avg_rewards else 0
        max_reward = max(rewards) if rewards else 0
        min_reward = min(rewards) if rewards else 0
        std_reward = np.std(rewards) if rewards else 0
        
        report.append(f"\nRecompensas:")
        report.append(f"  - Promedio final: {final_avg_reward:.2f}")
        report.append(f"  - Máxima: {max_reward:.2f}")
        report.append(f"  - Mínima: {min_reward:.2f}")
        report.append(f"  - Desviación estándar: {std_reward:.2f}")
        
        # Pasos
        final_avg_steps = avg_steps[-1] if avg_steps else 0
        min_steps = min(steps) if steps else 0
        max_steps = max(steps) if steps else 0
        std_steps = np.std(steps) if steps else 0
        
        report.append(f"\nPasos:")
        report.append(f"  - Promedio final: {final_avg_steps:.2f}")
        report.append(f"  - Mínimo: {min_steps}")
        report.append(f"  - Máximo: {max_steps}")
        report.append(f"  - Desviación estándar: {std_steps:.2f}")
        
        # Tasa de éxito
        final_success_rate = success[-1] if success else 0
        max_success_rate = max(success) if success else 0
        avg_success_rate = np.mean(success) if success else 0
        
        report.append(f"\nTasa de éxito:")
        report.append(f"  - Final: {final_success_rate:.1f}%")
        report.append(f"  - Máxima: {max_success_rate:.1f}%")
        report.append(f"  - Promedio: {avg_success_rate:.1f}%")
        
        # Análisis de convergencia
        if len(success) > 100:
            last_100_success = success[-100:]
            convergence_rate = np.mean(last_100_success)
            report.append(f"\nAnálisis de convergencia:")
            report.append(f"  - Tasa en últimos 100 episodios: {convergence_rate:.1f}%")
            
            if convergence_rate >= 80:
                report.append(f"  - Estado: ✅ CONVERGIDO EXITOSAMENTE")
            elif convergence_rate >= 50:
                report.append(f"  - Estado: ⚠️ PROGRESO DETECTADO")
            else:
                report.append(f"  - Estado: ❌ NECESITA MÁS ENTRENAMIENTO")
        
        # Exploración
        final_exploration = exploration[-1] if exploration else 0
        initial_exploration = exploration[0] if exploration else 0
        
        report.append(f"\nExploración:")
        report.append(f"  - Inicial: {initial_exploration:.1f}%")
        report.append(f"  - Final: {final_exploration:.1f}%")
        report.append(f"  - Reducción: {initial_exploration - final_exploration:.1f}%")
    
    # Comparación entre laberintos
    report.append(f"\n{'='*20} COMPARACIÓN ENTRE LABERINTOS {'='*20}")
    
    final_success_rates = []
    final_avg_rewards = []
    final_avg_steps = []
    
    for rewards, steps, success, avg_rewards, avg_steps, exploration in metrics:
        final_success_rates.append(success[-1] if success else 0)
        final_avg_rewards.append(avg_rewards[-1] if avg_rewards else 0)
        final_avg_steps.append(avg_steps[-1] if avg_steps else 0)
    
    report.append(f"\nTasas de éxito finales:")
    for size, rate in zip(sizes, final_success_rates):
        report.append(f"  - {size}: {rate:.1f}%")
    
    report.append(f"\nRecompensas promedio finales:")
    for size, reward in zip(sizes, final_avg_rewards):
        report.append(f"  - {size}: {reward:.2f}")
    
    report.append(f"\nPasos promedio finales:")
    for size, step in zip(sizes, final_avg_steps):
        report.append(f"  - {size}: {step:.2f}")
    
    # Recomendaciones
    report.append(f"\n{'='*20} RECOMENDACIONES {'='*20}")
    
    best_success = max(final_success_rates)
    best_success_idx = final_success_rates.index(best_success)
    best_success_size = sizes[best_success_idx]
    
    report.append(f"Mejor rendimiento: Laberinto {best_success_size} ({best_success:.1f}% de éxito)")
    
    for i, rate in enumerate(final_success_rates):
        if rate < 80:
            report.append(f"- El laberinto {sizes[i]} necesita más entrenamiento (actual: {rate:.1f}%)")
        elif rate < 90:
            report.append(f"- El laberinto {sizes[i]} está bien entrenado pero podría mejorar (actual: {rate:.1f}%)")
        else:
            report.append(f"- El laberinto {sizes[i]} está excelentemente entrenado (actual: {rate:.1f}%)")
    
    report.append(f"\n{'='*60}")
    report.append("FIN DEL REPORTE")
    report.append("=" * 60)
    
    # Guardar reporte en archivo
    with open('training_report.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    
    print("Reporte de entrenamiento guardado en 'training_report.txt'")
    
    # Mostrar resumen en consola
    print("\n" + "="*60)
    print("RESUMEN DEL ENTRENAMIENTO")
    print("="*60)
    for size, rate, reward, steps in zip(sizes, final_success_rates, final_avg_rewards, final_avg_steps):
        print(f"Laberinto {size}: {rate:.1f}% éxito, {reward:.1f} recompensa, {steps:.1f} pasos")


def visualize_q_tables(agent_15x15, agent_20x20, agent_25x25):
    """Visualiza las Q-tables de los tres agentes entrenados"""
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    
    agents = [agent_15x15, agent_20x20, agent_25x25]
    sizes = ['15x15', '20x20', '25x25']
    maze_sizes = [15, 20, 25]
    action_names = ['Arriba', 'Derecha', 'Abajo', 'Izquierda']
    
    for i, (agent, size, maze_size) in enumerate(zip(agents, sizes, maze_sizes)):
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
                        q_matrices[action][y, x] = q_values.get(action, 0)
        
        # Visualizar cada acción
        for action in range(4):
            ax = axes[i, action]
            q_matrix = q_matrices[action]
            
            # Crear mapa de calor
            im = ax.imshow(q_matrix, cmap='RdYlBu_r', aspect='equal')
            ax.set_title(f'{size} - {action_names[action]}')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            
            # Agregar valores en las celdas
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
    
    # Generar estadísticas de las Q-tables
    print("\n=== ESTADÍSTICAS DE LAS Q-TABLES ===")
    for i, (agent, size) in enumerate(zip(agents, sizes)):
        q_values = []
        for state_q in agent.q_table.values():
            q_values.extend(state_q.values())
        
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


def create_q_table_heatmap(agent, maze_size, title):
    """Crea un mapa de calor de la Q-table para un agente específico"""
    # Crear matriz de valores Q promedio por estado
    q_avg_matrix = np.zeros((maze_size, maze_size))
    q_max_matrix = np.zeros((maze_size, maze_size))
    
    for state, q_values in agent.q_table.items():
        if isinstance(state, tuple) and len(state) == 2:
            y, x = state
            if 0 <= y < maze_size and 0 <= x < maze_size:
                values = list(q_values.values())
                if values:
                    q_avg_matrix[y, x] = np.mean(values)
                    q_max_matrix[y, x] = max(values)
    
    return q_avg_matrix, q_max_matrix


def visualize_q_table_summary(agent_15x15, agent_20x20, agent_25x25):
    """Crea un resumen visual de las Q-tables mostrando valores promedio y máximos"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    agents = [agent_15x15, agent_20x20, agent_25x25]
    sizes = ['15x15', '20x20', '25x25']
    maze_sizes = [15, 20, 25]
    
    for i, (agent, size, maze_size) in enumerate(zip(agents, sizes, maze_sizes)):
        q_avg, q_max = create_q_table_heatmap(agent, maze_size, size)
        
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


def main():
    """Función principal de entrenamiento"""
    print("=== Entrenamiento de Agentes Q-Learning para Laberintos ===")
    
    # Entrenar agente para laberinto 15x15
    agent_15x15, rewards_15x15, steps_15x15, success_15x15, avg_rewards_15x15, avg_steps_15x15, exploration_rate_15x15 = train_agent(15, episodes=5000)
    test_agent(agent_15x15, 15)
    
    # Entrenar agente para laberinto 20x20
    agent_20x20, rewards_20x20, steps_20x20, success_20x20, avg_rewards_20x20, avg_steps_20x20, exploration_rate_20x20 = train_agent(20, episodes=7000)
    test_agent(agent_20x20, 20)
    
    # Entrenar agente para laberinto 25x25
    agent_25x25, rewards_25x25, steps_25x25, success_25x25, avg_rewards_25x25, avg_steps_25x25, exploration_rate_25x25 = train_agent_25x25()
    test_agent(agent_25x25, 25)
    
    # Generar estadísticas detalladas
    generate_training_statistics(
        (rewards_15x15, steps_15x15, success_15x15, avg_rewards_15x15, avg_steps_15x15, exploration_rate_15x15),
        (rewards_20x20, steps_20x20, success_20x20, avg_rewards_20x20, avg_steps_20x20, exploration_rate_20x20),
        (rewards_25x25, steps_25x25, success_25x25, avg_rewards_25x25, avg_steps_25x25, exploration_rate_25x25)
    )
    
    # Visualizar métricas detalladas
    plot_training_metrics(
        (rewards_15x15, steps_15x15, success_15x15, avg_rewards_15x15, avg_steps_15x15, exploration_rate_15x15),
        (rewards_20x20, steps_20x20, success_20x20, avg_rewards_20x20, avg_steps_20x20, exploration_rate_20x20),
        (rewards_25x25, steps_25x25, success_25x25, avg_rewards_25x25, avg_steps_25x25, exploration_rate_25x25)
    )
    
    # Crear análisis comparativo
    create_comparative_analysis(
        (rewards_15x15, steps_15x15, success_15x15, avg_rewards_15x15, avg_steps_15x15, exploration_rate_15x15),
        (rewards_20x20, steps_20x20, success_20x20, avg_rewards_20x20, avg_steps_20x20, exploration_rate_20x20),
        (rewards_25x25, steps_25x25, success_25x25, avg_rewards_25x25, avg_steps_25x25, exploration_rate_25x25)
    )
    
    # Crear resumen de rendimiento
    create_performance_summary(
        (rewards_15x15, steps_15x15, success_15x15, avg_rewards_15x15, avg_steps_15x15, exploration_rate_15x15),
        (rewards_20x20, steps_20x20, success_20x20, avg_rewards_20x20, avg_steps_20x20, exploration_rate_20x20),
        (rewards_25x25, steps_25x25, success_25x25, avg_rewards_25x25, avg_steps_25x25, exploration_rate_25x25)
    )
    
    # Visualizar Q-tables
    visualize_q_tables(agent_15x15, agent_20x20, agent_25x25)
    visualize_q_table_summary(agent_15x15, agent_20x20, agent_25x25)
    
    # Guardar modelos
    save_models(agent_15x15, agent_20x20, agent_25x25)
    
    # Guardar solo el modelo 25x25
    save_model_25x25(agent_25x25)
    
    # Generar reporte de entrenamiento
    generate_training_report(
        (rewards_15x15, steps_15x15, success_15x15, avg_rewards_15x15, avg_steps_15x15, exploration_rate_15x15),
        (rewards_20x20, steps_20x20, success_20x20, avg_rewards_20x20, avg_steps_20x20, exploration_rate_20x20),
        (rewards_25x25, steps_25x25, success_25x25, avg_rewards_25x25, avg_steps_25x25, exploration_rate_25x25),
        agent_15x15, agent_20x20, agent_25x25
    )
    
    print("\n=== Entrenamiento completado ===")
    print("Archivos generados:")
    print("- training_metrics.png: Métricas detalladas de entrenamiento")
    print("- comparative_analysis.png: Análisis comparativo entre laberintos")
    print("- performance_summary.png: Resumen de rendimiento final")
    print("- q_tables_visualization.png: Visualización detallada de Q-tables")
    print("- q_tables_summary.png: Resumen de Q-tables")
    print("- training_report.txt: Reporte completo de entrenamiento")


if __name__ == "__main__":
    main()