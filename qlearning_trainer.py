import numpy as np
import pickle
import matplotlib.pyplot as plt
from collections import defaultdict
import random
from tqdm import tqdm
from maze import Maze  # Importar la clase Maze desde maze.py

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
    
    # Variables para tracking
    recent_successes = []
    
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
        
        # Calcular tasa de éxito (últimos 100 episodios)
        success = total_reward == 99  # 100 - 1 (por el último movimiento)
        recent_successes.append(success)
        if len(recent_successes) > 100:
            recent_successes.pop(0)
        
        success_rate.append(np.mean(recent_successes) * 100)
        
        # Mostrar progreso cada 1000 episodios
        if (episode + 1) % 1000 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            avg_steps = np.mean(episode_steps[-100:])
            print(f"Episodio {episode + 1}: Reward promedio: {avg_reward:.2f}, "
                  f"Pasos promedio: {avg_steps:.2f}, Epsilon: {agent.epsilon:.3f}, "
                  f"Tasa de éxito: {success_rate[-1]:.1f}%")
    
    return agent, episode_rewards, episode_steps, success_rate


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


def plot_training_metrics(metrics_15x15, metrics_20x20, metrics_25x25):
    """Visualiza las métricas de entrenamiento"""
    fig, axes = plt.subplots(3, 3, figsize=(18, 18))
    
    sizes = ['15x15', '20x20', '25x25']
    metrics = [metrics_15x15, metrics_20x20, metrics_25x25]
    
    for i, (size, (rewards, steps, success)) in enumerate(zip(sizes, metrics)):
        # Reward por episodio
        axes[i, 0].plot(rewards, alpha=0.6)
        axes[i, 0].plot(np.convolve(rewards, np.ones(100)/100, mode='valid'), 'r-', linewidth=2)
        axes[i, 0].set_title(f'Rewards - Laberinto {size}')
        axes[i, 0].set_xlabel('Episodio')
        axes[i, 0].set_ylabel('Reward')
        axes[i, 0].grid(True, alpha=0.3)
        
        # Pasos por episodio
        axes[i, 1].plot(steps, alpha=0.6)
        axes[i, 1].plot(np.convolve(steps, np.ones(100)/100, mode='valid'), 'r-', linewidth=2)
        axes[i, 1].set_title(f'Pasos por Episodio - Laberinto {size}')
        axes[i, 1].set_xlabel('Episodio')
        axes[i, 1].set_ylabel('Pasos')
        axes[i, 1].grid(True, alpha=0.3)
        
        # Tasa de éxito
        axes[i, 2].plot(success, 'g-', linewidth=2)
        axes[i, 2].set_title(f'Tasa de Éxito - Laberinto {size}')
        axes[i, 2].set_xlabel('Episodio')
        axes[i, 2].set_ylabel('Tasa de Éxito (%)')
        axes[i, 2].grid(True, alpha=0.3)
    
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
    recent_successes = []
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
        success = reward == 1000
        recent_successes.append(success)
        if len(recent_successes) > 100:
            recent_successes.pop(0)
        tasa_exito = np.mean(recent_successes) * 100
        success_rate.append(tasa_exito)
        if (episode + 1) % 1000 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            avg_steps = np.mean(episode_steps[-100:])
            print(f"Episodio {episode + 1}: Reward promedio: {avg_reward:.2f}, "
                  f"Pasos promedio: {avg_steps:.2f}, Epsilon: {agent.epsilon:.3f}, "
                  f"Tasa de éxito: {success_rate[-1]:.1f}%")
        # Early stopping: si la tasa de éxito es >= 90% en los últimos 100 episodios
        if len(recent_successes) == 100 and tasa_exito >= 90.0:
            print(f"Entrenamiento detenido anticipadamente en el episodio {episode+1} por alta tasa de éxito ({tasa_exito:.1f}%)")
            break
    return agent, episode_rewards, episode_steps, success_rate


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


def main():
    """Función principal de entrenamiento"""
    print("=== Entrenamiento de Agentes Q-Learning para Laberintos ===")
    
    # Entrenar agente para laberinto 15x15
    agent_15x15, rewards_15x15, steps_15x15, success_15x15 = train_agent(15, episodes=5000)
    test_agent(agent_15x15, 15)
    
    # Entrenar agente para laberinto 20x20
    agent_20x20, rewards_20x20, steps_20x20, success_20x20 = train_agent(20, episodes=7000)
    test_agent(agent_20x20, 20)
    
    # Entrenar agente para laberinto 25x25
    agent_25x25, rewards_25x25, steps_25x25, success_25x25 = train_agent_25x25()
    test_agent(agent_25x25, 25)
    
    # Visualizar métricas
    plot_training_metrics(
        (rewards_15x15, steps_15x15, success_15x15),
        (rewards_20x20, steps_20x20, success_20x20),
        (rewards_25x25, steps_25x25, success_25x25)
    )
    
    # Guardar modelos
    save_models(agent_15x15, agent_20x20, agent_25x25)
    
    # Guardar solo el modelo 25x25
    save_model_25x25(agent_25x25)
    
    print("\n=== Entrenamiento completado ===")


if __name__ == "__main__":
    main()