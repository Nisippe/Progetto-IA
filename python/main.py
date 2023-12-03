import argparse
import os
import sys
import torch
import pandas as pd
import time
import traci
import random
import generate_route
from sumo_rl import SumoEnvironment
from sumo_rl.agents import QLAgent
from sumo_rl.exploration import EpsilonGreedy

# Funzione per inizializzare l'ambiente SUMO con parametri predefiniti
def initialize_sumo_environment(n):
    return SumoEnvironment(
        net_file="C:/Users/drugo/Desktop/PROGETTO-IA/XML/TEST1.net.xml",
        route_file="C:/Users/drugo/Desktop/PROGETTO-IA/XML/trips.trips.xml",
        use_gui=True,
        num_seconds=n+100,  # Durata della simulazione in secondi
        min_green=5,      # Durata minima del semaforo verde
        delta_time=5,     # Intervallo di tempo tra le iterazioni
        additional_sumo_cmd= '--collision.stoptime 30',
    )

# Funzione per inizializzare gli agenti Q-learning con parametri specifici
def initialize_q_learning_agents(env, alpha, gamma, decay):
    initial_states = env.reset()  # Stati iniziali dell'ambiente
    ql_agents = {
        ts: QLAgent(
            starting_state=env.encode(initial_states[ts], ts),  # Stato iniziale codificato
            state_space=env.observation_space,  # Spazio degli stati
            action_space=env.action_space,      # Spazio delle azioni
            alpha=alpha,  # Tasso di apprendimento
            gamma=gamma,  # Fattore di sconto
            exploration_strategy=EpsilonGreedy(initial_epsilon=0.05, min_epsilon=0.005, decay=decay),  # Strategia di esplorazione
        )
        for ts in env.ts_ids  # Loop su tutti i semafori nell'ambiente SUMO
    }
    return ql_agents

# Funzione per addestrare gli agenti Q-learning sull'ambiente SUMO
def train_agents(env, ql_agents, episodes,run):
    
    for episode in range(1, episodes + 1):  # Loop su ogni episodio
        if episode != 1:
            initial_states = env.reset()  # Reset degli stati iniziali
            for ts in initial_states.keys():
                ql_agents[ts].state = env.encode(initial_states[ts], ts)  # Imposta lo stato degli agenti

        done = {"__all__": False}  # Indicatore di completamento dell'episodio
        while not done["__all__"]:  # Loop finché non si completa l'episodio
            actions = {ts: ql_agents[ts].act() for ts in ql_agents.keys()}  # Scelta delle azioni
            s, r, done, info = env.step(action=actions)  # Esecuzione delle azioni sull'ambiente

            for agent_id in s.keys():
                # Apprendimento degli agenti con stato successivo e ricompensa
                ql_agents[agent_id].learn(next_state=env.encode(s[agent_id], agent_id), reward=r[agent_id])

        # Salvataggio dei dati in formato CSV alla fine di ogni episodio
        env.save_csv(f"outputs/grid_run{run}", episode)


# Main
def main():
    # Parametri di apprendimento
    alpha = 0.000000001  # Tasso di apprendimento
    gamma = 0.95  # Fattore di sconto
    decay = 1  # Parametro di decadimento
    runs = 2  # Numero di iterazioni
    episodes = 5  # Numero di episodi per iterazione
    n=random.randint(200,500)

    # Inizializzazione dell'ambiente SUMO
    env = initialize_sumo_environment(n)

    # Loop sul numero di iterazioni
    for run in range(1, runs + 1):
        
        # Inizializzazione degli agenti Q-learning per l'ambiente SUMO
        ql_agents = initialize_q_learning_agents(env, alpha, gamma, decay)
        
        # Addestramento degli agenti Q-learning
        train_agents(env, ql_agents, episodes,run)

    # Chiusura dell'ambiente SUMO alla fine dell'esecuzione
    env.close()

# Esecuzione della funzione main al lancio dello script
if __name__ == "__main__":
    main()
