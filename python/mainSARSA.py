import os
import sys
import fire
import traci
import random
import torch
import generate_route
from linear_rl.true_online_sarsa import TrueOnlineSarsaLambda
from sumo_rl import SumoEnvironment
from datetime import datetime



# Funzione per inizializzare l'ambiente SUMO con parametri predefiniti
def initialize_sumo_environment(n):
    out_csv = "outputs/double/sarsa-double"

    return SumoEnvironment(
        net_file="C:/Users/drugo/Desktop/PROGETTO-IA/XML/TEST1.net.xml",
        route_file="C:/Users/drugo/Desktop/PROGETTO-IA/XML/trips.trips.xml",
        use_gui=False,
        single_agent=False,
        num_seconds=n+150,  # Durata della simulazione in secondi
        min_green=5,      # Durata minima del semaforo verde
        delta_time=5,     # Intervallo di tempo tra le iterazioni
        #max_green=60,
        #yellow_time=3,
        additional_sumo_cmd= '--collision.stoptime 10',
    )


# Funzione per inizializzare gli agenti SARSA con parametri specifici
def initialize_SARSA_agents(env):
    agents = {
        ts_id: TrueOnlineSarsaLambda(
            env.observation_spaces(ts_id),
            env.action_spaces(ts_id),
            alpha=0.5,
            gamma=0.99,
            epsilon=0.2,
            lamb=1,
            fourier_order=7,
        )
        for ts_id in env.ts_ids
    }
    return agents

# Funzione per addestrare gli agenti Q-learning sull'ambiente SUMO
def train_agents(env, agents, run, episodes):
    obs = env.reset()
    for episode in range(1, episodes + 1):
        
        done = {"__all__": False}

        while not done["__all__"]:
            actions = {ts_id: agents[ts_id].act(obs[ts_id]) for ts_id in obs.keys()}
            next_obs, r, done, _ = env.step(action=actions)

            for ts_id in next_obs.keys():
                agents[ts_id].learn(
                    state=obs[ts_id], action=actions[ts_id], reward=r[ts_id], next_state=next_obs[ts_id], done=done[ts_id]
                )
                obs[ts_id] = next_obs[ts_id]


        # Salvataggio dei dati in formato CSV alla fine di ogni episodio
        env.save_csv(f"outputs/sarsa/sarsa_run{run}", episode)


def plot(runs,episodes):
    for j in range(runs):
        for i in range(episodes):
            os.system('cmd /c "py "C:/Users/drugo/Desktop/PROGETTO-IA/Python/plot.py" -f "C:/Users/drugo/Desktop/PROGETTO-IA/outputs/sarsa/sarsa_run{}_conn{}_ep{}.csv""'.format(j+1,j,i+1))


if __name__ == "__main__":
     # Parametri di apprendimento
    runs = 3 # Numero di iterazioni
    episodes = 10  # Numero di episodi per iterazione


    # Loop sul numero di iterazioni
    for run in range(1, runs + 1):
        
        n=100*run
        generate_route.generate_route_file(n)
        env = initialize_sumo_environment(n)
        # Inizializzazione degli agenti Q-learning per l'ambiente SUMO

        agents = initialize_SARSA_agents(env)
        
        # Addestramento degli agenti Q-learning
        train_agents(env, agents, run, episodes)
    
    # Chiusura dell'ambiente SUMO alla fine dell'esecuzione
    
    torch.save(agents,r"C:\Users\drugo\Desktop\PROGETTO-IA\models\SARSA_Model.pth")
    env.close()
    plot(runs,episodes)
