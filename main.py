import environment
import yaml
import time
from memory import Memory



def test(config):
    isTrain = config.setdefault("isTrain", True)
    experiment_name = config.setdefault("experiment_name", "")
    time_slot = config.setdefault("time_slot", 10000)
    memory_size = config.setdefault("memory_size", 1000)

    print("==========")
    print("Experiment: " + experiment_name)
    print("==========")

    memory = Memory(max_size=memory_size)

    start_time = time.time()
    episode = 1

    env = environment.Env(**config["EnvironmentParams"])
    num_vehicles = env.get_num_vehicles()
    num_actions = env.get_num_actions()

if __name__ == '__main__':
    config = yaml.load(open("./experiment.yaml", Loader=yaml.FullLoader))
    test(config)
