from workerstationprob import WorkerStationProblem
import yaml

def parse_yaml(config_file):
    """
    Function to parse the yaml configuration file
    and returns a dict.
    """
    with open(config_file, 'r') as file:
        try:
            cfg = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
        print(cfg)
        return cfg


def create_experiment(cfg):
    """
    Function to create the experiment from the given configuration
    :param cfg: Configuration as a nested dict (can be from Yaml or a JSON file)
    :return: None
    """
    setup_cfg = cfg['worker_station_problem']
    w = WorkerStationProblem(num_workers= setup_cfg['Number_of_Workers'],
                             num_stations= setup_cfg['Number_of_Stations'],
                             num_time_steps= setup_cfg['Number_of_Timesteps'],
                             demands= setup_cfg['Demands'])

    for i in range(setup_cfg['Number_of_Workers']):
        worker_cfg = cfg['Worker_{}_info'.format(i+1)]
        w.add_worker(lr = worker_cfg['Learning_rate'],
                     fr = worker_cfg['Forgetting_rate'],
                     initial_skill= worker_cfg['Initial_skill'],
                     station_id= worker_cfg['Initial_station'])

    for i in range(setup_cfg['Number_of_Stations']):
        station_cfg = cfg['Station_{}_info'.format(i+1)]
        w.add_station(S_max= station_cfg['Max_skill_level'],
                      S_min= station_cfg['Min_skill_level'],
                      delta= station_cfg['Delta'],
                      eps= station_cfg['Epsilon'])
    w.Solve()


if __name__ == '__main__':
    # w = WorkerStationProblem(num_workers=4, num_stations=4, num_time_steps=3, demands=[12, 12, 5])
    cfg = parse_yaml('config.yaml')
    create_experiment(cfg)