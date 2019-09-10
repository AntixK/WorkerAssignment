import os
import yaml
import logging
import numpy as np
import pandas as pd
from time import time
from tqdm import trange
from typing import List
from datetime import datetime
import matplotlib.pyplot as plt
from ortools.graph import pywrapgraph


plt.style.use('seaborn')


class Worker:
    """
    Worker class to store all the variables associated with a
    given worker.
    """
    def __init__(self, id: int,
                 learning_rate: float,
                 forgetting_rate: float,
                 initial_skill: List[int],
                 station_id: int):
        """
        Initialise the variables associated with the current worker
        :param id (int): Worker ID
        :param learning_rate (float): Learning rate of the worker
        :param forgetting_rate (float): Forgetting rate of the worker
        :param initial_skill (list of ints): Skill levels for each station; length
                                             must be equal to the number of stations
        :param station_id (int): Initial station assignment
        """
        self.id = id
        self.beta = np.log2(learning_rate)
        self.gamma = np.log2(forgetting_rate)
        self.initial_skill = initial_skill

        # Remaining skills of the worker for each station
        self.rem_skill = initial_skill
        self.assign_history = [station_id]


class Station:
    """
    Station class to store all the variables associated with a
    given station.
    """
    def __init__(self, id: int,
                 S_max: float,
                 S_min: float,
                 delta: float,
                 eps: float):
        """
        Initialise the variables associated with the current station.
        :param id (int): ID of the current worker
        :param S_max (float): Max theoretical skill level attainable
        :param S_min (float): Min theoretical skill level attainable
        :param delta (float): Practical max skill threshold
        :param eps (float): Practical min skill threshold
        """
        self.id = id
        self.S_max = S_max
        self.S_min = S_min

        # Practically achievable skill levels
        self.S_UB = int(S_max - delta)
        self.S_LB = int(S_min + eps)

        self.delta = delta
        self.eps = eps


class WorkerStationProblem:
    def __init__(self, num_workers: int,
                 num_stations: int,
                 num_time_steps: int,
                 demands: List[int], experiment_name: str):
        """
        Initial setup for the Worker-Station problem
        :param num_workers: Number of workers in the problem
        :param num_stations: Number of stations in the problem;
                             Must be equal to num_workers
        :param num_time_steps: Number of time steps to optimize
        :param demands: List of ints containing demands at each time step
        """

        assert num_workers == num_stations, "Number of workers must be equal to" \
                                            "the number of work stations"
        assert num_time_steps == len(demands), "Invalid number of demands"

        self.num_workers = num_workers
        self.num_stations = num_stations
        self.num_time_steps = num_time_steps
        self.demands = demands

        self.source_id = 0
        self.worker_id = list(range(1, self.num_workers + 1))
        self.station_id = list(range(self.num_workers + 1, self.num_stations + self.num_stations + 1))
        self.sink_id = self.num_stations + self.num_workers + 1
        self.WIP_inventory = [0]*self.num_stations
        self.output_inventory = [0]*self.num_time_steps

        self.inventory_log = np.empty((0, 3))

        self.mean_units = 0
        self.actual_units = [0] * self.num_stations
        self.final_num_units = []

        self.experiment_time = 0

        self.Q = []

        # List to store the Worker and Station class objects
        self.workers = []
        self.stations = []

        # Handle for the MinCostFlow solver
        self.solver = pywrapgraph.SimpleMinCostFlow()

        self.SAVE_PATH = experiment_name + '/'
        os.makedirs(self.SAVE_PATH, exist_ok=True)

        if os.path.exists(self.SAVE_PATH + experiment_name+'.log'):
            os.remove(self.SAVE_PATH + experiment_name+'.log')
        logging.basicConfig(filename=self.SAVE_PATH + experiment_name+'.log', level=logging.INFO, format='%(message)s')
        logging.info("Experiment Date: {} \n".format(datetime.now().strftime("%d/%m/%Y %H:%M:%S")))
        logging.info("Demands: {}".format(self.demands))

        cols = []
        for i in range(self.num_time_steps):
            cols.append('Time Step {}'.format(i+1))
        self.pd_file = pd.DataFrame(columns=cols)

    def _build_graph(self, cost_fn):
        """
        Function to convert the Worker-Station assignment problem
        into a Min-cost flow problem. Based on the above given data,
        this function builds a graph, which can then be passed to the
        solver.

        Reference -
        [1] https://developers.google.com/optimization/assignment/assignment_min_cost_flow
        """

        """
        Graph Setup -
                ID          | Meaning
                ---------------------
                0           | sink
                1,2,..., N  | workers
                N+1, ...,2N | stations
                2N+1        | sink
                ----------------------
        """
        self.start_nodes = [0]*self.num_workers + \
                           sum([[i+1]*self.num_stations for i in range(self.num_workers)], []) + \
                           list(range(self.num_workers+1, self.num_workers + self.num_stations+1))

        self.end_nodes = list(range(1, self.num_workers+1)) +\
                         list(range(self.num_stations+1, 2*self.num_stations+1))*self.num_workers + \
                         [2*self.num_stations+1]*self.num_stations

        """
        Set the capacities all to one as this is just a simple mapping problem
        and we do not have any cost associated with the mapping itself.
        """
        self.capacities = [1]*self.num_workers
        self.capacities += ([1]*self.num_workers)*self.num_stations
        self.capacities += [1]*self.num_stations

        """
        Computation of costs
            - Zero costs for the links from the Source -> Worker
            - User-defined cost for Worker -> Station
            - Zero costs for links from Worker -> Sink
        """
        self.costs = [0]*self.num_workers

        # Compute the Worker - Station costs
        for worker_id in range(self.num_workers):
            for station_id in range(self.num_stations):
                cost = cost_fn(worker_id, station_id)
                self.costs.append(cost)

        self.costs += [0]*self.num_stations

        """
        Computation of Supplies
            - Source supply = number of workers
            - Zero supplies from  Worker -> Station
            - Sink supply = - number of stations
        [Law of conservation of resources]
        """
        self.supplies = [self.num_workers]
        self.supplies += [0]*(self.num_workers+self.num_stations)
        self.supplies += [-self.num_stations]

        # Add each link(arc) to the solver graph
        for i in range(len(self.start_nodes)):
            self.solver.AddArcWithCapacityAndUnitCost(self.start_nodes[i], self.end_nodes[i],
                                                        self.capacities[i], self.costs[i])

        # Add node supplies to the solver graph
        for i in range(len(self.supplies)):
            self.solver.SetNodeSupply(i, self.supplies[i])

    def run_optimizer(self, log_history):
        """
        Given a graph, this function runs the Mincostflow algorithm
        :log_history: Boolean to control when to log the history of the
                      assignment to station. This is helpful during the
                      second phase of optimization, which is actually
                      the final assignment, and hence needs to be logged.

        :return: None
        """

        # Find the minimum cost flow between node 0 and node 10.
        if self.solver.Solve() == self.solver.OPTIMAL:
            logging.info('Total Minimum cost = {}'.format(self.solver.OptimalCost()))
            for arc in range(self.solver.NumArcs()):

                # Can ignore arcs leading out of source or into sink.
                if self.solver.Tail(arc) != self.source_id and self.solver.Head(arc) != self.sink_id:

                    # Arcs in the solution have a flow value of 1. Their start and end nodes
                    # give an assignment of worker to task.

                    if self.solver.Flow(arc) > 0:
                        worker_id = self.solver.Tail(arc)
                        station_id = self.solver.Head(arc) - self.num_workers

                        self.theoretical_units_produced[station_id - 1] = \
                            self.workers[worker_id - 1].rem_skill[station_id - 1]
                        units_produced = self.workers[worker_id - 1].rem_skill[station_id - 1]

                        if log_history:
                            self.workers[worker_id - 1].assign_history.append(station_id)

                        logging.info('Worker {:d} assigned to Station {:d}, capable of {:d} units; Deficit = {:d}'.format(
                            worker_id,
                            station_id,
                            units_produced,
                            self.solver.UnitCost(arc)))

        elif self.solver.Solve() == self.solver.INFEASIBLE:
            raise RuntimeError("Infeasible problem input. Terminating code.")
        else:
            raise RuntimeError("Bad Result! Terminating code.")

    def _compute_units_produced(self, compute_WIP):
        # Compute the actual units produced & Inventory
        """
        This calculation is simple - the current station cannot produce
        more than its previous station (assuming empty inventory)
        """
        self.actual_units = [0] * self.num_stations
        for i in range(self.num_stations):

            self.actual_units[i] = self.theoretical_units_produced[i]

            if i > 0:

                """
                If the current station can produce more than the previous station,
                then max number of units produced by the current station is equal
                to its previous station.
                """
                if self.actual_units[i] > self.actual_units[i - 1]:
                    self.actual_units[i] = min(self.theoretical_units_produced[i],
                                               self.actual_units[i - 1] + self.WIP_inventory[i])

                if compute_WIP:
                    self.WIP_inventory[i] += self.actual_units[i - 1] - self.actual_units[i]
            logging.info('Station {:d} practically produces {:d} units'.
                         format(i + 1, self.actual_units[i]))

        if compute_WIP:
            logging.info("WIP Inventory {}".format(self.WIP_inventory))
            self.final_num_units.append(self.actual_units[-1])
            self.output_inventory[self.current_time_step] = self.final_num_units[-1] - \
                                                            self.demands[self.current_time_step]


            if self.current_time_step > 0:
                self.output_inventory[self.current_time_step] += \
                    self.output_inventory[self.current_time_step-1]



    def Solve(self):
        """
        Function to solve for the optimum assignment using the
        "Cost-scaling push-relabel algorithm"

        Reference -
        [1] https://developers.google.com/optimization/reference/graph/min_cost_flow/
        """

        # Check if the worker and station objects have been added
        assert len(self.workers) == self.num_workers, "Number of given workers" \
                                                      "less than num_workers"

        assert len(self.stations) == self.num_stations, "Number of given stations" \
                                                      "less than num_stations"

        self.current_time_step = 0
        # Display the skills for each worker
        logging.info("Initial Skill Levels:--------------------------------------------------")
        for worker_id in range(self.num_workers):
            logging.info("worker ID: {} Skill Levels: {}".format(worker_id + 1,
                                                                 self.workers[worker_id].rem_skill))

        for _ in trange(self.num_time_steps, desc="Time Step"):
            start = time()
            self._build_graph(cost_fn= self._compute_lost_sales_cost)
            build_time = time() - start
            logging.info("================== Solving for Time Step {} ==========================" .format(
                        self.current_time_step + 1))
            logging.info("Graph Info:-----------------------------------------------------------")
            logging.info("Worker IDs : {}".format(self.worker_id))
            logging.info("Station IDs : {}".format(self.station_id))
            logging.info("Start Nodes: {}".format(self.start_nodes))
            logging.info("End Nodes: {}".format(self.end_nodes))
            logging.info("Capacities: {}".format(self.capacities))
            logging.info("Costs: {}".format(self.costs))
            logging.info("Supplies: {}".format(self.supplies))
            logging.info("------------------- Minimizing Lost Sales ----------------------------")

            self.mean_units = 0
            # Reset for each iteration
            self.theoretical_units_produced = [0] * self.num_stations

            start = time()

            self.run_optimizer(log_history=False) # DO NOT log history

            solve_time = time() - start
            logging.info("---------------------------------------------------------------------")
            self._compute_units_produced(compute_WIP=False)

            del self.solver
            self.solver = pywrapgraph.SimpleMinCostFlow()

            logging.info("------------------- Minimizing Standard Deviation --------------------")

            start = time()
            self._build_graph(cost_fn= self._compute_std_cost)
            build_time += time() - start

            start = time()
            self.run_optimizer(log_history=True) # Log final assignment
            solve_time += time() - start
            self.experiment_time += build_time + solve_time

            logging.info("---------------------------------------------------------------------")

            self._compute_units_produced(compute_WIP=True)
            self.inventory_log = np.append(self.inventory_log, [self.WIP_inventory], axis=0)
            logging.info("---------------------------------------------------------------------")
            logging.info('Graph building time: {:.5f}s'.format(build_time))
            logging.info('Solving time: {:.5f}s'.format(solve_time))
            logging.info('Total time: {:.5f}s'.format(build_time + solve_time))
            logging.info("======================================================================")
            logging.info('\n')


            self.current_time_step += 1

            # Update the skills for each worker
            logging.info("Updated Skill Levels:--------------------------------------------------")
            for worker_id in range(self.num_workers):
                for station_id in range(self.num_stations):
                    skill_level = self.update_skills(worker_id, station_id)
                    self.workers[worker_id].rem_skill[station_id] = skill_level
                logging.info("worker ID: {} Skill Levels: {}".format(worker_id+1,
                                                             self.workers[worker_id].rem_skill))

            del self.solver
            self.solver = pywrapgraph.SimpleMinCostFlow()
        logging.info("======================================================================")
        logging.info('Total experiment time: {:.5f}s'.format(self.experiment_time))

        # Save output to CSV
        for i, worker in enumerate(self.workers):
            self.pd_file = self.pd_file.append(pd.Series(worker.assign_history[1:],
                                                         index=self.pd_file.columns,
                                                         name='Worker {}'.format(i+1)))
        self.pd_file.to_csv(self.SAVE_PATH + "Assignment Result.csv")

    def add_worker(self, lr: float,
                   fr: float,
                   initial_skill: List[int],
                   station_id: int):
        """
        Helper function to add a worker object to the problem setup
        :param lr: Learning rate of the worker
        :param fr: Forgetting rate of the worker
        :param initial_skill: Initial skill levels for each station
        :param station_id: Initial station assignment
        :return:
        """
        id = len(self.workers)
        assert len(initial_skill) == self.num_stations, "len(initial_skill)" \
                                                        "must be equal to the number of" \
                                                        "stations"
        """
        NOTE: station_id for assign_history is only for display purposes.
        Therefore, we use 1-based indexing.
        """
        self.workers.append(Worker(id, lr, fr, initial_skill, station_id))

    def add_station(self, S_max: float,
                    S_min: float,
                    delta: float,
                    eps: float):
        """
        Helper function to add Station object to the problem setup
        :param S_max (float): Max theoretical skill level attainable
        :param S_min (float): Min theoretical skill level attainable
        :param delta (float): Practical max skill threshold
        :param eps (float): Practical min skill threshold
        """
        id = len(self.stations)
        self.stations.append(Station(id, S_max, S_min, delta, eps))

    def update_skills(self, worker_id: int, station_id: int):
        """
        Function to update the skill levels of a given
        worker and a station at the end of each time step.
        The skill is improved if the station id is the same
        as the given worker's station id; else the skill
        is deteriorated.

        :param worker_id: Worker ID
        :param station_id: Station ID
        """
        if self.workers[worker_id].assign_history[-1] == station_id + 1:
            S = self._skill_improvement(worker_id, station_id)
        else:
            S = self._skill_deterioration(worker_id, station_id)
        return S

    def _skill_improvement(self, worker_id, station_id) -> int:
        """
        The skill improvement formula for a given worker_id k and
        station_id j at time step l is given as follows -

            S_jkl = S_max_j - (S_max_j - S_rem_jk)e^(beta_k*l)

        :return: Improved skill level (int)
        """
        l = 1 if self.current_time_step > 0 else 0

        S_max = self.stations[station_id].S_max
        S_rem = self.workers[worker_id].rem_skill[station_id]
        beta = self.workers[worker_id].beta
        exp_term = np.exp(beta*l)

        S =  (1 - exp_term)*S_max + S_rem*exp_term #S_max -  (S_max - S_rem)*exp_term
        S = int(round(abs(S)))

        return min(S, self.stations[station_id].S_UB)

    def _skill_deterioration(self, worker_id, station_id) -> int:
        """
        The skill deterioration formula for a given worker_id k and
        station_id j at time step l is given as follows -

            S_rem_jkl = S_min_j + (S_jk - S_min_j)e^(gamma_k*l)

        :return: Deteriorated skill level (int)
        """
        l = 1 if self.current_time_step > 0 else 0

        S_min = self.stations[station_id].S_min
        S_curr = self.workers[worker_id].rem_skill[station_id]
        gamma = self.workers[worker_id].gamma
        exp_term = np.exp(gamma*l)

        S = (1 - exp_term)*S_min + S_curr*exp_term #S_min +  (S_curr - S_min)*exp_term
        S = int(round(abs(S)))
        return max(S, self.stations[station_id].S_LB)

    def _compute_lost_sales_cost(self, worker_id, station_id) -> int:
        S = self.workers[worker_id].rem_skill[station_id]

        # if worker_id > 0:
        #     th = max([x if x < S else 0 for x in self.workers[worker_id - 1].rem_skill])
        # else:
        #     th = 0

        current_loss =  max(0, self.demands[self.current_time_step] - S) + \
                        max(0, S - self.demands[self.current_time_step])

        return current_loss


    def _compute_std_cost(self, worker_id, station_id):
        self.mean_units = sum(self.actual_units)/len(self.actual_units)
        S = self.workers[worker_id].rem_skill[station_id]

        future_loss = 0
        demand_erraticity = abs(np.diff(self.demands)).sum() # Total variation Distance
        future_loss_weight = 3 #int(round(np.exp(-0.00405058 *demand_erraticity + 1.54692256)))
        # print(future_loss_weight, demand_erraticity)

        for i in range(self.current_time_step + 1, self.num_time_steps):
            future_loss += max(0, self.demands[i] - S) + \
                           future_loss_weight*max(0, S - self.demands[i])

        return int(round(abs((S - self.mean_units)))) + future_loss

    def get_plots(self, is_show  = False):
        """
        Function to display all required plots.
        :return:
        """

        """
        Plot 1: Worker Station Assignment
        """
        plt.figure()
        for i, worker in enumerate(self.workers):
            plt.plot(worker.assign_history, lw=2, ls='-', marker='o',
                     alpha=0.7, ms=10, label="Worker {}".format(i+1))
        plt.xticks(np.arange(1, self.num_time_steps + 1, 1))
        plt.yticks(np.arange(1, self.num_workers + 1, 1))
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5),
                   fontsize=15)
        plt.xlabel('Time Steps', fontsize=15)
        plt.ylabel('Station ID', fontsize=15)
        plt.title('Worker Station Assignment', fontsize=15)
        plt.tight_layout()
        plt.savefig(self.SAVE_PATH + "Worker Station Assignment.pdf", dpi=300)

        """
        Plot 2: Demand Vs Output
        """
        plt.figure()
        inds = np.arange(1, self.num_time_steps + 1)
        width = 0.35
        plt.bar(inds - width/2, self.demands,
                width, label="Demands")
        plt.bar(inds + width / 2, self.final_num_units,
                width, label="Output")

        plt.xticks(np.arange(1, self.num_time_steps + 1, 1))
        plt.legend(fontsize=15)
        plt.xlabel('Time Steps', fontsize=15)
        plt.ylabel('Units', fontsize=15)
        plt.title('Demand Vs Output', fontsize=15)
        plt.tight_layout()
        plt.savefig(self.SAVE_PATH + "Demand Vs Output.pdf", dpi=300)

        """
        Plot 3: Inventory Vs Time
        """
        plt.figure()
        for i in range(self.num_stations):
            plt.plot(np.arange(1, self.num_time_steps + 1, 1), self.inventory_log[:, i], lw=2, ls='-', marker='o',
                     alpha=0.7, ms=10, label="Station {}".format(i + 1))
        plt.xticks(np.arange(1, self.num_time_steps + 1, 1))
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5),
                   fontsize=15)
        plt.xlabel('Time Steps', fontsize=15)
        plt.ylabel('Inventory Units', fontsize=15)
        plt.title('WIP Inventory Vs Time', fontsize=15)
        plt.tight_layout()
        plt.savefig(self.SAVE_PATH + "WIP Inventory Vs Time.pdf", dpi=300)

        """
        Plot 4: Output Inventory Vs Time
        """
        plt.figure()
        inds = np.arange(1, self.num_time_steps + 1)
        width = 0.35
        plt.bar(inds, self.output_inventory,
                width, label="Output Inventory")

        plt.xticks(np.arange(1, self.num_time_steps + 1, 1))
        plt.legend(fontsize=15)
        plt.xlabel('Time Steps', fontsize=15)
        plt.ylabel('Units', fontsize=15)
        plt.title('Output Inventory Vs Time', fontsize=15)
        plt.tight_layout()
        plt.savefig(self.SAVE_PATH + "Output Inventory Vs Time.pdf", dpi=300)

        """
        Plot 5: Production Results Vs Time
        """
        plt.figure()
        inds = np.arange(self.num_time_steps)
        width = 0.35
        plt.bar(inds - width/2, self.final_num_units,
                width, label="Output")
        plt.bar(inds+width/2, self.output_inventory,
                width, label="Output Inventory")

        inds = np.arange(1, self.num_time_steps + 1)
        plt.plot(inds, self.demands, 'r-',lw=2, ms=5, marker='o', label='Demands')

        plt.xticks(np.arange(1, self.num_time_steps + 1, 1))
        plt.legend(fontsize=15)
        plt.xlabel('Time Steps', fontsize=15)
        plt.ylabel('Units', fontsize=15)
        plt.title('Production Results Vs Time', fontsize=15)
        plt.tight_layout()
        plt.savefig(self.SAVE_PATH + "Production Results Vs Time.pdf", dpi=300)

        if is_show:
            plt.show()


    # def _compute_lost_sales(self): #Z_1
    #     """
    #      Assume worker 1 in station 1
    #      worker 2 in station 2 and so on
    #      :return:
    #      """
    #     I_f = 0
    #     units_processed = []
    #
    #     for i, w in enumerate(self.workers):
    #         I_f += w.initial_skill[i]
    #         units_processed.append(w.initial_skill[i])
    #
    #     self.Q.append(units_processed)
    #     self.WIP_inventory.append(I_f)
    #     return max(0, self.demands[0] - I_f)

    #
    #
    # def _compute_bn(self):
    #
    #     bn_idx = np.argmin(self.Q[0])
    #     return sum([q - self.Q[0][bn_idx] for q in self.Q])


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
        # print(cfg)
        return cfg


def create_experiment(cfg):
    """
    Function to create the experiment from the given configuration
    :param cfg: Configuration as a nested dict (can be from Yaml or a JSON file)
    :return: None
    """
    setup_cfg = cfg['worker_station_problem']
    w = WorkerStationProblem(num_workers=setup_cfg['Number_of_Workers'],
                             num_stations=setup_cfg['Number_of_Stations'],
                             num_time_steps=setup_cfg['Number_of_Timesteps'],
                             demands=setup_cfg['Demands'],
                             experiment_name=setup_cfg['Experiment_Name'])

    for i in range(setup_cfg['Number_of_Workers']):
        worker_cfg = cfg['Worker_{}_info'.format(i+1)]
        w.add_worker(lr=worker_cfg['Learning_rate'],
                     fr=worker_cfg['Forgetting_rate'],
                     initial_skill=worker_cfg['Initial_skill'],
                     station_id=worker_cfg['Initial_station'])

    for i in range(setup_cfg['Number_of_Stations']):
        station_cfg = cfg['Station_{}_info'.format(i+1)]
        w.add_station(S_max=station_cfg['Max_skill_level'],
                      S_min=station_cfg['Min_skill_level'],
                      delta=station_cfg['Delta'],
                      eps=station_cfg['Epsilon'])
    w.Solve()
    w.get_plots(is_show=False)

if __name__ == '__main__':
    # w = WorkerStationProblem(num_workers=4, num_stations=4, num_time_steps=3, demands=[12, 12, 5])
    cfg = parse_yaml('config.yaml')
    create_experiment(cfg)


