class Config(object):
    """Object to hold the config requirements for an agent/game"""

    def __init__(self):
        self.seed = None
        self.environment = None
        self.requirements_to_solve_game = None
        self.num_episodes_to_run = None
        self.file_to_save_data_results = None
        self.file_to_save_results_graph = None
        self.runs_per_agent = None
        self.visualise_overall_results = None
        self.visualise_individual_results = None
        self.hyperparameters = None
        self.use_GPU = False
        self.overwrite_existing_results_file = None
        self.save_model = False
        self.standard_deviation_results = 1.0
        self.randomise_random_seed = True
        self.show_solution_score = False
        self.debug_mode = False
        self.evaluation = False
        self.learnt_network = False
        self.average_score_required_to_win = 0
        self.number_chargers = None
        self.maximum_power = None
        self.number_power_options = None
        self.evaluation_after_training = False
        self.batch_size = 256
        self.updates_per_step = 1
        self.do_evaluation_iterations = False
