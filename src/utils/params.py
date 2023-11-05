import yaml

class Params:
    """
        Reads content from the yaml file referenced in the config path argument, which exposes the variables in a nested structure.
        For instance, if the yaml file includes...

        meta:
            somevar: 23
            some_other: 
                another_one: 25

        it can be accessed through params.meta['somevar'] and params.meta['some_other']['another_one'] respectively
    """

    def __init__(self, config_path):
        with open(config_path, 'r') as file:
            self.model_config = yaml.safe_load(file)
            for section in self.model_config:
                parameters = self.model_config[section]
                setattr(self, section, parameters)

    def save(self):
        # overwrite this function with code from your experiment logger of choice (e.g. mlflow, weights and biases, kubeflow, etc.)
        pass