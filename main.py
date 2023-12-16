import hydra

import experiment
from experiment import ExpBase

@hydra.main(config_name="main", version_base=None, config_path="conf")
def main(config) -> None:
    exp: ExpBase = getattr(experiment, config.exp.name)(config)
    exp.run()

if __name__ == "__main__":
    main()



