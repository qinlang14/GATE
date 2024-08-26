from KnowledgeSelection import GATE_Trainer
from utils import read_json


if __name__ == '__main__':
    data_name = "WoW"
    config = read_json(f"KnowledgeSelection/Configs/{data_name}_config.json")
    GATE_Trainer.gate_training(config)