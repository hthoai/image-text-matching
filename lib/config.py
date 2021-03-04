from typing import Any
import yaml
import torch
import lib.models as models
import lib.criterions as criterions
import lib.datasets as datasets


class Config:
    def __init__(self, config_path: str):
        self.config = {}
        self.config_str = ""
        self.load(config_path)

    def load(self, path: str) -> None:
        with open(path, "r") as file:
            self.config_str = file.read()
        self.config = yaml.load(self.config_str, Loader=yaml.FullLoader)

    def __repr__(self):
        return self.config_str

    def get_dataset(self, data_split: str) -> None:
        return getattr(datasets, self.config["datasets"][data_split]["type"])(
            **self.config["datasets"][data_split]["parameters"]
        )

    def get_model(self, vocab_size: int, **kwargs) -> Any:
        name = self.config["model"]["name"]
        parameters = self.config["model"]["parameters"]
        parameters["vocab_size"] = vocab_size
        return getattr(models, name)(**parameters, **kwargs)

    def get_optimizer(self, model_parameters: Any) -> Any:
        return getattr(torch.optim, self.config["optimizer"]["name"])(
            model_parameters, **self.config["optimizer"]["parameters"]
        )

    def get_criterion(self) -> Any:
        name = self.config["criterion"]["name"]
        parameters = self.config["criterion"]["parameters"]
        return getattr(criterions, name)(**parameters)

    def get_lr_update(self) -> None:
        return self.config["lr_update"]

    def get_vocab_path(self) -> str:
        return self.config["vocab"]

    def __getitem__(self, item):
        return self.config[item]

    def __contains__(self, item):
        return item in self.config
