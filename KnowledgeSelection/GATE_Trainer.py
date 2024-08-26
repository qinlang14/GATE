import os
import cProfile
import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
import warnings

from KnowledgeSelection.ks_dataset import KSLoader
from KnowledgeSelection.GATEModel.model import KnowledgeSelectionModel


def gate_training(config):
    # Fix seed for PyTorch
    torch.manual_seed(42)
    # Fix seed for CUDA (if using GPU)
    torch.use_deterministic_algorithms(mode=True, warn_only=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    warnings.filterwarnings("ignore", message="Sparse CSR tensor support is in beta state.")
    warnings.filterwarnings("ignore", message="invalid value encountered in true_divide")

    opt_data = config["data"]
    opt_model = config["model"]
    opt_data["data_name"] = opt_model["data_name"] = config["data_name"]
    opt_data["batch_size"] = opt_model["batch_size"] = config["batch_size"]
    opt_data["rollouts"] = opt_model["rollouts"] = config["rollouts"]

    Loader = KSLoader(opt_data)
    loader_dict, loader_length = Loader.get_loader(opt_data['topic_split'], train=True)

    checkpoint_callback = ModelCheckpoint(
        monitor='Val_Seen top-1',
        filename='model-{epoch:02d}-{Val_Seen top-1:.4f}',
        save_top_k=5,
        mode='max',
        every_n_epochs=1
    )
    logger = CSVLogger(os.getcwd())

    samples = [opt_model["rollouts"] * loader_length["train"]] +\
              [loader_length["valid_seen"], loader_length["valid_unseen"], loader_length["test_seen"], loader_length["test_unseen"]]
    opt_model["samples"] = samples
    opt_model["device"] = device

    ks = KnowledgeSelectionModel(opt_model).to(device)
    # ks = torch.compile(ks)
    trainer = pl.Trainer(accelerator='gpu', max_epochs=opt_model["epochs"], callbacks=[checkpoint_callback], logger=logger,
                         precision=opt_model["precision"])

    trainer.fit(ks, train_dataloaders=loader_dict["train"],
                val_dataloaders=[loader_dict["valid_seen"], loader_dict["valid_unseen"], loader_dict["test_seen"], loader_dict["test_unseen"]])