import omnixas
from omnixas.model.metrics import ModelMetrics
from omnixas.data.xas import ElementSpectrum, IntensityValues, Material, EnergyGrid
from omnixas.data.scaler import ScaledMlSplit,ThousandScaler
from omnixas.core.periodic_table import Element, SpectrumType
from omnixas.featurizer.m3gnet_featurizer import M3GNetSiteFeaturizer, M3GNetFeaturizer
from omnixas.model.xasblock_regressor import XASBlockRegressor
from omnixas.data.ml_data import MLData, MLSplits
from omnixas.data.material_split import MaterialSplitter

import os
import optuna
import numpy as np
import matplotlib.pyplot as plt
from pymatgen.core.structure import Structure


import os, random, numpy as np, torch, pytorch_lightning as pl

from matgl.utils.io import load_model as matgl_load
from pathlib import Path

### some important changes from the tutorial for loading the model: 

m3g_path = Path("/pscratch/sd/h/henry/OmniXAS/OmniXAS_/models/M3GNet-MP-2021.2.8-PES")
pot = matgl_load(m3g_path)     
m3g_model = pot.model          

site_featurizer = M3GNetSiteFeaturizer(model=m3g_model)

###



elemental_file = "CU_FEFF.npz"
elemental_data = np.load(elemental_file)

FEFF_IDs = elemental_data['ids']
FEFF_sites = elemental_data['sites']
FEFF_energies = elemental_data['energies']
FEFF_spectra = elemental_data['spectras']

Cu_mpids = os.listdir("FEFF/Cu/")
published_structures = {}

for mpid in Cu_mpids:
    if mpid[0] != 'm':
        continue
    published_structure = Structure.from_file("FEFF/Cu/"+mpid+"/POSCAR")
    published_structures[mpid] = published_structure



structs = published_structures
features = {}
spectra = {}

for i in range(len(FEFF_IDs)):

    ID = FEFF_sites[i]
    mat = FEFF_IDs[i]
    
    if mat not in Cu_mpids:
        continue

    if mat not in features.keys():
        features[mat] = {}

    if mat not in spectra.keys():
        spectra[mat] = {}
    
    struct = structs[mat]

    spectrum = ElementSpectrum(
        element=Element.Cu,
        type=SpectrumType.FEFF,
        index=ID,
        material= Material( id = mat, structure = struct),
        intensities=IntensityValues(FEFF_spectra[i]*1000),
        energies=EnergyGrid(root=FEFF_energies),
    )

    # featurize material structure corresponding to the spectrum_1
    feature = site_featurizer.featurize(spectrum.material.structure, spectrum.index)


    features[mat][str(ID)] = feature*1000
    spectra[mat][str(ID)] = spectrum.intensities

idSite = [(FEFF_IDs[i], FEFF_sites[i]) for i in range(len(FEFF_IDs))]
split_idSite = MaterialSplitter.split(
    idSite=idSite,
    target_fractions=[0.7,0.2,0.1],
    seed=762345
)


train_idSite, val_idSite, test_idSite = split_idSite

train_data = MLData(
    X=[features[id][site] for id, site in train_idSite],
    y=[spectra[id][site] for id, site in train_idSite],
)
val_data = MLData(
    X=[features[id][site] for id, site in val_idSite],
    y=[spectra[id][site] for id, site in val_idSite],
)
test_data = MLData(
    X=[features[id][site] for id, site in test_idSite],
    y=[spectra[id][site] for id, site in test_idSite],
)

split = MLSplits(train=train_data, val=val_data, test=test_data)



#hyperparamater tuning: 

print("reached optuna")


def objective(trial):
    import numpy as np
    import gc
    try:
        # Hyperparams
        num_layers = trial.suggest_int("num_layers", 1, 6)  # start modest
        widths_choices = list(range(64, 800, 32))           # narrower range at first
        layer_widths = [int(trial.suggest_categorical(f"layer_{i}_width", widths_choices))
                        for i in range(num_layers)]

        batch_size = int(trial.suggest_categorical("batch_size", [16, 32, 64, 128]))
        lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)  # v3-friendly

        # Sanity checks on data shapes
        assert split.train.X.shape[1] == 64, f"Expected input_dim=64, got {split.train.X.shape[1]}"
        assert split.train.y.shape[1] == 141, f"Expected output_dim=141, got {split.train.y.shape[1]}"

        model = XASBlockRegressor(
            directory="checkpoints",
            max_epochs=150,
            early_stopping_patience=25,
            overwrite_save_dir=True,
            input_dim=64,
            output_dim=141,
            hidden_dims=[int(w) for w in layer_widths],
            initial_lr=lr,
            batch_size=batch_size,
        )

        model.fit(split)

        # Inference
        model.eval() if hasattr(model, "eval") else None
        preds = model.predict(split.val.X)
        targets = split.val.y

        # Ensure numpy and finite
        preds = np.asarray(preds)
        targets = np.asarray(targets)
        if preds.shape != targets.shape:
            raise ValueError(f"Pred/target shape mismatch: {preds.shape} vs {targets.shape}")
        if not np.all(np.isfinite(preds)):
            raise ValueError("Non-finite values in predictions.")

        metrics = ModelMetrics(predictions=preds, targets=targets)
        val = float(np.asarray(metrics.mse).mean())
        if not np.isfinite(val):
            raise ValueError(f"Objective is non-finite: {val}")

        return val

    except Exception as e:
        # Log full traceback so you know why the trial failed
        import traceback
        print(f"Trial {trial.number} failed:")
        traceback.print_exc()
        # Optionally store the error message with the trial
        try:
            trial.set_user_attr("error", str(e))
        except Exception:
            pass
        return np.inf

    finally:
        # Free memory between trials (important if using GPU)
        try:
            import torch
            del model
            torch.cuda.empty_cache()
        except Exception:
            pass
        gc.collect()


N_seconds = 60 * 30

study = optuna.create_study(direction='minimize')
study.optimize(objective, timeout=N_seconds)

print('Best hyperparameters: ', study.best_params)
print('Best performance: ', study.best_value)
