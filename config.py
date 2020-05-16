from utils import dice_coef
from tensorflow.keras.optimizers import Adam


def get_config():
    return {
        # Network architecture
        "net": {
            "epochs": 1000,
            "verbose": 1,
        },
        "net_cmp": {
            "optimizer": Adam(learning_rate=3e-4),
            "metrics": [dice_coef]
        },

        # For checkpoint saving, early stopping...
        "train": {
            "ckpt": {
                "ckpt_path": 'ckpt',
                "verbose": 1,
                "save_best_only": True
            },
            "early_stopping": {
                "patience": 50,
                "monitor": 'val_loss'
            }

        }
    }
