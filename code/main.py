import numpy as np
import tensorflow as tf
import os.path
import params
import production
import ntn_training


def main():
    if os.path.exists(params.CKPT_DIR):
        production.run_production(evaluation=False)
    else:
        ntn_training.run_training()
    pass


if __name__ == '__main__':
    main()
