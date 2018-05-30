import os.path
import production
import training
import evaluation
import params

MODE = 1


def main():
    if os.path.exists(params.CKPT_DIR):
        production.run_production(evaluation=False)
    else:
        if MODE == 0:
            print('\n=====Training Mode=====\n')
            training.run_training()
        if MODE == 1:
            print('\n=====Evaluation Mode=====\n')
            evaluation.run_evaluation()
    pass


if __name__ == '__main__':
    main()
