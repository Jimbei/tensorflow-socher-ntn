import params
import evaluation
import dataprocessing


def main():
    # dataprocessing.generate_data(1)
    if params.MODE == 1:
        evaluation.run_evaluation()


if __name__ == '__main__':
    main()
