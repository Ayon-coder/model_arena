from processing.processing import collect_data_info
from trainer_evaluator import model_train
from pprint import pprint

def main():

    dataset_path = input("Enter your dataset path: ")
    target_col = input("Enter your target column: ")

    config = {
        "dataset_path": dataset_path,
        "target_col": target_col
    }

    data = collect_data_info(config)
    verdict = model_train(data)
    pprint(verdict)

if __name__ == "__main__":
    main()