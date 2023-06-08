from hackathon_code.Base import baseline
import pandas as pd
from task_1.code.hackathon_code.Utils import utils
from sklearn.metrics import precision_score, recall_score
if __name__ == '__main__':
    df = utils.load_data("hackathon_code/Datasets/train_set_agoda.csv")
    f1, precision, recall = utils.pipeline(df)
    print("Results:\n"
          f"\t Precision: {precision}\n"
          f"\t Recall: {recall}\n"
          f"\t F1: {f1}\n")
