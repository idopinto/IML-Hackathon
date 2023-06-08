from ydata_profiling import ProfileReport
from task_1.code.hackathon_code.Utils.pp import preprocess
from task_1.code.hackathon_code.Utils.utils import load_data




def make_report(filename, title):
    df = load_data(filename)
    X, y = preprocess(df)

    X.head()

    X['is_cancel'] = y

    ProfileReport(X, title=title).to_file(title + ".html")

make_report("../Datasets/train_set_agoda.csv", "agoda_data_profiling")