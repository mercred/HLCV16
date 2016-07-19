import pandas as pd
from sklearn.cross_validation import train_test_split

FILE_PATH = '../data/driver_imgs_list.csv'
TRAIN_FILE = '../data/train.txt'
TEST_FILE = '../data/test.txt'
VALIDATION_FILE = '../data/validation.txt'

def split():
    data_frame = pd.read_csv(FILE_PATH, sep=',')
    # shuffle data frame
    data_frame = data_frame.sample(frac=1).reset_index(drop=True)
    # subject is the column name that corresponds to driver
    drivers = data_frame.subject.unique()
    train, test = train_test_split(drivers, test_size=0.2, random_state=42)
    train, val = train_test_split(train, test_size=0.2, random_state=43)

    df_train = data_frame.loc[data_frame['subject'].isin(train)]
    df_val = data_frame.loc[data_frame['subject'].isin(val)]
    df_test = data_frame.loc[data_frame['subject'].isin(test)]

    print df_train.subject.unique()
    print df_val.subject.unique()
    print df_test.subject.unique()
    df_train.to_csv(TRAIN_FILE, sep=',', header=False, index=False)
    df_val.to_csv(VALIDATION_FILE, sep=',', header=False, index=False)
    df_test.to_csv(TEST_FILE, sep=',', header=False, index=False)

split()

