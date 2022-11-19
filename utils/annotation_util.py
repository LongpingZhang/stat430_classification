import glob
import pandas as pd
import numpy as np

def annotation_df(train_dir: str, val_dir: str, test_dir: str):
    '''
    The output dataframes has the columns (img_path, class).
    'Class' is either equal to 1=PNEUMONIA or 0=NORMAL.
    '''
    train_normal = glob.glob(train_dir + '/Normal/*')
    train_pneumonia = glob.glob(train_dir + '/Pneumo/*')

    val_normal = glob.glob(val_dir + '/Normal/*')
    val_pneumonia = glob.glob(val_dir + '/Pneumo/*')

    test_normal = glob.glob(test_dir + '/Normal/*')
    test_pneumonia = glob.glob(test_dir + '/Pneumo/*')

    df_train = concat_files(train_normal, train_pneumonia)
    df_val = concat_files(val_normal, val_pneumonia)
    df_test = concat_files(test_normal, test_pneumonia)
    
    return df_train, df_val, df_test

def concat_files(normal, pneumonia):
    files = [x for x in normal]
    files.extend([x for x in pneumonia])
    df = pd.DataFrame({'img_path': files,
                        'class': np.concatenate(([0] * len(normal),
                                                [1] * len(pneumonia)))})
    return df
