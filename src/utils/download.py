from pathlib import Path
import os
import pandas as pd
import kagglehub



def download_data_and_create_annotations(path_to_save_annotations=None, train_file_name=None, test_file_name=None):
    ds_path = kagglehub.dataset_download("alsaniipe/chest-x-ray-image")
    df = dict()
    file_names = []
    split_labels = []
    for dir in Path(os.path.join(ds_path,'Data/')).iterdir():
        split_label = dir.name
        for sub_dir in Path(dir).iterdir():
           for _file in Path(sub_dir).iterdir():
                file_names.append(str(_file.absolute()))
                split_labels.append(split_label)
                
    
    df = pd.DataFrame(data={'file_name': file_names, 'split_label': split_labels, 'label_name': [file_name.split('/')[-1].split('(')[0] for file_name in file_names]})
    df['label'] = 0
    #df.loc[df['label_name'] == 'NORMAL', 'label'] = 0
    df.loc[df['label_name'] == 'PNEUMONIA', 'label'] = 1
    df.loc[df['label_name'] == 'COVID19', 'label'] = 2
    
    df.loc[df['split_label'] == 'train', ['file_name', 'label']].copy().to_csv(os.path.join(path_to_save_annotations, f'{train_file_name}.csv'), index=False)
    df.loc[df['split_label'] == 'test', ['file_name', 'label']].copy().to_csv(os.path.join(path_to_save_annotations,f'{test_file_name}.csv'), index=False)

    return ds_path