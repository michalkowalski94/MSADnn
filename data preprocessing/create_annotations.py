#Below code is an example suggesting how to create annotation file for YOLO
#Execution of this code will create annotation file for abnormality detection only


import numpy as np
import pandas as pd
import os
from tqdm import tqdm


##id_list = []
##density_list = []
##side_list = []
##view_list = []
##abn_list = []
##pathology_list = []
##x_lo_list = []
##y_lo_list = []
##x_hi_list = []
##y_hi_list = []
##img_path_list = []
##mask_path_list = []
##df_dict = {}

def add_normals(metadata_path = '../processed_data_set/ddsm_description_cases.csv',
                normals_path = '../DATASET_TIFF/Normals',
                save_path = '../processed_data_set'):
    """This function adds normal cases to a ddsm description csv"""

    ddsm_df = pd.read_csv(metadata_path, index_col = False)
    for root, sub, f_list in os.walk(normals_path):
        df_dict = {}
        try:
            if 'MASK' not in f_list[0]:
                mammogram_path = os.path.join(root,f_list[0])
                if '_CC_' in f_list[0]:
                    view = 'CC'
                else:
                    view = 'MLO'
                if 'RIGHT' in f_list[0]:
                    side = 'RIGHT'
                else:
                    side = 'LEFT'
                patient_id = f_list[0].split('.')[0]
                density = int(patient_id[-1])
                x_lo = np.random.randint(100,600)
                y_lo = np.random.randint(100,600)
                x_hi = x_lo + np.random.randint(40,200)
                y_hi = y_lo + np.random.randint(40,200)
                df_dict['patient_id'] = patient_id
                df_dict['breast_density'] = density
                df_dict['side'] = side
                df_dict['view'] = view
                df_dict['ab_num'] = int(0)
                df_dict['pathology'] = 'NORMAL'
                df_dict['abnormality_type'] = 'normal'
                df_dict['x_lo'] = x_lo
                df_dict['y_lo'] = y_lo
                df_dict['x_hi'] = x_hi
                df_dict['y_hi'] = y_hi
                df_dict['od_img_path'] = mammogram_path
                mask_path = os.path.join(root,f_list[1])
                df_dict['mask_path'] = mask_path

            else:
                mammogram_path = os.path.join(root,f_list[1])
                if '_CC_' in f_list[1]:
                    view = 'CC'
                else:
                    view = 'MLO'
                if 'RIGHT' in f_list[1]:
                    side = 'RIGHT'
                else:
                    side = 'LEFT'
                patient_id = f_list[1].split('.')[0]
                density = int(patient_id[-1])
                x_lo = np.random.randint(100,600)
                y_lo = np.random.randint(100,600)
                x_hi = x_lo + np.random.randint(40,200)
                y_hi = y_lo + np.random.randint(40,200)
                df_dict['patient_id'] = patient_id
                df_dict['breast_density'] = density
                df_dict['side'] = side
                df_dict['view'] = view
                df_dict['abn_num'] = int(0)
                df_dict['pathology'] = 'NORMAL'
                df_dict['abnormality_type'] = 'normal'
                df_dict['x_lo'] = x_lo
                df_dict['y_lo'] = y_lo
                df_dict['x_hi'] = x_hi
                df_dict['y_hi'] = y_hi
                df_dict['od_img_path'] = mammogram_path
                mask_path = os.path.join(root,f_list[0])
                df_dict['mask_path'] = mask_path
            ddsm_df = ddsm_df.append(df_dict, ignore_index = True, sort = False)
        except:
            print('No files in {}'.format(root))
    ddsm_df.to_csv(os.path.join(save_path,'new_description.csv'), index=False)
    return ddsm_df

def create_classes_annotation(df, save_path = '../processed_data_set'):
    if type(df) == str:
        df = pd.read_csv(df)
    if os.path.exists(save_path) == False:
        os.mkdir(save_path)
    #To create annotation file for specific task, please establish the classes
    #dictionary
    classes = {'abnormality':0}
    with open(os.path.join(save_path,'ddsm_class.txt'), 'w') as fh:
        fh.write('abnormality')

    #This example is creating only abnormality annotations
    with open(os.path.join(save_path,'ddsm_annotation_ab.txt'), 'w') as fh:
        for idx in tqdm(df.index):
            mammogram_path = df.loc[idx,'od_img_path']
            ab_type = df.loc[idx,'abnormality_type']
            x_lo = df.loc[idx,'x_lo']
            y_lo = df.loc[idx,'y_lo']
            x_hi = df.loc[idx,'x_hi']
            y_hi = df.loc[idx,'y_hi']
            annotation = mammogram_path + ' '
            annotation += str(x_lo)+','+str(y_lo)+','+str(x_hi)+','+str(y_hi)+',0 '
            annotation = annotation[:-1] + '\n'
            fh.write(annotation)
    print("Classes and annotation files prepared and saved in {}".format(save_path))

if __name__ == '__main__':
##    ddsm_df = add_normals()
    create_classes_annotation("../processed_data_set/ddsm_description_cases.csv")
