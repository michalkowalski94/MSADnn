import numpy as np
annotation_path = "D:\\CBIS-DDSM\\master\\processed_data_set\\cbis_ddsm_annotation.txt"
val_split = .1
test_split = .5
with open(annotation_path) as f:
    lines = f.readlines()
np.random.seed(2137)
np.random.shuffle(lines)
np.random.seed(2137)
np.random.shuffle(lines)
np.random.seed(None)
num_val = int(len(lines)*val_split)
num_train = len(lines) - num_val
num_test = int(num_val * test_split)

with open("D:\\CBIS-DDSM\\master\\processed_data_set\\train_annotation.txt", "w") as f:
    for line in lines[:num_train]:
        f.write(line)
with open("D:\\CBIS-DDSM\\master\\processed_data_set\\val_annotation.txt", "w") as f:
    for line in lines[num_train:num_train + num_test]:
        f.write(line)
        
with open("D:\\CBIS-DDSM\\master\\processed_data_set\\test_annotation.txt", "w") as f:
    for line in lines[num_train + num_test:]:
        f.write(line)
        
