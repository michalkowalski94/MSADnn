import os
import numpy as np
#import pip
#from backports.functools_lru_cache import lru_cache
import matplotlib.pyplot as plt
from tqdm import tqdm
from ddsm_classes import ddsm_abnormality
from ddsm_util import *




main = "/mnt/e/CBIS-DDSM/figment.csee.usf.edu/figment.csee.usf.edu/pub/DDSM/cases/normals"
output_directory = "/mnt/e/CBIS-DDSM/DATASET_TIFF/Normals"
if os.path.exists(output_directory) == False:
        os.mkdir(output_directory)

for root, subdir, files in tqdm(os.walk(main)):
    ics_aslist = [x for x in files if ".ics" in x]
   # print ics_aslist
    if len(ics_aslist) > 0:
        ics_path = os.path.join(root, ics_aslist[0])
        ics_dict = get_ics_info(ics_path)
        for _file in files: #added progressbar form tqdm
            if ".LJPEG" in _file:
                if ".LJPEG.1" not in _file:
                #try:
                    #Path directories
                    ljpeg_dir = os.path.join(root, _file)
                    print ljpeg_dir
                    print "dupa1"
                    out_path = os.path.join(output_directory, _file)
                    print "dupa2"
                    out_path_im = out_path.split('.LJPEG')[0]
                    print "dupa3"
                    out_path_mask = os.path.join(out_path_im, os.path.basename(out_path_im) + "_MASK.tif")
                    print "dupa4"
                    ddsm = ddsm_abnormality(ljpeg_dir, "normal", None, ics_dict)
                    print "dupa5"
                    ddsm._decompress_ljpeg()
                    print "dupa6"
                    ddsm._read_raw_image()
                    print "dupa7"
                    mask = np.zeros(shape=ddsm._raw_image.shape)
                    ddsm._raw_image = None
                    print "dupa8"
                    if os.path.exists(out_path_im) == False:
                        os.mkdir(out_path_im)
                    ddsm.save_image(out_path_im, crop = False, od_correct = True, make_dtype = None, force = False)
                    print "dupa9"
                    plt.imsave(out_path_mask, mask, cmap='gray')
                    print("succeded")
                #except:
                    #print("Niepowodzenie z plikiem {}".format(ddsm.input_file_path))
print("Przekonwertowano wszystkie LJPEG do formatu TIFF wraz ze standaryzacja OD")
