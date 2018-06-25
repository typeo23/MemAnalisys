import os
import pandas as pd
from AnalisysFunctions import load_and_filter, get_se, fit_spectra, mod_from_dir

#filedir_se = '/home/itay/projects/Q2specCG/gmx1k/SEa16/'
root = '/home/itay/projects/Q2specCG/'

mods_list = []
dirs = os.listdir(root)

for curr_dir in dirs:
    print(curr_dir)
    mods = mod_from_dir(root+curr_dir+'/', curr_dir)
    mods['Name'] = curr_dir
    mods_list.append(mods)


df = pd.DataFrame(mods_list)
col_order = ['Name', 'k_c', 'k_c_se', 'c_par', 'c_par__se',
             'k_t', 'k_t_se', 'k_tw', 'k_tw_se', 'c_per', 'c_per__se']

df = df[col_order]
df.to_csv('resSmall.csv')
print(df)
