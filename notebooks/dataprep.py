import pandas as pd
import numpy as np

#data loading on experiment

df0 =  pd.read_csv("../data/experiment_vol1.csv")
df1 =  pd.read_csv("../data/experiment_vol2.csv")
df2 =  pd.read_csv("../data/experiment_vol3.csv")

#data on geom
df3 =  pd.read_csv("../data/geom_vol1.csv")
df4 =  pd.read_csv("../data/geom_vol2.csv")
df5 =  pd.read_csv("../data/geom_vol3.csv")



#Display Data

df1.head()
df2.head()
df3.head()

df1.info()
df2.info()
df3.info()

# check further for any missing valus 

df1.isna().sum()
df2.isna().sum()
df3.isna().sum()


def concat_files(df_list):
    '''
    this function will take list of dataframes and concat them 
    printout each data frames columns count and final concat df column count 
    '''
    a = len(df_list)
    i=0
    raw_total = []
    while i<a:
        raw_size =df_list[i].shape[0] 
        print(f'dataframe {i} raw size {raw_size}')
        i = i+1
        raw_total.append(raw_size) 
 
    new_df = pd.concat(df_list)
    print(f'total concat raws are {sum(raw_total)}')
    return new_df
#concat the files and rename the columns to python convension
exp_df = concat_files([df0,df1,df2]).rename(
   columns = {"Propeller's Name":"propellers_name", "Blade's Name":"blades_name", "Propeller's Brand":"propellers_brand",
       "Number of Blades":"number_of_blades", "Propeller's Diameter":"propellers_diameter", "Propeller's Pitch":"propellers_pitch",
       "Advanced Ratio Input":"advanced_ratio_input", "RPM Rotation Input":"rpm_rotation_input",
       "Thrust Coefficient Output":"thrust_coefficient_output", "Power Coefficient Output":"power_coefficient_output",
       "Efficiency Output":"efficiency_output"} )
geo_df = concat_files([df3,df4,df5]).rename(
    columns ={"Blade's Name":"blades_name", "Propeller's Brand":"propellers_brand", "Propeller's Diameter":"propellers_diameter",
       "Propeller's Pitch":"propellers_pitch", "Adimensional Chord - c/R":"adimensional_chord_c/R",
       "Adimensional Radius - r/R":"adimensional_radius_r/R", "beta - Angle Relative to Rotation":"beta_angle_relative_to_rotation"}) 

geo_df.head()
len(geo_df["propellers_diameter"].unique())
len(exp_df["propellers_diameter"].unique())

geo_df['radius'] = geo_df['propellers_diameter']/2

geo_df['chord_distribution'] = geo_df['radius']*geo_df['adimensional_chord_c/R']
geo_df['radius_distribution'] = geo_df['radius']*geo_df['adimensional_radius_r/R']

# calculating the area of the propeller

step_size = 0.01 # step size define as 0.01 for better accuracy of the area. 
R = geo_df['radius'].to_numpy()
C = geo_df['adimensional_chord_c/R'].to_numpy() 
Area = []
for i,c in zip(R,C):
    x = np.arange(0,i,step_size)
    y=c*x
    area  = np.trapz(y , dx = step_size)
    Area.append(area)
#print(Area)

geo_df['blade_area'] = pd.Series(Area)
geo_df.head()
exp_df.head()

len(geo_df['blades_name'].unique().sum())
len(exp_df['blades_name'].unique().sum())


# merging to df to a single df based on "blades_name" column

main_df = pd.merge(exp_df,geo_df,on='blades_name',how="inner")
main_df.head()
main_df.columns

main_df['total_area_of_blades'] = main_df['number_of_blades']*main_df['blade_area']

#Disk area calculation
main_df['Disk_area'] = np.pi*main_df['radius']**2
main_df.head()

# Calculation of Solidity_value

main_df['solidity_value'] = main_df['blade_area']/main_df['Disk_area']

# get the unique count of blades types in each df
geo_df_uniqe_blade_count = len(geo_df['blades_name'].unique().sum())
exp_df_uniqe_blade_count =  len(exp_df['blades_name'].unique().sum())
blade_count_diff = exp_df_uniqe_blade_count - geo_df_uniqe_blade_count


massage = (
f"geo_df contain {geo_df_uniqe_blade_count } unique blade types.\n" 
f"exp_df contain {exp_df_uniqe_blade_count} unique blade types.\n"   
f"exp_df contain extra {blade_count_diff} blades in the dataset that do not have geo data "

)
print(massage)
'''
 get names of the blades that do not have their geo_data available
main_df = pd.merge(exp_df,geo_df,on='blades_name',how="inner")


'''

'''
hi this is just a test

'''