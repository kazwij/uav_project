#importing libraries 
import pandas as pd
import numpy as np
import missingno as msno
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
#data loading on experiment

df0 =  pd.read_csv("../data/experiment_vol1.csv")
df1 =  pd.read_csv("../data/experiment_vol2.csv")
df2 =  pd.read_csv("../data/experiment_vol3.csv")

#data on geom
df3 =  pd.read_csv("../data/geom_vol1.csv")
df4 =  pd.read_csv("../data/geom_vol2.csv")
df5 =  pd.read_csv("../data/geom_vol3.csv")

def concat_files(df_list):
    '''
    this function will take list of dataframes and concat them 
    printout each data frames columns count and final concat df column count 
    '''
    try:
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
        print(f'concatination successfully completed  ')
        return new_df
    
    except Exception as e:
        print(f'below  error occured when executing the funtion {e}')

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

print('Data frame exp_df and geo_df were created by merging relavent data bases. Columns renamed as per naming convention')

'''
EDA - Exploratary Data Analysis on exp_df and geo_df.   
'''
#exp_df and geo_df merge to single df as master_df.
#using "outer" method to merge both to avoid any data lost 
master_df = pd.merge(exp_df,geo_df,on='blades_name',how='outer')
master_df.head()
master_df.info()
print(f'A dataset called master_df suceesfully created by merging all datasets. ')
print(f'it contain {master_df.shape[0]} raws and {master_df.shape[1]}  columns ')
# dataset contain NaN values.

# checking for missing values
print('missing values in master_df') 
print(master_df.isna().sum())

# check for duplicated values considering all the columns

df_duplicated = master_df[master_df.duplicated()]

print(f'dataset after merging contains {df_duplicated.shape[0] } raws of duplicates. ')

# dropping duplicated values
master_df = master_df.drop_duplicates()

print(f'after dropping duplicates dataset  has {master_df.shape[0] } raws and {master_df.shape[1] }. ')

print('master data set contain duplicated values. It is expected as several experiments can be done on single type blade  ')

# calculate the chord distrubution and radius distribution 
master_df['radius'] = master_df['propellers_diameter_x']/2
master_df.columns
master_df['chord_distribution'] = master_df['radius']*master_df['adimensional_chord_c/R']
master_df['radius_distribution'] = master_df['radius']*master_df['adimensional_radius_r/R']

print('calculated the chord distrubution and radius distribution and created seperate columns for each in master_df')

# calculating the area of the propeller

step_size = 0.01 # step size define as 0.01 for better accuracy of the area. 
R = master_df['radius'].to_numpy()
C = master_df['adimensional_chord_c/R'].to_numpy() 
Area = []
for i,c in zip(R,C):
    x = np.arange(0,i,step_size)
    y=c*x
    area  = np.trapz(y , dx = step_size)
    Area.append(area)
#print(Area)

master_df['blade_area'] = pd.Series(Area)
#master_df.head()
print('calculated the blade area and new column addes as blade_area to master_df ')
# total area of the blades 
master_df['total_area_of_blades'] = master_df['number_of_blades']*master_df['blade_area']

print('calculated the total blade area and new column addes as total_blade_area to master_df ')

#Disk area calculation
master_df['disk_area'] = np.pi*master_df['radius']**2
master_df.head()

print('calculated the Disk area and create a column called disk_area in the master_df')
# Calculation of Solidity_value

master_df['solidity_value'] = master_df['blade_area']/master_df['disk_area']
print('calculated the solidity value and create a column called solidity_value in the master_df')

print('updated master_df  ')
master_df.head()

#checking for missing values in solidity_value 
missing_value_count = master_df['solidity_value'].isna().sum()

missing_value =  round((missing_value_count/master_df.shape[0])*100,2)
print(f'solidity values has {missing_value} % of missing values ')



# blades that does not have a solidity value

blades_missing_solidity_value = master_df[master_df['solidity_value'].isna()][['blades_name','solidity_value']]
blades_missing_solidity_value
print(f"out of {len(master_df['blades_name'].unique())} unique blades types" )
print(f'{len(blades_missing_solidity_value['blades_name'].unique())} blades does not have solidity values ')

#visual representation of missing values in the dataset.
msno.bar(master_df)


#Visualization
type(master_df)
master_df.plot(kind='scatter', x='number_of_blades', y = 'thrust_coefficient_output')
master_df.columns

fig1 = master_df.plot(kind='scatter', x='power_coefficient_output', y = 'thrust_coefficient_output',title='power coefficiant vs trust coefficient')
sns.scatterplot(x='power_coefficient_output' ,y='thrust_coefficient_output',data=master_df)
master_df.plot(kind='scatter', x='power_coefficient_output', y = 'thrust_coefficient_output')

print(f'since the solidity values consider to be very important parameter in UAV preformance below pair value comparisons done ')
sns.pairplot(data=master_df,vars=['disk_area','solidity_value','beta_angle_relative_to_rotation'])
plt.show()

print(f'solidity value reduce with disk area. but no linear relationship shown ')
print(f'higher the beta angle the solidity value tend to reduce')


sns.pairplot(data=master_df,vars=['rpm_rotation_input','efficiency_output','solidity_value',])
plt.show()
master_df.dtypes
master_df.columns
df_correlation  =master_df[[
    'number_of_blades', 'propellers_diameter_x', 'propellers_pitch_x',
       'advanced_ratio_input', 'rpm_rotation_input',
       'thrust_coefficient_output', 'power_coefficient_output',
       'efficiency_output', 'propellers_diameter_y',
       'propellers_pitch_y', 'adimensional_chord_c/R',
       'adimensional_radius_r/R', 'beta_angle_relative_to_rotation', 'radius',
       'chord_distribution', 'radius_distribution', 'blade_area',
       'total_area_of_blades', 'disk_area', 'solidity_value']].dropna().corr()

df_correlation

master_df.dtypes
sns.heatmap(df_correlation)

# dropping columns that are not required for model development
master_df = master_df.drop(columns=[
'propellers_name', 'blades_name', 'propellers_brand_x',
 'propellers_brand_y'])
master_df.columns

#missing values treating.
def missingvalueprecentage(col):
    '''
    This function will provide  missing values precentage of each columns
    
    '''    
    missing_value_count = master_df[col].isna().sum()
    missing_value =  round((missing_value_count/master_df.shape[0])*100,2)
    print(f'{col}  has {missing_value} %  of missing values ')
for col in master_df.columns:
    missingvalueprecentage(col)

print("sinnce missing values % is less than 10 %  of total value, dropping the missing values.")

# dropping missing values and create 3 df for models 
# model_1 - without missing values
#model_2 - with missing values
#model_3 - without solidity value
 
model_1= master_df.dropna()
model_2 = master_df
model_3 = master_df.drop(columns=['solidityP_value'])

########copy 
# Identify features and target variables
features = master_df.drop(columns=['thrust_coefficient_output', 'power_coefficient_output', 'efficiency_output'])
target_thrust = master_df['thrust_coefficient_output']
target_power = master_df['power_coefficient_output']
target_efficiency = master_df['efficiency_output']


# Split into training and testing sets
X_train, X_test, y_train_thrust, y_test_thrust = train_test_split(features, target_thrust, test_size=0.2, random_state=42)
X_train, X_test, y_train_power, y_test_power = train_test_split(features, target_power, test_size=0.2, random_state=42)
X_train, X_test, y_train_efficiency, y_test_efficiency = train_test_split(features, target_efficiency, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Model without missing value imputation
model_without_imputation = xgb.XGBRegressor(objective='reg:squarederror')
model_without_imputation.fit(X_train_scaled, y_train_thrust)

# Predict and evaluate
preds_without_imputation = model_without_imputation.predict(X_test_scaled)
mse_without_imputation = mean_squared_error(y_test_thrust, preds_without_imputation)
r2_without_imputation = r2_score(y_test_thrust, preds_without_imputation)

# Model with missing value imputation
imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)
X_train_imputed_scaled = scaler.fit_transform(X_train_imputed)
X_test_imputed_scaled = scaler.transform(X_test_imputed)

model_with_imputation = xgb.XGBRegressor(objective='reg:squarederror')
model_with_imputation.fit(X_train_imputed_scaled, y_train_thrust)

# Predict and evaluate
preds_with_imputation = model_with_imputation.predict(X_test_imputed_scaled)
mse_with_imputation = mean_squared_error(y_test_thrust, preds_with_imputation)
r2_with_imputation = r2_score(y_test_thrust, preds_with_imputation)

# Model without solidity value
X_train_no_solidity = X_train.drop(columns=['solidity_value'])
X_test_no_solidity = X_test.drop(columns=['solidity_value'])
X_train_no_solidity_scaled = scaler.fit_transform(X_train_no_solidity)
X_test_no_solidity_scaled = scaler.transform(X_test_no_solidity)

model_without_solidity = xgb.XGBRegressor(objective='reg:squarederror')
model_without_solidity.fit(X_train_no_solidity_scaled, y_train_thrust)

# Predict and evaluate
preds_without_solidity = model_without_solidity.predict(X_test_no_solidity_scaled)
mse_without_solidity = mean_squared_error(y_test_thrust, preds_without_solidity)
r2_without_solidity = r2_score(y_test_thrust, preds_without_solidity)

print(f"Model without imputation: MSE = {mse_without_imputation}, R² = {r2_without_imputation}")
print(f"Model with imputation: MSE = {mse_with_imputation}, R² = {r2_with_imputation}")
print(f"Model without solidity: MSE = {mse_without_solidity}, R² = {r2_without_solidity}")






### end