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

df0.columns

df3.columns


# fuction to concat the files
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
master_df = pd.merge(exp_df,geo_df,how='left')
master_df.columns
master_df.shape
master_df.head()
master_df.info()
master_df.describe().T
print(f'A dataset called master_df suceesfully created by merging all datasets. ')
print(f'it contain {master_df.shape[0]} raws and {master_df.shape[1]}  columns ')


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
master_df['radius'] = master_df['propellers_diameter']/2
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
plt.title('Missing values in the data - bar chart represent ',fontsize = 20)
plt.show()
## EDA ##
#Visualization



fig, axes = plt.subplots(3, 3, figsize=(15, 12))
sns.histplot(data=master_df, x='thrust_coefficient_output', kde=True, color='blue', ax=axes[0, 0])
sns.histplot(data=master_df, x='power_coefficient_output', kde=True, color='yellow', ax=axes[0, 1])
sns.histplot(data=master_df, x='efficiency_output', kde=True, color='purple', ax=axes[0, 2])
sns.histplot(data=master_df, x='solidity_value', kde=True, color='teal', ax=axes[1, 0])
sns.histplot(data=master_df, x='disk_area', kde=True, color='orange', ax=axes[1, 1])
sns.histplot(data=master_df, x='chord_distribution', kde=True, color='red', ax=axes[1, 2])
sns.histplot(data=master_df, x='beta_angle_relative_to_rotation', kde=True, color='olive', ax=axes[2, 0])
sns.histplot(data=master_df, x='radius_distribution', kde=True, color='green', ax=axes[2, 1])
sns.histplot(data=master_df, x='total_area_of_blades', kde=True, color='brown', ax=axes[2, 2])
fig.suptitle('Visual Representation of the Data Distribution', fontsize=20)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])  
plt.show()


print('thrust coefficient values peak area noticeble around 0.1')
print('power coefficient concentrate around 0.04 and 0.06 value area and the peak seems to haveing around 0.05')
print('efficiency output has large number of non values and peak around 0.6')
print('solidity value peak is around 0.025')
print('disk area concetrate arounf 50 and 100')
print('chord distribution is peak at 0.8 ')
print('beta angle skewed to right and peak around 10-20')
print('radias distribution almost having equal distribution')
print('total area of blades uneven and more dense around 4')



#master_df['propellers_pitch_x'].value_counts()
def outlier_checker(df, column):
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    IQR = q3 - q1
    upper = q3 + 1.5 * IQR
    lower = q1 - 1.5 * IQR
    df = df[(df[column] >= lower) & (df[column] <= upper)]
    return df

columns_to_check = [ 'thrust_coefficient_output', 'power_coefficient_output', 'efficiency_output']
       

for col in columns_to_check:
    master_df = outlier_checker(master_df, col)

master_df['propellers_diameter'].value_counts()
master_df.shape


#bivariant analysis againts thrust coefficient
plt.figure(figsize=(5,4))
sns.jointplot(x='propellers_diameter',y='thrust_coefficient_output',data=master_df)
plt.show()
print('according to the diagram propellers diameter is a major factor for thrust coefficient, but need more comprehensive analysis with other  factors when modeling ')

plt.figure(figsize=(5,4))
sns.jointplot(x='advanced_ratio_input',y='thrust_coefficient_output',data=master_df)
plt.show()

print('this plot helps in understanding the inverse relationship between advanced_ratio_input and thrust_coefficient_output. ')
print(' The advanced ratio input is a crucial factor in predicting the thrust coefficient, with higher advanced ratios typically resulting in lower thrust coefficients. This insight can be valuable for modeling and optimizing propeller performance')

plt.figure(figsize=(5,4))
sns.jointplot(x='rpm_rotation_input',y='thrust_coefficient_output',data=master_df)
plt.show()
print("The plot reveals that thrust_coefficient_output generally increases with higher rpm_rotation_input values, with most data points concentrated between 0.075 and 0.15 for thrust_coefficient_output across a wide range of rpm_rotation_input")

plt.figure(figsize=(5,4))
sns.jointplot(x='total_area_of_blades',y='thrust_coefficient_output',kind='hist',data=master_df)
plt.show()
print('when considering the histrogram propellers blades total area between 3-5 tend to provide coefficient between 0-0.1 ')

#bivariant analysis againts power_coefficient_output

plt.figure(figsize=(5,4))
sns.jointplot(x='propellers_diameter',y='power_coefficient_output',data=master_df)
plt.show()
print('it seems higer the propellers diameter lower the power coefficient, more propellers have diamter range of 9-12.5 and gibing higher power coefficient')

plt.figure(figsize=(5,4))
sns.jointplot(x='advanced_ratio_input',y='power_coefficient_output',data=master_df)
plt.show()
print('power coefficient and advance ratio input having inverrse relationship')

plt.figure(figsize=(5,4))
sns.jointplot(x='rpm_rotation_input',y='power_coefficient_output',kind='hist',data=master_df)
plt.show()
print("dara seems more concentraded arond rpm 4000 , 5000,and 6000 giving power coeffcient range from 0 - 0.08")

plt.figure(figsize=(5,4))
sns.jointplot(x='total_area_of_blades',y='power_coefficient_output',kind='hist',data=master_df)
plt.show()
print('data concentration arounf 3-5 total blade area')

#bivariant analysis againts efficiency _output 
plt.figure(figsize=(5,4))
sns.jointplot(x='propellers_diameter',y='efficiency_output',data=master_df)
plt.show()
print('')

plt.figure(figsize=(5,4))
sns.jointplot(x='advanced_ratio_input',y='efficiency_output',data=master_df)
plt.show()
print('power coefficient and advance ratio input having inverrse relationship')

plt.figure(figsize=(5,4))
sns.jointplot(x='rpm_rotation_input',y='efficiency_output',kind='hist',data=master_df)
plt.show()
print("dara seems more concentraded arond rpm 4000 , 5000,and 6000 giving power coeffcient range from 0 - 0.08")

plt.figure(figsize=(5,4))
sns.jointplot(x='total_area_of_blades',y='efficiency_output',kind='hist',data=master_df)
plt.show()
print('data concentration arounf 3-5 total blade area')

sns.pairplot(data=master_df,vars=['disk_area','solidity_value','beta_angle_relative_to_rotation'])
plt.show()

print(f'solidity value reduce with disk area. but no linear relationship shown ')
print(f'higher the beta angle the solidity value tend to reduce')


# correlation analysis

# Identify numerical columns
numerical_columns = master_df.select_dtypes(include=[np.number]).columns
type(numerical_columns)
# Compute the correlation matrix only for numerical columns
corr_matrix = master_df[numerical_columns].corr()

# Plot the correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('correlation between numerical variables ', fontsize =20)
plt.show()


# dropping columns that are not required for model development
master_df = master_df.drop(columns=[
'propellers_name', 'blades_name', 'propellers_brand',
 'propellers_brand'])
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

model_1= master_df # without imputation

imputer = SimpleImputer(strategy='mean')
df_imputed = pd.DataFrame(imputer.fit_transform(master_df), columns=master_df.columns)
model_2 = df_imputed # with imputation

model_3 = master_df.drop(columns=['solidity_value']) #without solidity value

# performance of a propeller is given by efficiency,power and thrust coefficients. So in order to identify the preformace of a propeller we need to 
# develop models for each parameters 

def split_data(features, target):
    return train_test_split(features, target, test_size=0.2, random_state=42)

# Scale features
def scale_features(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

# Model training and evaluation

# Model training, evaluation, and plotting
def train_and_evaluate(X_train, X_test, y_train, y_test, description, ax):
    model = xgb.XGBRegressor(objective='reg:squarederror')
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    print(f"{description}: MSE = {mse}, R² = {r2}")
    
    # Plotting Actual vs Predicted values
    ax.scatter(y_test, preds, color='blue')
    ax.set_xlabel('Actual Values')
    ax.set_ylabel('Predicted Values')
    ax.set_title(f'{description} - Actual vs. Predicted Values')
    ax.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--k')
    
    return preds, mse, r2

# Feature and target extraction (assuming this function needs to be defined)
def feature_and_target(df):
    # Placeholder for actual feature and target extraction logic
    features = df.drop(columns=['thrust_coefficient_output', 'power_coefficient_output', 'efficiency_output'])
    target_thrust = df['thrust_coefficient_output']
    target_power = df['power_coefficient_output']
    target_efficiency = df['efficiency_output']
    return features, target_thrust, target_power, target_efficiency

# Model builder for different blade numbers
def model_builder(number, df):
    model = df[df['number_of_blades'] == number]
 
    features, target_thrust, target_power, target_efficiency = feature_and_target(model)
   
    # Prepare data splits
    X_train, X_test, y_train_thrust, y_test_thrust = split_data(features, target_thrust)
    _, _, y_train_power, y_test_power = split_data(features, target_power)
    _, _, y_train_efficiency, y_test_efficiency = split_data(features, target_efficiency)

    # Standardize features
    X_train_scaled, X_test_scaled = scale_features(X_train, X_test)

    # Prepare subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Train and evaluate models for each target
    print("Thrust Model Evaluation")
    train_and_evaluate(X_train_scaled, X_test_scaled, y_train_thrust, y_test_thrust, "Thrust", axes[0])

    print("Power Model Evaluation")
    train_and_evaluate(X_train_scaled, X_test_scaled, y_train_power, y_test_power, "Power", axes[1])

    print("Efficiency Model Evaluation")
    train_and_evaluate(X_train_scaled, X_test_scaled, y_train_efficiency, y_test_efficiency, "Efficiency", axes[2])
    
    plt.tight_layout()
    plt.show()
     
# modeling for 2 blade propellers 

model_builder(2, model_1)
model_builder(2,model_2)
model_builder(2,model_3)

blade2_model_evaluation  = '''
#################
Model  performance evaluation when blade count is 2

Model 1 : without missing value imputation 
has the best performance for Thrust and Power.

Model 2: with missing value imputation
has the best performance for Efficiency.

Model 3 : without solidity value 
has the lowest performance across all metrics.

################

'''

print(blade2_model_evaluation)
# modeling for 2 blade propeller

model_builder(3,model_1)
model_builder(3,model_2)
model_builder(3,model_3)

blade3_model_evaluation  = '''
#################
Model  performance evaluation when blade count is 3

Model 1 : without missing value imputation 
has the lowest MSE for Power but slightly more scatter and outliers in Efficiency 
compared to the other models

Model 2: with missing value imputation
performs well but is slightly inferior to Model 3, particularly in Power and Efficiency.

Model 3 : without solidity value 
shows the best overall performance based on both the metrics and visualization, 
with the lowest MSE for Thrust and Efficiency and highest R² values
################

'''

print(blade3_model_evaluation)


# modeling for 2 blade propellers 

model_builder(4,model_1)
model_builder(4,model_2)
model_builder(4,model_3)


blade4_model_evaluation  = '''
#################
Model  performance evaluation when blade count is 4

Model 1 : without missing value imputation 
Best Power performance, but slightly lower Efficiency R².

Model 2: with missing value imputation
Best Efficiency performance, but higher Thrust MSE.

Model 3 : without solidity value 
Best Thrust performance (lowest MSE, highest R²), but lower Power performance
################

'''

print(blade4_model_evaluation)

