#DF
#pip install pandas
import pandas as pd

#Your file paths here, \\ for windows, / for mac/linux
file_paths = [
    "Your_Path\\ACSDT5Y2022.B19083-Data.csv",
    "Your_Path\\ACSST5Y2022.S0101-Data.csv",
    "Your_Path\\ACSST5Y2022.S1903-Data.csv"
]

#Read the data into dfs
df1, df2, df3 = [pd.read_csv(path, index_col=False, low_memory=False).iloc[:, :-1] for path in file_paths]

#Merge dfs on the common column "GEO_ID" which are the unique identifiers
merged_df = pd.merge(pd.merge(df1, df2, on="GEO_ID", how="inner"), df3, on="GEO_ID", how="inner")

#Save the merged dataframe to a new CSV file, waypoint
merged_df.to_csv("Your_Path\\combined_data.csv", index=False)

#Read the merged dataframe, our previous waypoint
df4 = pd.read_csv("Your_Path\\combined_data.csv")

#Display all rows and thier dtypes, reset
pd.set_option('display.max_rows', None)
print("\nData types in df4:")
print(df4.dtypes)
pd.reset_option('display.max_rows')

#Quick summary of whats in the first rows for a few columns
df4.head()

#Drop the second row
df4 = df4.drop(0)

#Reset the index after dropping row
df4 = df4.reset_index(drop=True)

df4.head()

#The columns we want to use
selected_columns = ["GEO_ID", "NAME_x", "B19083_001E", "S1903_C02_012E", 
    "S1903_C03_012E", "S0101_C01_001E", "S0101_C01_007E", "S0101_C01_008E", 
    "S0101_C01_009E", "S0101_C01_010E", "S0101_C01_032E", "S0101_C01_033E"]

#Simplify df
df5 = df4[selected_columns]

#Rename columns to custom names
column_mapping = {
    "GEO_ID": "Geography_ID",
    "NAME_x": "Geographic_Area_Name",
    "B19083_001E": "Gini_Index",
    "S1903_C02_012E": "Percent_of_Distribution_HOUSEHOLD_INCOME_BY_AGE_25_to_44_years",
    "S1903_C03_012E": "Median_income_by_age_25_to_44_years_in_dollars",
    "S0101_C01_001E": "Total_Population",
    "S0101_C01_007E":'Total_Population_25_29',
    "S0101_C01_008E":'Total_Population_30_34',
    "S0101_C01_009E":'Total_Population_35_39',
    "S0101_C01_010E":'Total_Population_40_44',
    "S0101_C01_032E": "Median_age_years",
    "S0101_C01_033E": "Sex_ratio_males_per_100_females",
}

#Apply rename columns in df5
df5 = df5.rename(columns=column_mapping)


#Show header of simplifed df to inspect datatypes 
pd.set_option('display.max_columns', None)
print("\nData types in df5:")
print(df5.head())
pd.reset_option('display.max_columns')

#Coerce to numeric, for numeric columns
numeric_columns = ["Gini_Index", "Percent_of_Distribution_HOUSEHOLD_INCOME_BY_AGE_25_to_44_years", "Median_income_by_age_25_to_44_years_in_dollars", "Total_Population", 
                    'Total_Population_25_29','Total_Population_30_34','Total_Population_35_39','Total_Population_40_44', "Median_age_years", 
                    "Sex_ratio_males_per_100_females"]
df5[numeric_columns] = df5[numeric_columns].apply(pd.to_numeric, errors='coerce')


#Check where the nans are
nan_values = df5.isna().sum()
print("NaN values in each column:")
print(nan_values)

#If nans present change to floats, convert to most efficient dtype that preserves info
custom_dtypes = {
    "Geography_ID": str,
    "Geographic_Area_Name": str,
    "Gini_Index": 'float32',
    "Percent_of_Distribution_HOUSEHOLD_INCOME_BY_AGE_25_to_44_years": 'float32',
    "Median_income_by_age_25_to_44_years_in_dollars": 'float32',
    "Total_Population": 'uint32',
    "Total_Population_25_29":'uint32',
    "Total_Population_30_34":'uint32',
    "Total_Population_35_39":'uint32',
    "Total_Population_40_44":'uint32',
    "Median_age_years": 'float32',
    "Sex_ratio_males_per_100_females": 'float32',
}

# Convert columns to custom data types
df5 = df5.astype(custom_dtypes)


#Sanity check
df5.head()

#Waypoint
df5.to_csv("Your_Path\\custom_combined_data.csv",float_format='%.6f', index=False)

#Modify

df6 = pd.read_csv("Your_Path\\custom_combined_data.csv", dtype=custom_dtypes)
#Sum the columns of ages to create one age total pop column
columns_to_sum = [
    "Total_Population_25_29",
    "Total_Population_30_34",
    "Total_Population_35_39",
    "Total_Population_40_44",
]

#Create a new column with the sum of specified columns for each row
df6["Total_population_25_to_44"] = df6[columns_to_sum].astype(int).sum(axis=1)


#Create age percentage column 
df6['Percent_of_population_25_to_44'] = df6['Total_population_25_to_44'] / df6['Total_Population']

#Inversed median age scaled inversed
df6['Inverse_Gini_Index'] = 1 - df6['Gini_Index']
df6['Inverse_Median_age_years'] = 1 - (df6['Median_age_years'] / 1000)

#Sanity check
pd.set_option('display.max_columns', None)

df6.head()

pd.reset_option('display.max_columns')

#Waypoint
df7 = df6.copy()

#Normalize the data
from sklearn.preprocessing import MinMaxScaler

#Numeric columns we want scaled
columns_to_scale = ['Inverse_Gini_Index', 'Percent_of_Distribution_HOUSEHOLD_INCOME_BY_AGE_25_to_44_years',
                    'Median_income_by_age_25_to_44_years_in_dollars', 'Total_Population',
                    'Total_population_25_to_44', 'Inverse_Median_age_years', 'Sex_ratio_males_per_100_females',
                    'Percent_of_population_25_to_44']

scaler = MinMaxScaler()

#Create new columns with scaled values
for column in columns_to_scale:
    df7[column + '_scaled'] = scaler.fit_transform(df7[[column]])

#Sanity check
pd.set_option('display.max_columns', None)

df7.head()

pd.reset_option('display.max_columns')
 
#Waypoint
df8 = df7.copy()

#Create an index or weighted sum, can create mutiple
#Set weight
weights = {
    'Inverse_Gini_Index_scaled': 0.1,
    'Percent_of_Distribution_HOUSEHOLD_INCOME_BY_AGE_25_to_44_years_scaled': 0.20,
    'Median_income_by_age_25_to_44_years_in_dollars_scaled': 0.55,
    'Inverse_Median_age_years_scaled': 0.10,
    'Percent_of_population_25_to_44_scaled': 0.05
}

#Create a new column with the weighted sum
df8['Highest_Early_Career_Income'] = df8[weights.keys()].mul(weights.values()).sum(axis=1)


weights = {
    'Inverse_Gini_Index_scaled': 0.1,
    'Percent_of_Distribution_HOUSEHOLD_INCOME_BY_AGE_25_to_44_years_scaled': 0.10,
    'Median_income_by_age_25_to_44_years_in_dollars_scaled': 0.55,
    'Inverse_Median_age_years_scaled': 0.20,
    'Percent_of_population_25_to_44_scaled': 0.05
}

# Create a new column with the weighted sum
df8['Highest_Early_Career_Income_2'] = df8[weights.keys()].mul(weights.values()).sum(axis=1)

#Add GEOID column for json identifier matching leading apostrophe to maintain leading zeros and last 11 digits
df8['GEOID'] = "'" + df8['Geography_ID'].astype(str).str[-11:]

#Sanity check
pd.set_option('display.max_columns', None)

df8.head()

pd.reset_option('display.max_columns')


#Save to CSV
custom_dtypes2 = {
    "Geography_ID": str,
    "Geographic_Area_Name": str,
    "GEOID": str,
    "Gini_Index": 'float32',
    "Percent_of_Distribution_HOUSEHOLD_INCOME_BY_AGE_25_to_44_years": 'float32',
    "Median_income_by_age_25_to_44_years_in_dollars": 'float32',
    "Total_Population": 'uint32',
    "Total_population_25_to_44": 'uint32',
    "Median_age_years": 'float32',
    "Sex_ratio_males_per_100_females": 'float32',
    "Highest_Early_Career_Income": 'float32',
    "Highest_Early_Career_Income_2": 'float32',
    "Percent_of_population_25_to_44_scaled": 'float32',
    "Sex_ratio_males_per_100_females_scaled": 'float32',
    "Inverse_Median_age_years_scaled": 'float32',
    "Total_population_25_to_44_scaled": 'float32',
    "Total_Population_scaled": 'float32',
    "Median_income_by_age_25_to_44_years_in_dollars_scaled": 'float32',
    "Percent_of_Distribution_HOUSEHOLD_INCOME_BY_AGE_25_to_44_years_scaled": 'float32',
    "Inverse_Gini_Index_scaled": 'float32',
    "Inverse_Median_age_years": 'float32',
    "Inverse_Gini_Index": 'float32',
    "Percent_of_population_25_to_44": 'float32',
}

#Convert columns to custom data types
df9 = df8.astype(custom_dtypes2)

df9.head()


#sanity check
#Save the DataFrame to a new CSV file with custom data types
df9.to_csv("Your_Path\\custom_combined_data_preprocessed_2.csv",float_format='%.6f', index=False)

#Lets look at top 250 counties
top_rows = df9.nlargest(250, 'Highest_Early_Career_Income')

#Extract county and state information from the 'Geographic_Area_Name' column
county_state_info = top_rows['Geographic_Area_Name'].str.split(';')

#Extract third and fourth elements after splitting
top_rows['County_State'] = county_state_info.str[1].str.strip() + ', ' + county_state_info.str[2].str.strip()

#Create a new DataFrame for county occurrences
county_occurrences_df = pd.DataFrame(top_rows['County_State'].value_counts()).reset_index()

#Rename columns
county_occurrences_df.columns = ['County_State', 'Occurrences']

pd.set_option('display.max_rows', None)
#Display the new DataFrame
print(county_occurrences_df)
pd.reset_option('display.max_rows')


#Redundant
pd.set_option('display.max_rows', None)

print(top_rows)

pd.reset_option('display.max_rows')

import matplotlib.pyplot as plt
import seaborn as sns
#Display statistics of the 'Highest_Early_Career_Income' column
income_stats = df9['Highest_Early_Career_Income'].describe()
print("Statistics of Highest_Early_Career_Income:")
print(income_stats)

#Visualize the distribution of 'Highest_Early_Career_Income'
plt.figure(figsize=(10, 6))
sns.histplot(df9['Highest_Early_Career_Income'], bins=30, kde=True, color='skyblue')
plt.title('Distribution of Highest_Early_Career_Income')
plt.xlabel('Income Index')
plt.ylabel('Frequency')
plt.show()



import matplotlib.pyplot as plt
import seaborn as sns
#Display statistics of the 'Highest_Early_Career_Income' column
income_stats = df9['Highest_Early_Career_Income_2'].describe()
print("Statistics of Highest_Early_Career_Income_2:")
print(income_stats)

#Visualize the distribution of 'Highest_Early_Career_Income'
plt.figure(figsize=(10, 6))
sns.histplot(df9['Highest_Early_Career_Income_2'], bins=30, kde=True, color='skyblue')
plt.title('Distribution of Highest_Early_Career_Income_2')
plt.xlabel('Income Index')
plt.ylabel('Frequency')
plt.show()

#Need Updated json
#Create Visual
import geopandas as gpd
import copy
import pandas as pd
import plotly.express as px
import json
from concurrent.futures import ProcessPoolExecutor

#Function to create the choropleth map
def catestvisual():
    #Read the matched CSV file
    csv_file = 'Your_Path\\custom_combined_data_preprocessed_2.csv'
    df = pd.read_csv(csv_file, dtype = custom_dtypes2)
    #Remove the leading apostrophe from the 'GEOID' column in the DataFrame
    #Assuming your DataFrame is named df
    df['GEOID'] = df['GEOID'].str.lstrip("'")
    df = df[df['Total_Population'] > 0]
    df = df[df['Median_income_by_age_25_to_44_years_in_dollars'].notna() & (df['Median_income_by_age_25_to_44_years_in_dollars'] > 0)]
    #Check data types of columns
    print("Data Types:")
    print(df.dtypes)
    #Check for missing values
    print("\nMissing Values:")
    print(df.isnull().sum())
    with open("Your_Path/accesstoken.txt", "r") as token_file:
        token = token_file.read().strip()
    #Load the JSON data for California tracts
    with open('Your_Path/tracts_simp_form_3.json', 'r') as json_file:
        json_data = json.load(json_file)
    mapbox_access_token = "pk.eyJ1IjoiamFrZWR1Z2kiLCJhIjoiY2xsZndndmFtMHUzdzNycnlwN3pwdWIxMCJ9.N33rh6ImGOt8Eqov3bWUzw"
    px.set_mapbox_access_token(mapbox_access_token)
    fig = px.choropleth_mapbox(df,
                               geojson=json_data,
                               locations='GEOID',
                               color='Highest_Early_Career_Income',
                               color_continuous_scale="viridis",
                               range_color=(0.0, 1.0),
                               featureidkey="properties.GEOID",
                               mapbox_style="light",
                               zoom=3.5,
                               opacity=1.0,
                               center={"lat": 37.0902, "lon": -95.7129},
                               title='Tracts with Highest Early Career Income',
                               hover_data={
                                "Geography_ID": True,
                                "Geographic_Area_Name": True,
                                "GEOID": True,
                                "Gini_Index": True,
                                "Percent_of_Distribution_HOUSEHOLD_INCOME_BY_AGE_25_to_44_years": True,
                                "Median_income_by_age_25_to_44_years_in_dollars": True,
                                "Total_Population": True,
                                "Total_population_25_to_44": True,
                                "Median_age_years": True,
                                "Sex_ratio_males_per_100_females": True,
                                "Highest_Early_Career_Income_2": True,
                                "Highest_Early_Career_Income": True,
                                "Inverse_Gini_Index_scaled": True,
                                "Inverse_Median_age_years": True,
                                "Inverse_Gini_Index": True,
                                "Percent_of_population_25_to_44": True,

                            },
                               labels={
                                "Geography_ID": 'Geography_ID',
                                "Geographic_Area_Name": 'Geographic_Area_Name',
                                "GEOID": 'GEOID',
                                "Gini_Index": 'Gini_Index',
                                "Percent_of_Distribution_HOUSEHOLD_INCOME_BY_AGE_25_to_44_years": 'Percent_of_Distribution_HOUSEHOLD_INCOME_BY_AGE_25_to_44_years',
                                "Median_income_by_age_25_to_44_years_in_dollars": 'Median_income_by_age_25_to_44_years_in_dollars',
                                "Total_Population": 'Total_Population',
                                "Total_population_25_to_44": 'Total_population_25_to_44',
                                "Median_age_years": 'Median_age_years',
                                "Sex_ratio_males_per_100_females": 'Sex_ratio_males_per_100_females',
                                "Highest_Early_Career_Income_2": 'Highest_Early_Career_Income_2',
                                "Highest_Early_Career_Income": 'Highest_Early_Career_Income',
                                "Inverse_Gini_Index_scaled": 'Inverse_Gini_Index_scaled',
                                "Inverse_Median_age_years": 'Inverse_Median_age_years',
                                "Inverse_Gini_Index": 'Inverse_Gini_Index',
                                "Percent_of_population_25_to_44": 'Percent_of_population_25_to_44',

                            },
                             )
    fig = fig.update_traces(
        marker_line_width=0.0, #0.000000001
        marker_line_color='#D3D3D3',
        marker_opacity=0.87
    )
    fig.update_layout(
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        title={'xanchor':'center','x':0.5,'y': 1},
        coloraxis_colorbar={
            'title': 'Early Career Income Index',
            'title_side': 'bottom',
            'tickformat': '.6f',
            'tickvals': [df['Highest_Early_Career_Income'].min(), df['Highest_Early_Career_Income'].mean(), df['Highest_Early_Career_Income'].max()],
            'ticktext': ['Lowest Early Career Income', ' ','Highest Early Career Income'], 
            'orientation':'h',
            'x': 0.5,
            'xanchor': 'center',
            'y': -0.00000001,
            'yanchor': 'top',
            'len': 0.9,
        },
        annotations=[
            dict(
                text='Source: U.S. Census Bureau 2023 Shapefiles,<br>American Community Survey 2022 5-Yr Estimate Tract Level Data',
                xref='paper',
                yref='paper',
                x=0.01,
                y=0.01,
                showarrow=False,
                font=dict(size=15),
                align='left'
            )
        ],
        font=dict(family="Calibri", size=24, color="black")
    )
    fig.write_html('Your_Path\\Highest_Early_Career_Income.html')
    fig.show()

if __name__ == '__main__':
    catestvisual()









