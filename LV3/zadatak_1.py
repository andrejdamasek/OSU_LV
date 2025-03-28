import pandas as pd

data = pd.read_csv ('data_C02_emission.csv')

#a)
print (f"Number of data points: {len(data)}")
print(data.dtypes)
x=data.isnull().sum().sum()
if x<=0:
    print("No missing values")
else:
    print(f"There are omitted values, and these are: {x}")
    data.dropna(axis=0)
    data = data.reset_index(drop=True)

print(f"Number of duplicated elements: {data.duplicated().sum()}")
data.drop_duplicates()
data = data.reset_index(drop=True)

data["Make"] = data["Make"].astype("category")
data["Model"] = data["Model"].astype("category")
data["Vehicle Class"] = data["Vehicle Class"].astype("category")
data["Fuel Type"] = data["Fuel Type"].astype("category")
data["Transmission"] = data["Transmission"].astype("category")
#print(data.dtypes)
#b)
data_b=data.sort_values(by='Fuel Consumption City (L/100km)')
print("The biggest consumtion in the city: \n",data_b[['Make','Model','Fuel Consumption City (L/100km)']].head(3))
print("The smallest consumtion in the city: \n",data_b[['Make','Model','Fuel Consumption City (L/100km)']].tail(3))

#c)
data_c=data[(data['Engine Size (L)'] >= 2.5 ) & (data['Engine Size (L)'] < 3.5 )]
print("Number of vehicle that have engine size between 2.5L and 3.5L: ",len(data_c))
print("Average CO2 emissions: ",data_c['CO2 Emissions (g/km)'].mean())

#d)
data_d1=data[(data['Make']=='Audi')]
print("Number of vehicles from Audi: ",len(data_d1))
data_d2=data[(data['Make']=='Audi')&(data['Cylinders']==4)]
print("Average CO2 emissions from Audi with 4 cylinders: ",data_d2['CO2 Emissions (g/km)'].mean())

#e)
data_e4=data[(data['Cylinders']==4)]
data_e6=data[(data['Cylinders']==6)]
data_e8=data[(data['Cylinders']==8)]
print("Number of vehicles with 4 cylinders: ",len(data_e4))
print("Number of vehicles with 6 cylinders:",len(data_e6))
print("Number of vehicles with 8 cylinders:",len(data_e8))
print("Average CO2 emissions from 4 cylinders",data_e4['CO2 Emissions (g/km)'].mean())
print("Average CO2 emissions from 6 cylinders",data_e6['CO2 Emissions (g/km)'].mean())
print("Average CO2 emissions from 8 cylinders",data_e8['CO2 Emissions (g/km)'].mean())

#f)

data_fd=data[(data['Fuel Type']=='D')]
data_fg=data[(data['Fuel Type']=='X')]
print("Average fuel consumption of diesel in city",data_fd['Fuel Consumption City (L/100km)'].mean())
print("Average fuel consumption of regular gasoline in city",data_fg['Fuel Consumption City (L/100km)'].mean())
print("Median value of disel",data_fd['Fuel Consumption City (L/100km)'].median())
print("Median value of regular gasoline",data_fg['Fuel Consumption City (L/100km)'].median())

#g)
data_g=data[(data['Fuel Type']=='D')&(data['Cylinders']==4)]
data_g.sort_values(by=['Fuel Consumption City (L/100km)'])
print(data_g[['Make','Model','Fuel Consumption City (L/100km)']].tail(1))

#h)
data_h=data[(data['Transmission'].str[0] =='M')]
print("Number of vehicles with manual transmission: ",len(data_h))

#i)
print(data.corr(numeric_only=True))
#sto je veci engine size to je veci broj cilindara, potrosnja i emisija co2, a manja je kombinirana potrosnja ili mpg