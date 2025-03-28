import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv ('data_C02_emission.csv')
#a)
plt.figure()
data['CO2 Emissions (g/km)'].plot(kind='hist', bins = 20)
#najvise automobila ima koji proizvode  od 180 do 350 g/km CO2-a, a malo onih koji proizvode 
# do 150g/km isto kao i za one koji proizvode vise od 400g/km
plt.show()

#b)
data['Fuel Type'] = data['Fuel Type'].astype('category')
colors = {'Z': 'green', 'X': 'red', 'E': 'blue', 'D': 'black', 'N':'yellow'}
data.plot.scatter(x="Fuel Consumption City (L/100km)", y="CO2 Emissions (g/km)", c=data["Fuel Type"].map(colors), s=10)
#ethanol proizvodi manje CO2 nego ostala goriva s obziron na veliku srednju potrošnju,
# najmanju emisiju CO2 ima regular gasoline, pa onda dizel, najvecu potrošnju i najvecu kolicinui co2 proizvode 
#automobili koji se voze na premium gasopline
plt.show()

#c)
data.boxplot(column=['Fuel Consumption Hwy (L/100km)'], by='Fuel Type')
plt.show()
#da za premijum gorivo imamo punu podataka koji su veci od maksimuma isto kao u za dizel 
# koji ima podatke koji su manji od minimuma i maksimuma

#d)
fuel_counts = data.groupby("Fuel Type").size()
plt.figure()
plt.bar(fuel_counts.index, fuel_counts.values)
plt.show()

#e)
cylinder_grouped = data.groupby('Cylinders')['CO2 Emissions (g/km)'].mean()
cylinder_grouped.plot(kind='bar', x=cylinder_grouped.index, y=cylinder_grouped.values, xlabel='Cylinders', ylabel='CO2 emissions (g/km)', title='CO2 emissions by number of cylinders')
plt.show()
