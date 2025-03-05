working_hours = int(input('Input number of working hours '))
price_per_hour= float(input('Euro/h :'))

def total_euro(working_hours,price_per_hour):
    return working_hours*price_per_hour
    

print(f'Total amount: { total_euro(working_hours,price_per_hour) } €')