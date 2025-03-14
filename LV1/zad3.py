numbers=[]

while True:
    num=input('Input a number: ')    
    if num=='Done':
        break
    else:
        try:
            number=(int(num))
            numbers.append(number)
        except:
            print("That is not a number")

print(f'length {len(numbers)}')
print(f'Average {sum(numbers)/len(numbers)}')
print(f'Min {min(numbers)}')
print(f'Max {max(numbers)}')
numbers.sort()
print(numbers)
