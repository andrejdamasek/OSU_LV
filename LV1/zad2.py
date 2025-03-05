try:
    number=float(input('Input number between 0.0 and 1.0 :'))
    if number<0.0 or number>1.0:
        raise ValueError("Number must be between 0.0 and 1.0")
    if number>=0.9:
        print('A')
    elif number>=0.8:
        print('B')
    elif number>=0.7:
        print('C')
    elif number>=0.6:
        print('D')
    elif number<0.6:
        print('F')

except ValueError as err:
    print(err)