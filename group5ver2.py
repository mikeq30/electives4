import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# import apriori as apr


#Data = pd.read_csv('restaurant_sales.csv', header= None) #Use this if file is in D: or C/ path
Data = pd.read_csv('D:\Codes\Python\Elective-4_4.1-PUP\electives4\\restaurant_sales.csv', header= None) #Lalaine Personal Path

print(Data)
Data.replace(np.nan, 0, inplace= False)
# change 0 value into Nan

#initializing the list
transacts = []

# Populating a list of transactions
for i in range (0, 1001):
    transacts.append([str(Data.values[i,j]) for j in range(0, 5)])
    from apyori import apriori
rule = apriori(transactions = transacts, min_support= 0.10, min_confidence = 0.50, min_lift= 3, max_length= 2)

output = list(rule) # non-tabular output
#! Putting output into a pandas dataframe

def inspect(output):
    lhs = [tuple(result[2][0][0])[0] for result in output]
    rhs = [tuple(result[2][0][1])[0] for result in output]
    support = [result[1] for result in output]
    confidence = [result[2][0][2] for result in output]
    lift = [result[2][0][3] for result in output]
    
    return list(zip(lhs, rhs, support, confidence, lift))

output_DataFrame= pd.DataFrame(inspect(output), columns =['Left_Hand', 'Right_Hand', 'Support', 'Confidence', 'Lift'])
print("Displaying the results non-sorted")
print(output_DataFrame)

print("\n\nDisplaying the results sorted")
output_DataFrame['Lift'] = pd.to_numeric(output_DataFrame['Lift'])
print(output_DataFrame.nlargest(n= 10, columns= 'Lift'))