
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

#Data = pd.read_csv('restaurant_sales.csv', header= None) #Use this if file is in D: or C/ path
Data = pd.read_csv('D:\Codes\Python\Elective-4_4.1-PUP\electives4\\restaurant_sales.csv', header= None) #Lalaine Personal Path


print(Data)
Data.replace(np.nan, 0, inplace= False)
# change 0 value into Nan

#! Support percentage
frequent_itemsets= apriori(Data, min_support= 0.10, use_colnames= True)
print(frequent_itemsets)

#! Confidence Threshold
rules= association_rules(frequent_itemsets, metric='lift', min_threshold=0.50)
print("\n\n", rules)

##* A leverage value of 0 indicates independence. Range will be [-1 1]
##* A high conviction value means that the consequent is highly depending on the antecedent and range [0 inf]

rules.sort_values('lift',ascending=False) # Sort Data
print("\n\n", rules)
print(type(rules))

rules.to_csv("Apriori_Output.csv", index=False) # Get Output into CSV File
