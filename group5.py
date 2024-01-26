
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

#Data = pd.read_csv('restaurant_sales.csv', header= None) #Use this if file is in D: or C/ path
DataFrame = pd.read_csv('localrest_ver2.csv')
# Not Including the first two columns 
DataFrame = DataFrame.iloc[:, 2:]
DataFrame = DataFrame.map(lambda x: 1 if x != 0 else 0)

print(DataFrame)

#Remove the warning of using DataFrame with non bool type
DataFrame = DataFrame.astype(bool)
    
print("Enter minimum support in decimal (Example: .10)")
support = float(input("Support: "))

print("Enter minimum lift in decimal (Example: .30)")
liftnum = float(input("Lift: "))

try:
    #! Support percentage
    frequent_itemsets= apriori(DataFrame, min_support= support, use_colnames=True)
    print(frequent_itemsets)

    #! Confidence Threshold
    rules= association_rules(frequent_itemsets, metric='lift', min_threshold=liftnum)
    # print("\n\n", rules)

    ##* A leverage value of 0 indicates independence. Range will be [-1 1]
    ##* A high conviction value means that the consequent is highly depending on the antecedent and range [0 inf]

    rules.sort_values('lift',ascending=False) # Sort Data
    print("\n\n", rules)
    print(type(rules))

    rules.to_csv("Apriori_Output.csv", index=False) # Get Output into CSV File
except: 
    print("Theres no result with the given threshold")
    exit