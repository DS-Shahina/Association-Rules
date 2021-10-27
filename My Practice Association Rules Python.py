# Implementing Apriori algorithm from mlxtend

# conda install mlxtend
# or
# pip install mlxtend - console

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# It is not in dataframe format it's an text file or excel file you can see where rach row is a transaction
groceries = []
# Pandas can't be used
#use open function , you can read, write and execute also
with open("D:/C DRIVE-SSD DATA backup 15-12-2020/Desktop/360digitmg material/Association Rules/groceries.csv") as f:
    groceries = f.read()
    
# splitting the data into separate transactions using separator as "\n"
groceries = groceries.split("\n")   # \n is new line character  

# I want each product separate now so i use , as delimetere
groceries_list = []
for i in groceries:
    groceries_list.append(i.split(","))

# Choose each item from the list
all_groceries_list = [i for item in groceries_list for i in item]

# Same product purchase multiple times so i want to count
from collections import Counter # ,OrderedDict 

item_frequencies = Counter(all_groceries_list) # How many times each product purchase

#after sorting
item_frequencies = sorted(item_frequencies.items(), key = lambda x:x[1]) # How many times each product purchase in ascending order
# Lambda function is a nameless fuction and in this it used to count in ascending order from 1 to last (name of product : number of item appers)

# see 3rd row is empty, original it has 9835 transaction and now it show 9836 because new line character it adds one more transaction, a new row and a new record is capture

# Storing frequencies and items in separate variables 
frequencies = list(reversed([i[1] for i in item_frequencies])) # descending order
items = list(reversed([i[0] for i in item_frequencies]))

# barplot of top 10 
import matplotlib.pyplot as plt

plt.bar(height = frequencies[0:11], x = list(range(0, 11)), color = 'rgbkymc')
plt.xticks(list(range(0, 11), ), items[0:11]) # Scale Values
plt.xlabel("items")
plt.ylabel("Count")
plt.show()

# Creating Data Frame for the transactions data
groceries_series = pd.DataFrame(pd.Series(groceries_list))
groceries_series = groceries_series.iloc[:9835, :] # removing the last empty transaction

groceries_series.columns = ["transactions"] # column name

# creating a dummy columns for the each item in each transactions ... Using column names as item name
X = groceries_series['transactions'].str.join(sep = '*').str.get_dummies(sep = '*') # only take unique category - * denotes (unique columns)

frequent_itemsets = apriori(X, min_support = 0.0075, max_len = 4, use_colnames = True)
# max_len = 4 means (4 item combinations)

# Most Frequent item sets based on support 
frequent_itemsets.sort_values('support', ascending = False, inplace = True) #descending order

plt.bar(x = list(range(0, 11)), height = frequent_itemsets.support[0:11], color ='rgmyk') # top 10 products
plt.xticks(list(range(0, 11)), frequent_itemsets.itemsets[0:11], rotation=20)
plt.xlabel('item-sets')
plt.ylabel('support')
plt.show()

rules = association_rules(frequent_itemsets, metric = "lift", min_threshold = 1)
rules.head(20)
rules.sort_values('lift', ascending = False).head(10)

#profusion - observer carefully they are duplicate (product combinations)
#Antecedent and consequent is exchanging
# we want without duplicate

################################# Extra part ###################################
def to_list(i):
    return (sorted(list(i)))

ma_X = rules.antecedents.apply(to_list) + rules.consequents.apply(to_list)

ma_X = ma_X.apply(sorted)

rules_sets = list(ma_X) # convert into list

unique_rules_sets = [list(m) for m in set(tuple(i) for i in rules_sets)]
#Tuple is immutable but sets resist duplicate that's why we convert it into sets and then convert it into list

index_rules = [] # capture index (row id)

for i in unique_rules_sets:
    index_rules.append(rules_sets.index(i))

# getting rules without any redudancy 
rules_no_redudancy = rules.iloc[index_rules, :] # rules only which are unique

# Sorting them with respect to list and getting top 10 rules 
rules_no_redudancy.sort_values('lift', ascending = False).head(10)



