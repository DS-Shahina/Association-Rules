"Problem 1"

import pandas as pd

# Read data into Python
book = pd.read_csv("D:/C DRIVE-SSD DATA backup 15-12-2020/Desktop/360DigiTmg Assignment/Association Rules/book.csv")
book.head()
book.shape
book.columns.values
book.dtypes
book.info()

#Exploratory Data Analysis
# Check the descriptive statistics of numeric variables
book.describe()
#1st moment Business Decision # Measures of Central Tendency / First moment business decision
book.mean()
book.median()
book.mode()

#2nd moment business decision # Measures of Dispersion / Second moment business decision
book.var() 
book.std()

#3rd moment Business Decision
book.skew()

#4th moment Business Decision
book.kurtosis()

#Graphical Representation
import matplotlib.pyplot as plt # mostly used for visualization purposes 
import numpy as np
import seaborn as sns
#Barplot for cetagorical variable like Binary variables 0's & 1's
for i, predictor in enumerate(book):
    plt.figure(i)
    sns.barplot(data=book, x=predictor)
    
#Model Building    
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules


frequent_itemsets = apriori(book, min_support = 0.0075, max_len = 4, use_colnames = True)
# max_len = 4 means (4 item combinations)
frequent_itemsets.shape
# Most Frequent item sets based on support 
frequent_itemsets.sort_values('support', ascending = False, inplace = True) #descending order

plt.bar(x = list(range(0, 11)), height = frequent_itemsets.support[0:11], color ='rgmyk')
plt.xticks(list(range(0, 11)), frequent_itemsets.itemsets[0:11], rotation=20)
plt.xlabel('item-sets')
plt.ylabel('support')
plt.show()

rules = association_rules(frequent_itemsets, metric = "lift", min_threshold = 1)
rules.shape
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
rules_no_redudancy = rules.iloc[index_rules, :]

# Sorting them with respect to list and getting top 10 rules 
top10rules = rules_no_redudancy.sort_values('lift', ascending = False).head(10)

# So, we see top 10 Association Rules of books which is more purchased, because of this combinations we will again make a profit and their lift ratio is also more that 1 so that is the best rule we can say.

top10rules.to_csv("booktop10Rules.csv", encoding = "utf-8")

import os
path = "D:/C DRIVE-SSD DATA backup 15-12-2020/Desktop/360DigiTmg Assignment/Association Rules"
os.chdir(path) # current working directory
    
    
    