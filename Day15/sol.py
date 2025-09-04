import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules


df = pd.read_csv("Dataset_Day15.csv", header=None)
transactions = df.apply(lambda row: [item for item in row if pd.notnull(item) and item != ''], axis=1)

print(f" Original dataset shape: {df.shape}")
print(f" Sample cleaned transactions:\n{transactions.head().tolist()}")

#  Encode transactions
te = TransactionEncoder()
te_array = te.fit(transactions).transform(transactions)
df_encoded = pd.DataFrame(te_array, columns=te.columns_)
print("\n Transactions encoded for market basket analysis.")
print(df_encoded.head())
print(df_encoded.columns.tolist())

#  Find frequent itemsets (min_support=0.02)
frequent_itemsets = apriori(df_encoded, min_support=0.02, use_colnames=True)
print(f"\n Total frequent itemsets found: {len(frequent_itemsets)}")

# Step 4: Generate association rules (min confidence = 15%)
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.15)
print(f" Total association rules generated: {len(rules)}")

# 4a: Top 5 rules by lift
top_lift = rules.sort_values(by='lift', ascending=False).head(5)
print("\n Top 5 rules by LIFT:")
print(top_lift[['antecedents', 'consequents', 'lift']])

#  4b: Top 5 rules by leverage
top_leverage = rules.sort_values(by='leverage', ascending=False).head(5)
print("\n Top 5 rules by LEVERAGE:")
print(top_leverage[['antecedents', 'consequents', 'leverage']])

#  Zhang's metric
rules['zhang'] = rules.apply(
    lambda x: ((x['support'] * (1 - x['lift'])) /
              ((x['support'] - x['confidence'] * x['consequent support']) + 1e-10))
    if x['lift'] != 1 else 0,
    axis=1
)

top_zhang = rules.sort_values(by='zhang', ascending=False).head(2)
bottom_zhang = rules.sort_values(by='zhang', ascending=True).head(2)

print("\nTop 2 rules by Zhang’s metric:")
print(top_zhang[['antecedents', 'consequents', 'zhang']])

print("\n Bottom 2 rules by Zhang’s metric:")
print(bottom_zhang[['antecedents', 'consequents', 'zhang']])



