import pandas as pd
from apyori import apriori  # For implementing the Apriori algorithm

# Load the groceries dataset
data = pd.read_csv('groceries - groceries.csv', header=None, on_bad_lines='skip')

# Prepare the dataset for the Apriori algorithm
transactions = [[str(value) for value in row if pd.notnull(value)] for row in data.values]

# Apply the Apriori algorithm with lower thresholds
rules = apriori(transactions, min_support=0.01, min_confidence=0.1)

# Extract rules and their support and confidence
results = list(rules)

# Print support and confidence for each rule
if results:
    for result in results:
        support = result.support
        items = result.items
        if result.ordered_statistics:
            for ordered_stat in result.ordered_statistics:
                confidence = ordered_stat.confidence
                print(f"Rule: {list(items)}, Support: {support:.4f}, Confidence: {confidence:.4f}")
else:
    print("No rules found with the given support and confidence thresholds.")
