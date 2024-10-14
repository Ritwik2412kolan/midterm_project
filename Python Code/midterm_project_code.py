# Import necessary libraries
import pandas as pd
from itertools import combinations
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth
from mlxtend.preprocessing import TransactionEncoder
import time

# Define file paths for the datasets (use raw strings to avoid escape sequence issues)
file_paths = {
    "Amazon": r"C:\Users\rithv\Downloads\Amazon_Updated_Transactions.csv",
    "BestBuy": r"C:\Users\rithv\Downloads\BestBuy_Updated_Transactions.csv",
    "KMart": r"C:\Users\rithv\Downloads\KMart_Updated_Transactions.csv",
    "Nike": r"C:\Users\rithv\Downloads\Nike_Updated_Transactions.csv",
    "Generic" : r"C:\Users\rithv\midterm_project\Generic_Updated_Transactions.csv"
}

# Load transactions from CSV files
def load_transactions(file_path):
    df = pd.read_csv(file_path)
    transactions = df['Items'].apply(lambda x: x.split(', ')).tolist()
    return transactions

# Brute Force Method for generating frequent itemsets
def generate_frequent_itemsets(transactions, support_threshold):
    item_count = {}
    for transaction in transactions:
        for item in transaction:
            item_count[item] = item_count.get(item, 0) + 1

    frequent_itemsets = {1: {item: count for item, count in item_count.items() if count / len(transactions) >= support_threshold}}

    k = 2
    while True:
        prev_itemsets = list(frequent_itemsets[k - 1].keys())
        new_itemsets = list(combinations(prev_itemsets, k))
        item_count = {}
        for transaction in transactions:
            transaction_set = set(transaction)
            for itemset in new_itemsets:
                if set(itemset).issubset(transaction_set):
                    item_count[itemset] = item_count.get(itemset, 0) + 1

        frequent_itemsets[k] = {itemset: count for itemset, count in item_count.items() if count / len(transactions) >= support_threshold}
        if not frequent_itemsets[k]:
            del frequent_itemsets[k]
            break
        k += 1
    return frequent_itemsets

# Apriori Algorithm
def apriori_algorithm(transactions, support_threshold, confidence_threshold):
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df = pd.DataFrame(te_ary, columns=te.columns_)

    frequent_itemsets = apriori(df, min_support=support_threshold, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=confidence_threshold)

    return frequent_itemsets, rules

# FP-Growth Algorithm
def fpgrowth_algorithm(transactions, support_threshold, confidence_threshold):
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df = pd.DataFrame(te_ary, columns=te.columns_)

    frequent_itemsets = fpgrowth(df, min_support=support_threshold, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=confidence_threshold)

    return frequent_itemsets, rules

# Timing function for comparison
def measure_execution_time(algorithm_func, *args):
    start_time = time.time()
    result = algorithm_func(*args)
    end_time = time.time()
    return result, end_time - start_time

# Main program loop
while True:
    # Prompt user to select a database or exit
    print("\nAvailable databases:")
    for i, name in enumerate(file_paths.keys(), 1):
        print(f"{i}. {name}")
    print("0. Exit")
    
    choice = int(input("Enter the number corresponding to the database you'd like to choose (or 0 to exit): "))

    # Exit the loop if the user chooses 0
    if choice == 0:
        print("Exiting the program.")
        break

    # Get the selected database name
    db_name = list(file_paths.keys())[choice - 1]

    # Load the selected transactions
    transactions = load_transactions(file_paths[db_name])
    print(f"Loaded {len(transactions)} transactions from {db_name}.")

    # Prompt user for support and confidence thresholds
    support_threshold = float(input("Enter support threshold in % (e.g., 10 for 10%): ")) / 100
    confidence_threshold = float(input("Enter confidence threshold in % (e.g., 20 for 20%): ")) / 100

    print(f"\nProcessing {db_name} with support {support_threshold * 100}% and confidence {confidence_threshold * 100}%...")

    # Brute Force
    bf_result, bf_time = measure_execution_time(generate_frequent_itemsets, transactions, support_threshold)
    print(f"\nBrute Force Frequent Itemsets:\n{bf_result}")
    print(f"Brute Force Time: {bf_time:.4f}s")

    # Apriori
    apriori_result, apriori_time = measure_execution_time(apriori_algorithm, transactions, support_threshold, confidence_threshold)
    print(f"\nApriori Frequent Itemsets:\n{apriori_result[0]}")
    print(f"Apriori Rules:\n{apriori_result[1]}")
    print(f"Apriori Time: {apriori_time:.4f}s")

    # FP-Growth
    fp_result, fp_time = measure_execution_time(fpgrowth_algorithm, transactions, support_threshold, confidence_threshold)
    print(f"\nFP-Growth Frequent Itemsets:\n{fp_result[0]}")
    print(f"FP-Growth Rules:\n{fp_result[1]}")
    print(f"FP-Growth Time: {fp_time:.4f}s")

    # Performance summary
    print("\nTiming Performance Comparison ")
    print(f"Brute Force Time: {bf_time:.4f}s")
    print(f"Apriori Time: {apriori_time:.4f}s")
    print(f"FP-Growth Time: {fp_time:.4f}s")

    # Ask if the user wants to run another analysis
    continue_choice = input("\nDo you want to analyze another dataset? (yes/no): ").strip().lower()
    if continue_choice != 'yes':
        print("Exiting the program.")
        break
