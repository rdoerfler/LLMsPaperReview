import pandas as pd

# Read the JSONL file into a DataFrame
df = pd.read_json('./data/ReviewCritique.jsonl', lines=True)

# Access the 'decision' column
decision_column = df['decision']

# First write the script to g
# 
# et human reviews from 'ReviewCritique.jsonl' file
# key names are 'review#1', 'review#2', '#review3', '#review4', '#review5'
# other keys are 'decision', 'title', 'body_text'

# Also need to check whether 