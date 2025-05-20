import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the BIO-tagged CSV
df = pd.read_csv("bert/data/bio_output.csv")

# Display basic statistics
print("Total token count:", len(df))
print("Unique BIO labels:", df['bio_label'].unique())
print("Distribution of BIO labels:")
print(df['bio_label'].value_counts())

# Plot distribution of BIO labels
plt.figure(figsize=(8, 4))
sns.countplot(x='bio_label', data=df, 
              order=df['bio_label'].value_counts().index,
              palette="viridis")
plt.title("Distribution of BIO Labels")
plt.xlabel("BIO Label")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Explore token counts per document: group by doc_id
doc_token_counts = df.groupby('doc_id').size().reset_index(name="token_count")
print("\nToken counts per document:")
print(doc_token_counts.head())

# Plot distribution of token counts per document
plt.figure(figsize=(8, 4))
sns.histplot(doc_token_counts['token_count'], bins=20, kde=True, color='skyblue')
plt.title("Distribution of Token Counts per Document")
plt.xlabel("Token Count")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# If you want to analyze only tokens that have an entity tag (ignoring "O"):
entity_df = df[df['bio_label'] != "O"]
print("\nTop tokens among named entities:")
entity_counts = entity_df['token'].value_counts().reset_index()
entity_counts.columns = ['token', 'count']
print(entity_counts.head(20))

# Plot the top 20 frequent tokens from the entities
plt.figure(figsize=(8, 6))
sns.barplot(data=entity_counts.head(20), x='count', y='token', palette="magma")
plt.title("Top 20 Frequent Tokens among Named Entities")
plt.xlabel("Frequency")
plt.ylabel("Token")
plt.tight_layout()
plt.show()

# Optional: Explore token sequences by joining tokens per document for further analysis
docs = df.groupby('doc_id')['token'].apply(lambda x: " ".join(x)).reset_index()
print("\nSample joined document tokens:")
print(docs.head())

# # Save exploration results (optional)
# doc_token_counts.to_csv("doc_token_counts.csv", index=False)
# entity_counts.to_csv("entity_token_counts.csv", index=False)
