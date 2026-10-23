import pandas as pd

# Load the student dataset
df = pd.read_csv("data.csv")

print("=" * 45)
print("   STUDENT PERFORMANCE PREDICTOR")
print("=" * 45)

print("\nDataset loaded successfully!")
print("\nFirst five student records:")
print(df.head())

print("\nDataset Information:")
print("Total student records:", len(df))
print("Total columns:", len(df.columns))

print("\nColumn names:")
print(df.columns.tolist())