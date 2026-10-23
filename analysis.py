import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("data.csv")

# Display basic analysis
print("=" * 45)
print("       STUDENT DATA ANALYSIS")
print("=" * 45)

print("\n1. Dataset Summary:")
print(df.describe().round(2))

print("\n2. Average Values:")
print(df.mean().round(2))

print("\n3. Missing Values:")
print(df.isnull().sum())

print("\n4. Correlation with Final Marks:")
print(df.corr()["Final_Marks"].round(2))

# Graph 1: Study Hours vs Final Marks
plt.figure(figsize=(8, 5))
sns.scatterplot(
    data=df,
    x="Study_Hours",
    y="Final_Marks",
    s=100
)
plt.title("Study Hours vs Final Marks")
plt.xlabel("Study Hours")
plt.ylabel("Final Marks")
plt.grid(True)
plt.tight_layout()
plt.savefig("study_hours_vs_marks.png")
plt.show()

# Graph 2: Attendance vs Final Marks
plt.figure(figsize=(8, 5))
sns.scatterplot(
    data=df,
    x="Attendance",
    y="Final_Marks",
    s=100
)
plt.title("Attendance vs Final Marks")
plt.xlabel("Attendance (%)")
plt.ylabel("Final Marks")
plt.grid(True)
plt.tight_layout()
plt.savefig("attendance_vs_marks.png")
plt.show()

# Graph 3: Correlation Heatmap
plt.figure(figsize=(8, 5))
sns.heatmap(
    df.corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)
plt.title("Student Performance Correlation Heatmap")
plt.tight_layout()
plt.savefig("correlation_heatmap.png")
plt.show()

print("\nAnalysis completed successfully!")
print("Three graphs saved in the project folder.")
