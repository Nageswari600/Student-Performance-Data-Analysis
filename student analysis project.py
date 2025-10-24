#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Block-1: Import Libraries and Load Dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset (make sure student_scores.csv is in the same folder)
df = pd.read_csv("C:\\Users\\bhuky\\Downloads\\student_scores.csv")

print("✅ Dataset Loaded Successfully!\n")
print(df.head())


# In[3]:


# Block-2: Q1 — Compute mean, min, max scores by gender
gender_stats = df.groupby('gender')[['math', 'reading', 'writing']].agg(['mean', 'min', 'max'])

print("📊 Mean, Min, and Max Scores by Gender:\n")
print(gender_stats)


# In[4]:


# Block-3: Q2 — Analyze effect of parental education on average scores
df['average_score'] = df[['math', 'reading', 'writing']].mean(axis=1)

parent_edu_analysis = df.groupby('parental_education')['average_score'].mean().sort_values(ascending=False)
print("\n📈 Average Scores by Parental Education:\n")
print(parent_edu_analysis)

# Visualization
plt.figure(figsize=(8,5))
sns.barplot(x=parent_edu_analysis.index, y=parent_edu_analysis.values, palette='coolwarm')
plt.xticks(rotation=45)
plt.title("Average Scores by Parental Education Level")
plt.xlabel("Parental Education Level")
plt.ylabel("Average Score")
plt.show()


# In[6]:


# Block-4: Q3 — Handle missing study_hours using median imputation
print("\n🩺 Missing values before imputation:", df['study_hours'].isnull().sum())

median_hours = df['study_hours'].median()
df['study_hours'] = df['study_hours'].fillna(median_hours)  # ✅ Assign back instead of inplace

print("✅ Missing values after imputation:", df['study_hours'].isnull().sum())


# In[7]:


# Block-5: Q4 — Compare performance across subjects (math vs reading vs writing)
subject_means = df[['math', 'reading', 'writing']].mean()
print("\n📚 Average Scores per Subject:\n", subject_means)

# Bar plot for comparison
plt.figure(figsize=(6,4))
sns.barplot(x=subject_means.index, y=subject_means.values, palette='viridis')
plt.title("Comparison of Average Scores Across Subjects")
plt.xlabel("Subject")
plt.ylabel("Average Score")
plt.show()


# In[8]:


# Block-6: Q5 — Visualization of subject averages and correlations
plt.figure(figsize=(6,4))
sns.barplot(x=subject_means.index, y=subject_means.values, palette='magma')
plt.title("Subject-wise Average Scores")
plt.ylabel("Average Score")
plt.show()

# Correlation heatmap
corr = df[['study_hours', 'math', 'reading', 'writing']].corr()

plt.figure(figsize=(6,4))
sns.heatmap(corr, annot=True, cmap='YlGnBu')
plt.title("Heatmap: Study Hours vs Scores Correlation")
plt.show()


# In[9]:


# Block-7: Summary
print("✅ Student Performance Data Analysis Completed Successfully!")


# In[ ]:




