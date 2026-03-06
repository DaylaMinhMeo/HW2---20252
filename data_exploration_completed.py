# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # BME-336546-C02-Data exploration and preprocessing
#
# In this tutorial we will start with getting familiar with `PyCharm`, which is one of the most powerful IDEs
# for Python and then we will focus on a core skill of data science: data exploration and preprocessing.

# Before we go on, make sure you have updated our environment `bm-336546` with `tutorial2.yml` file
# as explained in the previous tutorial. Once you are all set we can move on.
# ```shell
# conda env update --name bm-336546 --file tutorial2.yml
# ```

# # Interfacing with Pycharm
# Before we continue with this tutorial, we should all become familiar with one of the best Python IDEs
# which is `PyCharm`. In `PyCharm`, we can debug our code and create projects that contains many `.py` files
# that run with the same virtual environment.
#
# Use Anaconda prompt and `cd` to the correct location. Now you can perform one of the three operations:
# ```shell
# jupytext --to py notebook.ipynb                 # convert notebook.ipynb to a .py file
# jupytext --to notebook notebook.py              # convert notebook.py to an .ipynb file with no outputs
# jupytext --to notebook --execute notebook.py    # convert notebook.py to an .ipynb file and run it
# ```

# Notice that for converting a notebook to `py` file, you can also use `JupyterLab` directly as follows:
# File --> Export Notebook As --> Export Notebook to Executable Script

# # Medical topic
# Diabetes Mellitus affects hundreds of millions of people around the world and can cause many complications
# if not diagnosed early and treated properly. Diabetes can be predicted ahead using some medical explanatory
# variables. In our case we will use the study of Pima Indian population near Phoenix, Arizona.
# All of the patients were women above the age of 21. The population has been under continuous study
# since 1965 by the National Institute of Diabetes and Digestive and Kidney Diseases because of its
# high incidence rate of diabetes. In this tutorial we would only focus on the data exploration part.

# ## Dataset
# The following features have been provided to help us predict whether a person is diabetic or not:
# * Pregnancies: Number of times each woman was pregnant.
# * Glucose: Plasma glucose concentration over 2 hours in an oral glucose tolerance test [mg/dl].
# * BloodPressure: Diastolic blood pressure in [mm/Hg].
# * SkinThickness: Triceps skin fold thickness in [mm].
# * Insulin: Insulin serum over 2 hours in an oral glucose tolerance test [μU/ml].
# * BMI: Body mass index in [kg/m^2].
# * DiabetesPedigreeFunction: A function which scores likelihood of diabetes based on family history.
# * Age: Age in [years].
# * Outcome: Class variable (0 if non-diabetic, 1 if diabetic).
#
# Credit: The data was imported from [Kaggle](https://www.kaggle.com/uciml/pima-indians-diabetes-database).

# The most important thing to do before any machine learning task is to have a look at your dataset.
# After doing that, you should be able to answer the following questions:
# * Does your dataset values make sense to you?
# * How many missing values does your dataset contain?
# * What are the relevant variables are there to your task?
# * Does your data include correlated variables?
# * Is your data "clean"? Should you filter it?
# * Should you reorganize it in order to have an easy access to it later on?

# One of the most widely used packages in Python is called `pandas`.
# This tutorial will cover some of its' well known commands.
# Later on, we will also use what one might consider as the most useful package in the field
# of basic machine learning task which is `scikit-learn`.

# +
# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals

import numpy as np
import pandas as pd
from sklearn import preprocessing
from pathlib import Path


# to make this notebook's output stable across runs
np.random.seed(42)

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for script execution
import matplotlib.pyplot as plt
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

from sklearn.impute import SimpleImputer
from pandas.plotting import scatter_matrix
# -

# `pandas` can load many types of files into some kind of a table that is called a `DataFrame`.
# Every column within this table is called a `Series`.

col_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'Skin Thickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
df = pd.read_csv("data/PimaDiabetes.csv", header=None, names=col_names)

# Here are some of the most useful commands using `pandas`.

df.info()  # general information on data samples amount and their type


df.head()  # print the 5 first observations

# print 5 repetitive random observations using sample and random_state
df.sample(n=5, random_state=1)

df.describe()  # print summary statistics of variables

# ---
# ***Question:*** *In which cases do we only need the **mean** and the **std** for distribution estimation
# and in which cases do we need the whole **summary statistics**?*
#
# **Answer:**
# - Chỉ cần **mean** và **std** khi dữ liệu tuân theo **phân phối chuẩn (Gaussian/Normal distribution)**,
#   vì hai tham số này đủ để mô tả hoàn toàn phân phối.
# - Cần toàn bộ **summary statistics** (min, max, quartiles, median...) khi dữ liệu **không tuân theo
#   phân phối chuẩn** (ví dụ: skewed, bimodal, có outliers), vì mean & std không phản ánh đầy đủ
#   hình dạng phân phối. Các giá trị quartiles giúp phát hiện skewness, outliers và spread thực tế.
#
# ---

# ### Types of subsetting dataframe

df[1:5]  # Notice it does not include the last element

G = df[['Glucose', 'Insulin']]  # double brackets for column access (i.e. G is also a dataframe)

G['Glucose']

df[['Glucose', 'Insulin']][1:5]  # Double brackets for column access and additional outer brackets for observations

df.loc[1:5, ['Glucose', 'Insulin']]  # loc method allows indexing with string within variables (here it does include the last element!).

df.iloc[1:5, 1:3]  # iloc method allows indexing with integers within variables (does not include the last element!)

# `loc` and `iloc` should be used carefully. Basically, `loc` uses strings for columns and labels of rows
# and `iloc` uses indices.

# We would now like to examine the distribution of our data:

# +
axarr = df.hist(bins=50, figsize=(20, 15))  # histograms of dataframe variables
xlbl = ['# of pregnancies [$N.U$]', 'Glucose [$mg/dl$]', 'Blood Pressure [$mm/Hg$]', 'Skin Thickness [$mm$]',
        'Insulin [$\\mu U/ml$]', 'BMI [$Kg/m^2$]', 'DPF [$N.U$]', 'Age [$years$]', 'Diabetes [$N.U$]']

for idx, ax in enumerate(axarr.flatten()):
    ax.set_xlabel(xlbl[idx])
    ax.set_ylabel("Count")

plt.savefig("output_histograms_raw.png", dpi=100, bbox_inches='tight')
plt.close()
print("Saved: output_histograms_raw.png")
# -

print(axarr.shape)

# We can see from those histograms that some of the variables are impossible such as 0 values in
# BMI, insulin, skin thickness and blood pressure.
# First, we'll replace these values with nan.
#
# *All of the operations will be applied on a copy of the dataframe called* `df_nan`.

# +
df_nan = df.copy()
df_nan.loc[:, 'Glucose':'BMI'] = df_nan.loc[:, 'Glucose':'BMI'].replace(0, np.nan)  # replace the non-realistic (0 in our case) values with nan
print("Fraction of replaced (NaN) values per column:")
print(df_nan.isna().sum() / df.shape[0])  # fraction of replaced values
# -

# Most of the missing data are in the variables *insulin* and *skin thickness*. There are several ways
# to handle missing values. Here are some examples:
# * The variable's missing values can be imputed by some value (median for instance).
# * Can replaced by randomly picked values from the rest of the data's distribution.
# * The probability density function can be estimated from the variable's values histogram and
#   missing values would be replaced by sampled values from the pdf.
# * The missing values can be replaced by random values from the variable's values.
# * The total variable can be eliminated when there is no sufficient number of samples.
#
# Here, we will show only median imputation in two different methods within the relevant variables.

print("\n--- Method 1: fillna with median ---")
df_nan.head()

# +
df_nan.loc[:, 'Glucose':'BMI'] = df_nan.loc[:, 'Glucose':'BMI'].fillna(df_nan.median(numeric_only=True))  # method 1
print("After median imputation (method 1):")
print(df_nan.head())
# -

# +
# create df_nan again for second method
df_nan = df.copy()
df_nan.loc[:, 'Glucose':'BMI'] = df_nan.loc[:, 'Glucose':'BMI'].replace(0, np.nan)  # replace the non-realistic results with nan
print("\n--- Method 2: SimpleImputer with median ---")
print("Before imputation:")
print(df_nan.head())
# -

# +
imputer = SimpleImputer(strategy="median")  # method 2, mostly preferred due to it's generalized form
p = imputer.fit(df_nan)
X = imputer.transform(df_nan)
df1 = pd.DataFrame(X, columns=df_nan.columns)  # construct X object as Dataframe
print("\nAfter imputation (method 2):")
print(df1.head())
# -

# +
axarr = df1.hist(bins=50, figsize=(20, 15))  # histograms of dataframe variables
for idx, ax in enumerate(axarr.flatten()):
    ax.set_xlabel(xlbl[idx])
    ax.set_ylabel("Count")
plt.savefig("output_histograms_after_median.png", dpi=100, bbox_inches='tight')
plt.close()
print("\nSaved: output_histograms_after_median.png")
# -

# As expected, the median imputation did not "work well" for *insulin* and *skin thickness*
# due to multiple missing values.
#
# - For the *insulin* values, it might be better to replace them with values drawn from the
#   distribution which has a pretty low variance relative to the mean.
#
# - For the *skin thickness variable*, we can consider elimination of all the variable's values
#   if we assume that it does not affect the outcome.
#
# Either way, it is not right to just "drop" the missing samples of both variables because
# it will significantly reduce the amount of data but it is reasonable to "drop" a feature.

# ***
# Let's move forward and see how to apply a function on a `dataframe` variable.
# In our case we will replace `nan` with random values distributed as the current value distribution.
#
# In order to do so, we will now apply median imputation on all of the variables which are not
# *insulin* or *skin thickness*.
# We will then apply random sampling on *insulin* variable values and "drop" the "skin thickness"
# variable. All of the operations will now be applied directly on the original `dataframe`.

# +
df.loc[:, 'Glucose':'BMI'] = df.loc[:, 'Glucose':'BMI'].replace(0, np.nan)  # replace the non-realistic results with nan
df.loc[:, ['Glucose', 'BloodPressure', 'BMI']] = df.loc[:, ['Glucose', 'BloodPressure', 'BMI']].fillna(df.median(numeric_only=True))  # median imputation
df.drop(columns=['Skin Thickness'], inplace=True)
insulin_hist = df_nan.loc[:, 'Insulin'].dropna()
# -


# **FIX:** Original code used `x == np.nan` which never works because NaN != NaN.
# We use `pd.isna(x)` instead.
def rand_sampling(x, var_hist):
    if pd.isna(x):
        rand_idx = np.random.choice(len(var_hist))
        x = var_hist.iloc[rand_idx]
    return x


# **FIX:** Original code used deprecated `applymap` and did not assign the result back.
# We use `apply` and assign the result to `df['Insulin']`.

# +
df['Insulin'] = df['Insulin'].apply(lambda x: rand_sampling(x, insulin_hist))

xlbl_updated = ['# of pregnancies [$N.U$]', 'Glucose [$mg/dl$]', 'Blood Pressure [$mm/Hg$]',
                'Insulin [$\\mu U/ml$]', 'BMI [$Kg/m^2$]', 'DPF [$N.U$]', 'Age [$years$]',
                'Diabetes [$N.U$]']

axarr = df.hist(bins=50, figsize=(20, 15))
for idx, ax in enumerate(axarr.flatten()):
    if idx < len(xlbl_updated):
        ax.set_xlabel(xlbl_updated[idx])
    ax.set_ylabel("Count")
plt.savefig("output_histograms_after_fix.png", dpi=100, bbox_inches='tight')
plt.close()
print("Saved: output_histograms_after_fix.png")
# -

print(axarr.shape)

# Verify no NaN values remain in Insulin after random sampling fix
print(f"\nNaN count in Insulin after fix: {df['Insulin'].isna().sum()}")

# Pay attention to the missing category and the difference in the insulin figure.
#
# In many tasks, we may find that we need to scale our data. Each task will likely require
# a specific kind of scaling. The scaling process will help us to correctly identify the
# variables that are most important for our task regardless their magnitudes.
#
# Here is an example of scaling your data using the mean and standard deviation.

# +
scaled_features = preprocessing.StandardScaler().fit_transform(df.values)
scaled_features_df = pd.DataFrame(scaled_features, index=df.index, columns=df.columns)

scaled_features_df.hist(bins=50, figsize=(20, 15))
plt.savefig("output_histograms_scaled.png", dpi=100, bbox_inches='tight')
plt.close()
print("Saved: output_histograms_scaled.png")
# -

# ---
# ***Question:*** *Should we scale all of our data as we did?*
#
# **Answer:**
# - **Không nên scale cột Outcome** (biến mục tiêu, 0/1) vì đây là biến categorical dạng binary,
#   việc scale biến này không có ý nghĩa thống kê và sẽ làm mất đi ý nghĩa phân loại của nó.
# - Chỉ nên scale các **biến đầu vào (features)** để đảm bảo chúng có cùng scale,
#   đặc biệt quan trọng cho các thuật toán nhạy cảm với magnitude như SVM, KNN, Logistic Regression.
# - Các thuật toán dựa trên cây (Decision Tree, Random Forest) thường **không cần scaling**.
# - Cách đúng:
#   ```python
#   features = df.drop(columns=['Outcome'])
#   scaled = preprocessing.StandardScaler().fit_transform(features)
#   ```
#
# ---

# Now let's see if we can find correlations among selected variables.
# This can help us later on in choosing the most relevant variables with minimum redundancy.

# +
attributes = ["Age", "BMI", "Glucose", "Pregnancies"]
scatter_matrix(df[attributes], figsize=(12, 8))  # correlation between chosen variables
plt.savefig("output_scatter_matrix.png", dpi=100, bbox_inches='tight')
plt.close()
print("Saved: output_scatter_matrix.png")
# -

# Unfortunately, we can't really find any significant correlation except age and pregnancies
# which was pretty obvious to begin with.
#
# Another important thing that we would like to check within our data is the prevalence.
# Let's check what is the prevalence of diabetes in our dataset.

prc_diab = 100 * df['Outcome'].value_counts(normalize=True)  # normalize=True for percentage
print(r'%.2f%% of the Pima tribe women have diabetes.' % prc_diab[1])

# Sometimes we would like to count values above or below a specific threshold to get a sense
# of the data. Then, we can check if those conditions have any impact on the outcome prevalence,
# or in other words, check if they are *predicative*.

val = df[df['Glucose'] > 150].shape[0]  # how many of the tribe women have glucose values higher than 150
print(r'%d women have glucose values higher than 150 [mg/dl].' % val)

# +
selected_obs = df[(df['Glucose'] > 150) & (df['Insulin'] > 100)]  # Extract patients who have glucose values higher than 150 and insulin values higher than 100
val = 100 * selected_obs['Outcome'].value_counts(normalize=True)[1]  # show how many of the selected patients have diabetes.
print(r'Out of the women who have glucose values higher than 150[mg/dl] and insulin values higher than 100[uU/ml], %.2f%% have diabetes.' % val)
# -

# A significant deviation can be seen in the prevalence once we choose women with high levels
# of insulin and glucose.
#
# The last things that we will see in this tutorial is how to group, sort, filter and plot variables.
# Here are some examples:

print("\n--- Sorted by Pregnancies ---")
print(df.sort_values(by='Pregnancies').head(10))  # Notice the labels

print("\n--- Grouped by Pregnancies (Age statistics) ---")
print(df.groupby('Pregnancies').describe()['Age'])  # for a single variable

Preg_group = df.groupby('Pregnancies')

print(f"\nNumber of women who had 5 pregnancies: {Preg_group.get_group(5)['Age'].shape}")

print(f"\nFiltered groups with more than 24 samples:")
print(Preg_group.filter(lambda x: len(x) > 24).head(10))  # drop groups who have less than 24 samples

# +
df.plot('Age', 'Glucose', kind='scatter')  # scatter plot of two variables
plt.savefig("output_age_glucose_scatter.png", dpi=100, bbox_inches='tight')
plt.close()
print("Saved: output_age_glucose_scatter.png")
# -

print("\n" + "="*60)
print("ALL STEPS COMPLETED SUCCESSFULLY!")
print("="*60)

# #### *This tutorial was written by [Moran Davoodi](mailto:morandavoodi@gmail.com) with the
# assistance of [Yuval Ben Sason](mailto:yuvalbse@gmail.com) & Kevin Kotzen*
