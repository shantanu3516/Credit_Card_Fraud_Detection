# Importing Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score, roc_curve, roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.ensemble import RandomForestClassifier
# Importing Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import  confusion_matrix, classification_report, precision_score, recall_score, f1_score, roc_curve, roc_auc_score, accuracy_score, classification_report, confusion_matrix, auc
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler, PowerTransformer
import streamlit as st
import warnings
warnings.filterwarnings("ignore")
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import base64
from pickle import load
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
from tensorflow.keras.applications import VGG16, VGG19
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout, Concatenate
from tensorflow.keras.optimizers import Adam
import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import  confusion_matrix, classification_report, precision_score, recall_score, f1_score, roc_curve, roc_auc_score, accuracy_score, classification_report, confusion_matrix, auc

# Load the dataset
df = pd.read_csv('creditcard.csv')

# Print description of the initial data and shape
print("Initial data set:")
print(df.head())
print("\nData description:")
print(df.describe())
print("\nShape of the dataframe:", df.shape)
print("The dataset consists of 284,807 rows and 31 columns.\nThere is no zero value in the data.")

# Check missing values
print("\nMissing values:", df.isnull().values.sum())
print("The number of missing values in each column:")
print(df.isnull().sum())

# Percentage of null values
percent_missing = (df.isnull().sum().sort_values(ascending=False) / len(df)) * 100
print("\nPercentage of null values:")
print(percent_missing)

# Check if there are any duplicate rows
print("\nDuplicate rows:", df.duplicated(keep=False).sum())

# Delete duplicate rows
df = df.drop_duplicates()
print("\nDeleting duplicate rows was successful. This is a new data set:")
print(df.head())

# Separate legitimate and fraudulent transactions
fraud = df[df.Class == 1]
valid = df[df.Class == 0]

outlier_percentage = (df.Class.value_counts()[1] / df.Class.value_counts()[0]) * 100

print("\nFraud Cases:", len(fraud))
print("Valid Cases:", len(valid))
print("Compare the values for both transactions:")
print(df.groupby('Class').mean())
print("Fraudulent transactions are: %.3f%%" % outlier_percentage)

# Plotting the distribution of the variables (skewness) of all the columns
def skewness(data, cols):
    k = 0
    plt.figure(figsize=(17, 28))
    for col in cols:
        k = k + 1
        plt.subplot(6, 5, k)
        sns.distplot(data[col])
        plt.title(col + ' ' + str(data[col].skew()))
    plt.show()

# Plot relation with different scale
def plot_relation(df1, df2):
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    ax[0].scatter(df1['Time'], df1['Amount'], color='red', marker='*', label='Fraudulent')
    ax[0].set_title('Time vs Amount')
    ax[0].legend(bbox_to_anchor=(0.25, 1.15))
    ax[1].scatter(df2['Time'], df2['Amount'], color='green', marker='.', label='Non Fraudulent')
    ax[1].set_title('Time vs Amount')
    ax[1].legend(bbox_to_anchor=(0.3, 1.15))
    plt.show()

# Plotting the distribution of the variables (skewness) of all the columns
cols = df.columns[:-1]
skewness(df, cols)

# Plot relation with different scale
plot_relation(fraud, valid)

# Putting feature variables into X
X = df.drop(['Class'], axis=1)

# Putting target variable to y
y = df['Class']

# Splitting data into train and test set 80:20
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=42)

print("\nX_train:", X_train.shape)
print("y_train:", y_train.shape)
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)

# Instantiate the Scaler
scaler = StandardScaler()

# Fit the data into scaler and transform
X_train['Amount'] = scaler.fit_transform(X_train[['Amount']])

# Transform the test set
X_test['Amount'] = scaler.transform(X_test[['Amount']])

# Checking the Skewness
cols = X_train.columns
skewness(X_train, cols)

# Instantiate the powertransformer
pt = PowerTransformer(method='yeo-johnson', standardize=True, copy=False)

# Fit and transform the PT on training data
X_train[cols] = pt.fit_transform(X_train)

# Transform the test set
X_test[cols] = pt.transform(X_test)

# Plotting the distribution of the variables (skewness) of all the columns after transformation
skewness(X_train, cols)

# Function to visualize confusion matrix
def visualize_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Oranges',
                xticklabels=['No Credit Card Fraud Detection', 'Credit Card Fraud Detection'],
                yticklabels=['No Credit Card Fraud Detection', 'Credit Card Fraud Detection'])
    plt.title('Accuracy: {0:.4f}'.format(accuracy_score(y_test, y_pred)))
    plt.ylabel('True Values')
    plt.xlabel('Predicted Values')
    plt.show()
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

# Function to plot ROC curve
def ROC_AUC(Y, Y_prob):
    fpr, tpr, threshold = roc_curve(Y, Y_prob)
    model_auc = roc_auc_score(Y, Y_prob)
    plt.figure(figsize=(16, 9))
    plt.plot([0, 1], [0, 1], linestyle='--', label='No Skill')
    plt.plot(fpr, tpr, marker='.', label='Model - AUC=%.3f' % (model_auc))
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.show()

# Logistic Regression
LR_model = LogisticRegression(random_state=0)
LR_model.fit(X_train, y_train)
y_train_pred = LR_model.predict(X_train)
y_test_pred = LR_model.predict(X_test)
acc1 = accuracy_score(y_test, y_test_pred)

print("\nLogistic Regression Results:")
print("Training set:")
print("Recall score: %0.4f" % recall_score(y_train, y_train_pred))
print("Precision score: %0.4f" % precision_score(y_train, y_train_pred))
print("F1-Score: %0.4f" % f1_score(y_train, y_train_pred))
print("Accuracy score: %0.4f" % accuracy_score(y_train, y_train_pred))
print("AUC: %0.4f" % roc_auc_score(y_train, y_train_pred))

visualize_confusion_matrix(y_train, y_train_pred)
ROC_AUC(y_train, y_train_pred)

print("\nTesting set:")
print("Recall score: %0.4f" % recall_score(y_test, y_test_pred))
print("Precision score: %0.4f" % precision_score(y_test, y_test_pred))
print("F1-Score: %0.4f" % f1_score(y_test, y_test_pred))
print("Accuracy score: %0.4f" % accuracy_score(y_test, y_test_pred))
print("AUC: %0.4f" % roc_auc_score(y_test, y_test_pred))

visualize_confusion_matrix(y_test, y_test_pred)
ROC_AUC(y_test, y_test_pred)

# Naive Bayes
NB_model = GaussianNB()
NB_model.fit(X_train, y_train)
y_train_pred = NB_model.predict(X_train)
y_test_pred = NB_model.predict(X_test)
acc2 = accuracy_score(y_test, y_test_pred)

print("\nNaive Bayes Results:")
print("Training set:")
print("Recall score: %0.4f" % recall_score(y_train, y_train_pred))
print("Precision score: %0.4f" % precision_score(y_train, y_train_pred))
print("F1-Score: %0.4f" % f1_score(y_train, y_train_pred))
print("Accuracy score: %0.4f" % accuracy_score(y_train, y_train_pred))
print("AUC: %0.4f" % roc_auc_score(y_train, y_train_pred))

visualize_confusion_matrix(y_train, y_train_pred)
ROC_AUC(y_train, y_train_pred)

print("\nTesting set:")
print("Recall score: %0.4f" % recall_score(y_test, y_test_pred))
print("Precision score: %0.4f" % precision_score(y_test, y_test_pred))
print("F1-Score: %0.4f" % f1_score(y_test, y_test_pred))
print("Accuracy score: %0.4f" % accuracy_score(y_test, y_test_pred))
print("AUC: %0.4f" % roc_auc_score(y_test, y_test_pred))

visualize_confusion_matrix(y_test, y_test_pred)
ROC_AUC(y_test, y_test_pred)

# Decision Tree
DTR_model = DecisionTreeClassifier(criterion='entropy', random_state=0)
DTR_model.fit(X_train, y_train)
y_train_pred = DTR_model.predict(X_train)
y_test_pred = DTR_model.predict(X_test)
acc4 = accuracy_score(y_test, y_test_pred)

print("\nDecision Tree Results:")
print("Training set:")
print("Recall score: %0.4f" % recall_score(y_train, y_train_pred))
print("Precision score: %0.4f" % precision_score(y_train, y_train_pred))
print("F1-Score: %0.4f" % f1_score(y_train, y_train_pred))
print("Accuracy score: %0.4f" % accuracy_score(y_train, y_train_pred))
print("AUC: %0.4f" % roc_auc_score(y_train, y_train_pred))

visualize_confusion_matrix(y_train, y_train_pred)
ROC_AUC(y_train, y_train_pred)

print("\nTesting set:")
print("Recall score: %0.4f" % recall_score(y_test, y_test_pred))
print("Precision score: %0.4f" % precision_score(y_test, y_test_pred))
print("F1-Score: %0.4f" % f1_score(y_test, y_test_pred))
print("Accuracy score: %0.4f" % accuracy_score(y_test, y_test_pred))
print("AUC: %0.4f" % roc_auc_score(y_test, y_test_pred))

visualize_confusion_matrix(y_test, y_test_pred)
ROC_AUC(y_test, y_test_pred)

# Compare the accuracy of the models on the Testing set
mylist = [acc1, acc2, acc4]
mylist2 = ["Logistic Regression", "Naive Bayes", "Decision Tree"]

plt.figure(figsize=(22, 10))
sns.set_style("darkgrid")
ax = sns.barplot(x=mylist2, y=mylist, palette="Oranges", saturation=1.5)
plt.xlabel("Classification Models", fontsize=20)
plt.ylabel("Accuracy", fontsize=20)
plt.title("Accuracy of different Classification Models", fontsize=20)
plt.xticks(fontsize=11, horizontalalignment='center', rotation=0)
plt.yticks(fontsize=13)
for p in ax.patches:
    width, height = p.get_width(), p.get_height()
    x, y = p.get_xy()
    ax.annotate(f'{height:.2%}', (x + width / 2, y + height * 1.02), ha='center', fontsize='x-large')
plt.show()

# ROC Curve and Area Under the Curve
y_pred_logistic = LR_model.predict_proba(X_test)[:, 1]
logistic_fpr, logistic_tpr, threshold = roc_curve(y_test, y_pred_logistic)
auc_logistic = auc(logistic_fpr, logistic_tpr)

y_pred_nb = NB_model.predict_proba(X_test)[:, 1]
nb_fpr, nb_tpr, threshold = roc_curve(y_test, y_pred_nb)
auc_nb = auc(nb_fpr, nb_tpr)

y_pred_dtr = DTR_model.predict_proba(X_test)[:, 1]
dtr_fpr, dtr_tpr, threshold = roc_curve(y_test, y_pred_dtr)
auc_dtr = auc(dtr_fpr, dtr_tpr)

plt.figure(figsize=(10, 8), dpi=100)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(logistic_fpr, logistic_tpr, label='Logistic Regression (auc = %0.4f)' % auc_logistic)
plt.plot(nb_fpr, nb_tpr, label='Naive Bayes (auc = %0.4f)' % auc_nb)
plt.plot(dtr_fpr, dtr_tpr, label='Decision Tree (auc = %0.4f)' % auc_dtr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='best')
plt.show()

# Prediction using Random Forest
# separate legitimate and fraudulent transactions
legit = df[df.Class == 0]
fraud = df[df.Class == 1]

# undersample legitimate transactions to balance the classes
legit_sample = legit.sample(n=len(fraud), random_state=2)
data = pd.concat([legit_sample, fraud], axis=0)

# split data into training and testing sets
X = data.drop(columns="Class", axis=1)
y = data["Class"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2)
    
from tensorflow.keras.applications import VGG16, VGG19
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense, Dropout, Concatenate
from tensorflow.keras.optimizers import Adam

# Inputs
input_vgg16 = Input(shape=(224, 224, 3))
input_vgg19 = Input(shape=(224, 224, 3))

# Load base models without top
vgg16_base = VGG16(weights='imagenet', include_top=False)
vgg19_base = VGG19(weights='imagenet', include_top=False)

# Freeze layers
vgg16_base.trainable = False
vgg19_base.trainable = False

# Connect your input layers manually
vgg16_out = vgg16_base(input_vgg16)
vgg19_out = vgg19_base(input_vgg19)

# Global average pooling
vgg16_out = GlobalAveragePooling2D()(vgg16_out)
vgg19_out = GlobalAveragePooling2D()(vgg19_out)

# Concatenate
merged = Concatenate()([vgg16_out, vgg19_out])

# Dense layers
x = Dense(1024, activation='relu')(merged)
x = Dropout(0.5)(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(256, activation='relu')(x)

# Output
output = Dense(2, activation='softmax')(x)

# Hybrid model
hybrid_model = Model(inputs=[input_vgg16, input_vgg19], outputs=output)

# Compile
hybrid_model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

hybrid_model.summary()

# train Random Forest model
hybrid_model = RandomForestClassifier(n_estimators=100, random_state=42)
hybrid_model.fit(X_train, y_train)

# evaluate model performance
train_acc = accuracy_score(hybrid_model.predict(X_train), y_train)
test_acc = accuracy_score(hybrid_model.predict(X_test), y_test)

print("\n Hybird VGG16 and VGG19 Results:")
print("Training Accuracy:", train_acc)
print("Testing Accuracy:", test_acc)

# Manual transaction verification
input_df = input("Enter the following features to check if the transaction is legitimate or fraudulent (comma-separated): ")
input_df_lst = list(map(float, input_df.split(',')))
features = np.array(input_df_lst, dtype=np.float64)

# make prediction
prediction = hybrid_model.predict(features.reshape(1, -1))

# display result
if prediction[0] == 0:
    print("Legitimate transaction")
else:
    print("Fraudulent transaction")