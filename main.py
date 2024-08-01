import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno
import warnings
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import pickle

warnings.filterwarnings('ignore')

sns.set()
plt.style.use('ggplot')

# Read the CSV file
df = pd.read_csv("breast_cancer.csv")

# Visualize missing data
msno.bar(df, color="red")
plt.savefig("missing_data.png")
plt.show()

# Data preprocessing
df['diagnosis'] = df['diagnosis'].apply(lambda val: 1 if val == 'M' else 0)

# Drop 'id' column
df.drop('id', axis=1, inplace=True)

# Feature selection
corr_matrix = df.corr().abs()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
tri_df = corr_matrix.mask(mask)
to_drop = [x for x in tri_df.columns if any(tri_df[x] > 0.92)]
df = df.drop(to_drop, axis=1)

# Generate and save correlation heatmap
plt.figure(figsize=(20, 12))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", linewidths=1, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.savefig("correlation_heatmap.png")
plt.show()

# Exploratory Data Analysis (EDA)
plt.figure(figsize=(20, 15))
plotnumber = 1
for column in df:
    if plotnumber <= 30:
        ax = plt.subplot(5, 6, plotnumber)
        sns.distplot(df[column])
        plt.xlabel(column)
    plotnumber += 1
plt.tight_layout()
plt.savefig("density_graphs.png")
plt.show()

# Split the data into features and target variable
X = df.drop('diagnosis', axis=1)
y = df['diagnosis']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Scaling data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Apply Logistic Regression
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
log_reg_acc = accuracy_score(y_test, log_reg.predict(X_test))

# Apply KNN
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
knn_acc = accuracy_score(y_test, knn.predict(X_test))

# Apply SVC
svc = SVC(probability=True)
parameters = {
    'gamma': [0.0001, 0.001, 0.01, 0.1],
    'C': [0.01, 0.05, 0.5, 0.1, 1, 10, 15, 20]
}
grid_search = GridSearchCV(svc, parameters)
grid_search.fit(X_train, y_train)
svc = SVC(C=15, gamma=0.01, probability=True)
svc.fit(X_train, y_train)
svc_acc = accuracy_score(y_test, svc.predict(X_test))

# Decision Tree Classifier
dtc = DecisionTreeClassifier(criterion='entropy', max_depth=15, min_samples_leaf=4, min_samples_split=5, splitter='random')
dtc.fit(X_train, y_train)
dtc_acc = accuracy_score(y_test, dtc.predict(X_test))

# Random Forest Classifier
rand_clf = RandomForestClassifier(criterion='entropy', max_depth=10, max_features=0.5, min_samples_leaf=2, min_samples_split=3, n_estimators=130)
rand_clf.fit(X_train, y_train)
rand_clf_acc = accuracy_score(y_test, rand_clf.predict(X_test))

# Gradient Boosting Classifier
gbc = GradientBoostingClassifier(learning_rate=0.1, loss='exponential', n_estimators=180)
gbc.fit(X_train, y_train)
gbc_acc = accuracy_score(y_test, gbc.predict(X_test))

# Save the SVC model
model = svc
pickle.dump(model, open("breast_cancer.pkl", "wb"))

# Plot ROC curves
plt.figure(figsize=(8, 5))
models = [
    {'label': 'LR', 'model': log_reg},
    {'label': 'DT', 'model': dtc},
    {'label': 'SVM', 'model': svc},
    {'label': 'KNN', 'model': knn},
    {'label': 'RF', 'model': rand_clf},
    {'label': 'GBDT', 'model': gbc},
]

for m in models:
    model = m['model']
    y_pred_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X_test)
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
    auc = roc_auc_score(y_test, y_pred_prob)
    plt.plot(fpr, tpr, label='%s - ROC (area = %0.2f)' % (m['label'], auc))

plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([-0.01, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('1 - Specificity (False Positive Rate)', fontsize=12)
plt.ylabel('Sensitivity (True Positive Rate)', fontsize=12)
plt.title('ROC - Breast Cancer Prediction', fontsize=12)
plt.legend(loc="lower right", fontsize=12)
plt.savefig("roc_breast_cancer.jpeg", format='jpeg', dpi=400, bbox_inches='tight')
plt.show()

# Plot performance evaluation
means_accuracy = [
    100 * round(log_reg_acc, 4), 100 * round(dtc_acc, 4), 100 * round(svc_acc, 4),
    100 * round(knn_acc, 4), 100 * round(rand_clf_acc, 4), 100 * round(gbc_acc, 4)
]

means_roc = []
for m in models:
    model = m['model']
    y_pred_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X_test)
    auc = roc_auc_score(y_test, y_pred_prob)
    auc = 100 * round(auc, 4)
    means_roc.append(auc)

print("Accuracy scores: ", means_accuracy)
print("ROC AUC scores: ", means_roc)

# Data to plot
n_groups = 6
means_accuracy = tuple(means_accuracy)
means_roc = tuple(means_roc)

# Create plot
fig, ax = plt.subplots(figsize=(8, 5))
index = np.arange(n_groups)
bar_width = 0.35
opacity = 0.8

rects1 = plt.bar(index, means_accuracy, bar_width, alpha=opacity, color='mediumpurple', label='Accuracy (%)')
rects2 = plt.bar(index + bar_width, means_roc, bar_width, alpha=opacity, color='rebeccapurple', label='ROC (%)')

plt.xlim([-1, 7])
plt.ylim([70, 104])
plt.xlabel('Model', fontsize=12)
plt.ylabel('Score', fontsize=12)
plt.title('Performance Evaluation - Breast Cancer Prediction', fontsize=12)
plt.xticks(index + bar_width / 2, ('LR', 'DT', 'SVM', 'KNN', 'RF', 'GBDT'), rotation=40, ha='center', fontsize=12)
plt.legend(loc="upper right", fontsize=10)
plt.savefig("PE_breast_cancer.jpeg", format='jpeg', dpi=400, bbox_inches='tight')
plt.show()
