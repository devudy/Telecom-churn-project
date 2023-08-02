import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from keras.models import Sequential
from keras.layers import Dense
from enum import auto
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier
from matplotlib import pyplot
from sklearn.linear_model import LogisticRegression
import missingno as msno
from sklearn.metrics import roc_auc_score
from sklearn import metrics
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.ensemble import RandomForestClassifier
import lightgbm as 1gb
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy score, precision_score, recall_score, fl_score, classification_report,confusion_matrix, Confu
from sklearn.model_ selection import train_test split, cross_val_score, StratifiedKFold, GridSearchCv
from sklearn.metrics import roc_curve
from matplotlib import pyplot

tel = pd.read_csv('Telecom customer churn.csv')
data = tel.copy()

data.describe()

sns.set_theme(color_codes=True)
ax = sns.regplot(x="income", y="rev_Mean", data=data)

# totcalls vs income
sns.set_theme(color_codes=True)
ax = sns.regplot(x="income", y="totcalls", data=data)

# income vs prizm
sns.countplot(x= "prizm_social_one", hue="income", data=data);
data.groupby( income" )["prizm_social_one"].value_counts{normalize=True).unstack(fill_value=0)

# totcalls vs income
sns.set_theme(color_codes=True)
ax = sns.regplot(x="income", y="churn", data=data)

sns.countplot(x= "income", hue="churn", data=data);
data.groupby( income" )["churn”].value_counts(normalize=True).unstack(fill_value=0)

# refurb_new vs churn
sns.countplot(x= "refurb_new", hue="churn", data=data);
data. groupby('refurb_new')["churn"].value_counts(normalize=True).unstack(fill_value=0)

#churn vs newcell
sns.countplot(x= "new_cell”, hue=“churn", data=data);
data.groupby('new_cell')["churn"].value_counts({normalize=True).unstack(fill_value=0)

# crclscod vs churn
sns.countplot(x= "crclscod”, hue="churn", data=data);
data.groupby('crclscod’ )["churn”].value_counts{normalize=True).unstack(fill_value=0)

# asl_flag vs churn
sns.countplot(x= "asl_flag", hue="churn", data=data);
data.groupby( asl _flag')["churn"].value_counts(normalize=True).unstack(fill_value=0)

# prizm_social_one vs churn
sns.countplot(x= "prizm_social_one", hue="churn", data=data);
data.groupby('prizm_social one')["churn"].value_counts(normalize=True).unstack(fill_value=0)

# ownrent vs churn
sns.countplot(x= "ownrent", hue="churn", data=data);
data.groupby('ownrent')["churn*].value_counts(normalize=True).unstack(fill_value=0)


# dwellingtype vs churn
sns.countplot(x= "dwlltype", hue="churn", data=data);
data.groupby('dwlltype')["churn"].value_counts(normalize=True).unstack(fill_value=0)

# martialstatus vs churn
sns.countplot(x= "marital", hue="churn”, data=data);
data.groupby('marital')["churn"].value_counts(normalize=True).unstack(fill value=0)

sns.boxplot(x = data[‘actvsubs'], palette = 'Set3')

#infobase vs churn
sns.countplot(x= "infobase", hue="churn", data=data);
data.groupby('infobase')["churn"].value_counts({normalize=True).unstack(fill_value=0)

#Household status indicator vs churn
sns.countplot(x= "HHstatin", hue="churn", data=data);
data.groupby('HHstatin')["churn"].value_counts({normalize=True).unstack(fill_value=0)

#dwelling size vs churn
sns.countplot(x= "dwllsize", hue="churn", data=data);
data.groupby('dwllsize')["churn"].value_counts(normalize=True).unstack(fill_value=0)

#ethnicity vs churn
sns.countplot(x= "ethnic", hue="churn", data=data);
data.groupby( ‘ethnic’ )[“churn”].value_counts(normalize=True).unstack(fill_value=@)

# credit card status vs churn
sns.countplot(x= "creditcd", hue="churn", data=data);
data.groupby('creditcd')["churn"].value_counts{normalize=True).unstack(fill_value=0)

#counting people who churn and who dont

stay = data[(data['churn'] ==0) ].count()[1]
churn = data[(data[ ‘churn'] ==1) J].count(}[1]
print (“people who stay: "+ str(stay))

print ("people who churn: "+ str(churn))

label = ['Churn’, 'Not Churn']
size=[ (churn/(stay+churn))*1e0, (stay/(stay+churn))*16e]
explode=(0,0.1)
fig = plt.figure(figsize =(10, 7))
plt.pie(size, labels = label,explode=explode,autopct="%1.1f¥%%"')
fig.legend(label,

title ="Churn ratio”,

loc ="center left",

bbox_to_anchor =(1, 0, 0.5, 1))

#Now we start dealing with the missing values

#columns with missing values

missing = data.isnull().sum().sort_values(ascending = False).head(44)

miss_percentage = (missing/len(data))*100

# Percentage of missing values

pd.DataFrame({'No. missing values’: missing, '% of missing data': miss_percentage.values})

data.corr

data.drop(["Customer_ID"], axis = 1, inplace=True)
data.shape

# We dropped the columns that seem to have no significant contribution to the model.
data.drop([ 'prizm_social one’, 'infobase','crclscod’],axis=1,inplace=True)
data.drop(["area"], axis = 1, inplace=True)

data.drop(["ethnic"], axis = 1, inplace=True)

data.drop(["forgntvl"], axis = 1, inplace=True)

data.drop(["asl_flag"], axis = 1, inplace=True)

data[ "numbcars ]=data[ 'numbcars'].fillna{data[ 'numbcars'].mean())

data[ "dwllsize' ]=data[ 'dwllsize'].fillna( UNKW")

data[ 'ownrent' ]=data[ 'ownrent'].fillna( 'UNKW")

data[ "dwlltype' ]=data[ 'dwlltype'].fillna( UNKKW")

data[ "lor ]=data['lor"].fillna( 'UNKW")

data[ "truck']=data[ ‘truck'].fillna(®)

data[ "income ' ]=data[ 'income’].fillna(data[ 'income'].mean())

data[ ‘adults’ ]=data[ 'adults'].fillna(data[ 'adults'].mean())

data[ "hnd_webcap' ]=data[ 'hnd_webcap'].fillna("UNKW')

data[ "avgéqty' ]=data[ ‘avgéqty’].fillna(data[ 'avgéqty'].mean())

data[ "avgérev' ]=data[ 'avgérev'].fillna(data[ 'avgérev'].mean())

data[ "avgémou' ]=data[ ‘avgémou'].fillna(data[ 'avgémou'].mean())

data[ ‘change_mou' ]=data[ 'change_mou'].fillna(data[ 'change_mou"].mean(})
data[ change_rev' ]=data[ 'change_rev'].fillna(data[ 'change_rev'].mean(})
data[ ‘rev_Mean' ]=data[ 'rev_Mean'].fillna(data['rev_Mean'].mean())

data[ 'totmrc_Mean']=data[ 'totmrc_Mean'].fillna(data['totmrc_Mean'].mean())
data[ 'da_Mean']=data[ 'da_Mean'].fillna(data[ ‘da_Mean'].mean())

data[ "ovrmou_Mean' ]=data[ 'ovrmou_Mean'].fillna(data[ 'ovrmou_Mean'].mean(})
data[ 'ovrrev_Mean']=data[ 'ovrrev_Mean'].fillna(data['ovrrev_Mean'].mean())
data[ "vceovr_Mean' ]=data[ 'vceovr_Mean'].fillna(data[ 'vceovr_Mean'].mean(})
data[ "datovr_Mean']=data[ 'datovr_Mean'].fillna(data[ 'datovr_Mean'].mean()})
data[ ‘roam_Mean']=data[ ‘roam_Mean'].fillna(data[‘'roam_Mean'].mean())

data[ "‘mou_Mean' ]=data[ 'mou_Mean'].fillna(data[ 'mou_Mean'].mean())

msno.matrix(data);

data.dropna(inplace=True)
data.shape

obj_col = data.select_dtypes(include = 'object').columns
# obj_col['churn’] = data[’'churn']
obj_col

Index(['new_cell', 'dualband', 'refurb_new', 'hnd_webcap', 'ownrent', ‘lor’,
'dwlltype’, ‘marital’, ‘'HHstatin', ‘'dwllsize', ‘'kide 2', 'kid3_5°,
'kid6_10', 'kid1l_15', ‘kid16_17', 'creditcd'],

dtype="object')

for i in num_ft:
f_sqrt= (lambda x: np.sqrt(abs(x)) if (x>=1) or (x<=-1) else x)
data[i] = data[i].apply(f_ sqrt)

# Box plots of all numerical variables
fig, ax = plt.subplots(15, 4, figsize = (20, 58))
ax = ax.flatten()
for i, c¢ in enumerate (num_ft):

sns.boxplot(x = data[c], ax = ax[i], palette = 'Set3')
# plt.suptitle('Box Plot’, fontsize = 25)
fig.tight_layout()

#Now we treat the outliers in our numerical variables

def detect_outliers(data,features):
outlier_index = []

for c in features:
# 1st quartile
Q1 = np.percentile(data[c], 25)
# 3rd quartile
Q3 = np.percentile(data[c], 75)
# IQR
IQR = Q3 - Q1
# Outlier step
outlier_step = IQR * 1.5
outlier list_col = data[(data[c] < Q1 - outlier_step) | (data[c] >» Q3 + outlier_step)].index
outlier_index.extend(outlier_list col)

outlier_index = Counter(outlier_index)
multiple outliers = list(i for i, v in outlier_index.items() if v > 2)

return outlier_index

sns.boxplot(x = data['uniqsubs'], palette = *Set3')

num_col = data.select_dtypes(include = 'number').columns
num_cols = list{num_col)
num_cols.remove(' churn’)

# one-hot encoding for variables with more than 2 categories
datac = data.copy()
datac = pd.get_dummies(datac, drop_first=True, columns = obj_col, prefix = obj_col}
datac

#corr
c = data.corr()['churn'].abs()
sc = c.sort_values()
sc


dict(sc.tail(40))
b = a.keys()
print(sorted(b))

# Get Correlation of "churn" with other variables:
plt.figure(figsize=(15,8))

datac[b].corr()[ 'churn’].sort_values(ascending = False).plot(kind="'bar")

#modelling our dataset

# generate dataset
X = datac.drop('churn', axis=1)
y = datac['churn']

sc = StandardScaler()
X_sc = sc.fit_transform(X)
X = pd.DataFrame(X_sc, columns = X.columns)

X_num = X[num_cols]
X_obj = X.drop(num_cols, axis=1)

transformer = FunctionTransformer(np.loglp)

X_log = transformer.fit_transform{(X_num)

X_logged_df = pd.DataFrame(X_log, columns = X_num.columns)
X_logged_df.shape

X_logged_df2 = X_logged_df.fillna(®)
X_logged_df2.shape

Xs = [X_ obj, X_logged_df2]
X_final = pd.concat(Xs, axis=1)
X_final.shape

# define feature selection
fs = SelectKBest(score_func=f classif, k='all')

# apply feature selection

# X_final = X_final.astype('long')
X_selected = fs.fit_transform(X_final, y)
print(X_selected.shape)

# dependent and independent variables were determined.
# X = datac.drop('churn', axis=1)
# y = data 'churn’]

X_train, X_ test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random state=42)
print ("X_train",len(X_train))
print ("X_test",len(X_test))
print("y_train",len(y_train))
print("y_test",len(y_test))

#Defining the modelling function
def modeling(alg, alg_name, params={}):
model = alg(**params) #Instantiating the algorithm class and unpacking parameters if any
model. fit(X_train, y_train)
y_pred = model.predict(X test)

#Performance evaluation
def print_scores(alg, y_true, y_pred):
print(alg name)
acc_score = accuracy_score(y_true, y_pred)
print("accuracy: ",acc_score)
pre_score = precision_score(y_true, y pred)
print("precision: ",pre_score)
rec_score = recall_score(y_true, y_pred)
print("recall: ",rec_score)
f_score = fi1_score(y_true, y_ pred, average='weighted')
print("f1_score: ",f_score)
print_scores(alg, y_test, y_pred)

cm = confusion_matrix(y_ test, y_pred)
cmd_obj = ConfusionMatrixDisplay(cm, display_labels=['churn’, 'notChurn'])
cmd_obj.plot()

cmd_obj.ax_.set(
title="Sklearn Confusion Matrix’,
xlabel="Predicted Churn',
ylabel="Actual Churn")

return model


def showroc(model, model name):
model. fit(X_train,y_train)
md_probs = model.predict_proba(X_test)
md_probs = md_probs[:,1]
md_auc = roc_auc_score(y_test, md_probs)
md_fpr, md_tpr, _ = roc_curve(y_test, md_probs)
plt.plot(md_fpr, md_tpr, marker='.', label=model_name)
print{md_auc)

GNB_model = modeling(GaussianNB, ‘Gaussian Naive Bayes')
showroc(GNB_model, "Gaussian Naive Bayes”)

Ada_boost_model = modeling(AdaBoostClassifier, 'Ada Boost')
showroc(Ada_boost_model, "Ada Boost")

a={'C":1.0, 'class_weight':None, 'dual':False, 'fit_intercept':True,
‘intercept_scaling':1, 'l1_ratio®:None,
‘multi_class':'auto', 'n_jobs':None, 'penalty':'l2‘,
‘random_state':None, 'solver':'lbfgs', 'tol':0.0881, ‘verbose':0,
‘warm_start':False, 'max_iter':1000}

Log_reg_model = modeling(LogisticRegression, ‘Logistic Regression’,a)
showroc(Log_reg_model, "Logistic Regression")

j={'max_depth':2, 'n_estimators':100, 'random_state':@}
GB model = modeling(GradientBoostingClassifier, 'Gradient Boosting',i)
showroc(GB_model, "Graldent Boosting")

g={"'base_score':0.5, 'colsample_bylevel':1, 'colsample_bynode':1, 'colsample_bytree':1,'gamma’:®, 'importance_type':'gain','gpu_id':@,
XGB_model = modeling(XGBClassifier, 'XG Boosting',g)
showroc(XGB_model, "XG Boosting")

# Running RandomForestClassifier model
v={"bootstrap':True, 'ccp_alpha':0.0, 'class_weight':None,
‘criterion’: 'gini’, ‘max_depth’:None, 'max_features':'auto’,
'max_leaf_nodes’:None, 'max_samples"':None,
‘min_impurity_decrease':@.@,
‘min_samples_leaf':1, 'min_samples_split':2,
‘min_weight_fraction_leaf':0.0, 'n_estimators’:1ee,
'n_jobs':-1, 'oob_score':False, 'random_state':123, 'verbose':0,
‘warm_start’ :False}
RF_model = modeling(RandomForestClassifier, 'Random Forest',v)
showroc(RF_model, "Random Forest")

#Decision tree
dt_model = modeling(DecisionTreeClassifier, "Decision Tree Classification")
showroc(dt_model, "Decision Tree Classification)

# define the model
model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activatilon='relu', kernel_initializer="he_normal'}))
model.add(Dense(256, activation='relu’, kernel_initializer='he_normal'}))
model.add(Dense(512, activation='relu’, kernel_initializer="he_normal’})
model.add(Dense(1, activation='sigmoid'}})

# compile the keras model
model.compile(loss="binary_crossentropy', optimizer='adam', metrics=['accuracy’])

# fit the keras model on the dataset
fittedm=model.fit(X_train, y train, epochs=1@@, batch_size=16, verbose=2,validation_data=(X_ test, y_test))

# evaluate the keras model
_, accuracy = model.evaluate(X_ test, y_test, verbose=8)
print('Accuracy: %.2f' % (accuracy*100))
train_acc = model.evaluate(X train, y train, verbose=0)
test_acc = model.evaluate(X_test, y_test, verbose=0)
print('Training accuracy {:.4f}',train_acc)
print( Testing accuracy {:.4f}',test_acc)

# plot loss during training
pyplot.subplot(211)
pyplot.title('Loss')
pyplot.plot(fittedm.history['loss'], label='train')
pyplot.plot(fittedm.history[ 'val_loss'], label="test')
pyplot.legend()

# plot accuracy during training
pyplot.subplot(212)
pyplot.title('Accuracy')
pyplot.plot(fittedm.history['accuracy'], label='train')
pyplot.plot(fittedm.history[ 'val_accuracy'], label="test')
pyplot.legend()
pyplot.show()

