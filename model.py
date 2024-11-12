import pandas as pd
import sklearn
import csv

# train_data_raw = open('gameratings.csv','r', encoding='utf-8')

# train_data_obj = csv.reader(train_data_raw)

# train_data_df = pd.DataFrame(train_data_obj)

# print(train_data_df.head())
# print(train_data_df.shape )
# print(train_data_df.describe())

# test_data_raw = open('test_esrb-1.csv','r')

# test_data_obj = csv.reader(test_data_raw)

# test_data_df = pd.DataFrame(test_data_obj)

# print(test_data_df.head() )
# print(test_data_df.shape )
# print(test_data_df.describe())

# print(train_data_df.columns)


train_data_df = pd.read_csv('gameratings.csv')

print(train_data_df.head())
print(train_data_df.shape )
print(train_data_df.describe())

test_data_df = pd.read_csv('test_esrb-1.csv')

print(test_data_df.head() )
print(test_data_df.shape )
print(test_data_df.describe())

# Create train and test data
# Remove first column from both data sets

data_train = train_data_df.loc[:, 'console':'violence']

target_train = train_data_df['Target']

data_test = test_data_df.loc[:, 'console':'violence']

target_test = test_data_df['Target']

print(data_train.shape)
print(target_train.shape)
print(data_test.shape)
print(target_test.shape)


# 1) load the dataset and use the KNeighborsClassifier to train and test your model

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()

knn.fit(X=data_train, y = target_train)

# 2) Display all wrong predicted and expected pairs

predicted = knn.predict(data_test)

expected = target_test

wrong = [ (p,e) for (p,e) in zip(predicted, expected) if p != e]

print(wrong)

# 3) produce a csv file of the name of the game and the predicted rating



#new