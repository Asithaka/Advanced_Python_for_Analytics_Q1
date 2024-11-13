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

Title = test_data_df['title']

df = pd.DataFrame()
df['title'] = pd.Series(Title)
df['prediction'] = pd.Series(predicted)
print(df[:10])

# df[df['prediction'] == 1]['prediction'] = 'Everyone'
# df[df['prediction'] == 2]['prediction'] = 'Everyone 10+'
# df[df['prediction'] == 3]['prediction'] = 'Mature'
# df[df['prediction'] == 4]['prediction'] = 'Teen'
# print(df[:10])

df1 = df[df['prediction'] == 1]
df2 = df[df['prediction'] == 2]
df3 = df[df['prediction'] == 3]
df4 = df[df['prediction'] == 4]

df1['prediction'] = 'Everyone'
df2['prediction'] = 'Everyone 10+'
df3['prediction'] = 'Mature'
df4['prediction'] = 'Teen'

print(df1[:5])
print(df2[:5])
print(df3[:5])
print(df4[:5])



outfile = open('mypredictions.csv','w')

final_df = pd.concat([df1, df2, df3,df4], ignore_index=False)
sorted_final_df = final_df.sort_index()

list1 = pd.Series(sorted_final_df['title'])
list2 = pd.Series(sorted_final_df['prediction'])

outfile.write('title,prediction\n')

for item in zip(list1,list2):
    l1,l2 = item
    outfile.write(item[0]+','+item[1]+'\n')

outfile.close()


