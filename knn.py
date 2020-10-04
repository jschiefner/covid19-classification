from pandas import read_csv
from sklearn.neighbors import KNeighborsClassifier

max_k = 50
data_split = 1800  # out of 2100 total entries

metadata = read_csv("metadata.csv", usecols=list(range(3, 18)))  # omit File, Age, Sex
covid_labels = metadata["Covid"]
metadata.drop(columns=["Covid"], inplace=True)

training_data = metadata.head(data_split)
training_labels = covid_labels.head(data_split)
test_data = metadata.tail(2100 - data_split)
test_labels = covid_labels.tail(2100 - data_split)

for k in range(1, max_k+1):
    classifier = KNeighborsClassifier(n_neighbors=k)
    classifier.fit(training_data, training_labels)
    score = classifier.score(test_data, test_labels)

    print(f"(k = {k}) Mean accuracy: {score}")
