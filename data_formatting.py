import pandas as pd

# loading the dataset
dataset = pd.read_csv('dataset.csv', encoding='latin-1', header=None, index_col=1,
                      names=['sentiment', 'id', 'date', 'query', 'user', 'text'])

# normalizing labels
dataset.sentiment = dataset.sentiment.replace({0: 0, 4: 1})

# keeping only the sentiment and text
dataset = dataset[["sentiment", "text"]]

# reducing the amount of samples to make it more computable
dataset = dataset.sample(n=100000)

# saving
dataset.to_csv('dataset_formatted.csv', header=None, index=False)
print(dataset.head())



