## sentiment_rnn
Simple sentiment anylisis with bidirectional recurrent neural network with the use of keras framework. 

## Dataset
Data used for training is from [Sentiment140](http://help.sentiment140.com/for-students). It required small tweaks, done in *data_formatting.py*:
- sentiment normalization
- dropping unneeded collumns 
- size reduction (to make training feasible on my computer)

## Word embedding
The network uses word vectors from [this](https://github.com/facebookresearch/fastText/blob/master/docs/crawl-vectors.md) english model trained on Common Crawl.
