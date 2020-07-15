## sentiment_rnn
Simple sentiment anylisis with bidirectional recurrent neural network with the use of keras framework. 

## Dataset
Data used for training is from [Sentiment140](http://help.sentiment140.com/for-students). It required small tweaks, done in *data_formatting.py*:
- sentiment normalization
- dropping unneeded collumns 
- size reduction (to make training feasible on my computer)

## Word embedding
The network uses word vectors from [this](https://github.com/facebookresearch/fastText/blob/master/docs/crawl-vectors.md) english model trained on Common Crawl.

## Results
Following results were acquired from the same 100.000 sample from the dataset, with batch size equal to 32 and 80:20 training to validation ratio. Futher training on the whole dataset would allow the model the achieve better results (especially with 64 or more LSTM units), while a bigger batch size would make the training progress more stable.

<img src="/100k%20sample%2C%2032%20batch%20size%20results/accuracy_16_0.0.png" height="50%" width="40%"><img src="/100k%20sample%2C%2032%20batch%20size%20results/accuracy_16_0.2.png" height="50%" width="40%">

<img src="/100k%20sample%2C%2032%20batch%20size%20results/accuracy_32_0.0.png" height="50%" width="40%"><img src="/100k%20sample%2C%2032%20batch%20size%20results/accuracy_32_0.2.png" height="50%" width="40%">

<img src="/100k%20sample%2C%2032%20batch%20size%20results/accuracy_64_0.0.png" height="50%" width="40%"><img src="/100k%20sample%2C%2032%20batch%20size%20results/accuracy_64_0.2.png" height="50%" width="40%">
