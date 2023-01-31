Yandex Cup 2022 - ML Audio Content

Task Description

At first glance, the task of predicting the artist of a track looks strange, since it seems that this information is initially known to us. But upon closer examination, it turns out that not everything is so simple. First, there is the task of separating the performers of the same name. When a new release arrives in our catalog, we need to somehow compare the artist of this release with those that are already in our database, and for artists of the same name there is ambiguity. Secondly, and this is the less obvious part, by predicting the performers by audio, we implicitly get a model that learns the similarity of the performers by sound, and this can also be useful.

Input data format

According to license agreements, we cannot upload the original audio tracks, therefore, as part of this task, we decided to prepare a feature description for each track based on the audio signal. Initially, a random fragment of the track is selected from its central part (70 percent of the track) with a duration of 60 seconds, if the track is shorter than 60 seconds, then the entire track is taken. Further, this fragment is divided into chunks of about 1.5 seconds in size with a step of about 740 milliseconds, and then for each such chunk of the audio signal, a vector of numbers is calculated that describes this chunk, with a size of 512, this is a kind of embedding of this chunk. Thus, for each track, we get a sequence of vectors, or in other words, a 512xT matrix saved to a file as a numpy array. 
The compute_score.py file will help you calculate the metric for your solution. For ease of selection of the validation subset of tracks, on which the metric can be evaluated locally, we divided the training set of tracks into 10 subdirectories, within each of them the track performers also do not overlap.

The evaluation metric is nDCG@100 (Normalized Discounted Cumulative Gain at K, K=100)
Wikipedia link: https://en.wikipedia.org/wiki/Discounted_cumulative_gain#Normalized_DCG