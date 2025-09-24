IMDB Dataset: Sentiment Analysis

This project trains a sentiment analysis model on the **IMDB movie reviews dataset**.  
The model predicts whether a review is **positive** or **negative**.  
Models Used: Logistic Regression (Scikit-Learn), LSTM (Pytorch)

**Approach**
Aim: To compare the metrics of the two different models and plot the confusion matrix and training/test performance plots.
*Logistic Regression*:
This model works well with large datasets and provides solid results.
We start with data preprocessing : plotting sentiment-wise and analysing the output.

<img width="589" height="432" alt="image" src="https://github.com/user-attachments/assets/927747cb-4c0b-48cd-961a-645c208e74f4" />

As we can see, the data is balanced. This makes things a lot simpler for us, it makes training more stable.
The next part of preprocessing is to split the data into training and testing datasets. In my approach, I've used an 80-20 split, where 80 percent is training and 20 percent is testing. 
We also use stratify to maintain the same proportion of each label in both the training and testing sets.
*Vectorisation*
Models can't understand regular English : they understand numbers. To convert our data to numbers, we use the TF - IDF vectorisor provided by scikit learn, emitting common words like 'the', 'is', etc. and setting max features as 5000 to limit vocabulary size.
We then fit_transform for the training set so the model learns the vocabulary and plain transform for the testing set so it doesn't.
*Model Instantiation*
We are using the Logistic Regression model with 500 maximum iterations (to ensure weights are optimised).
We then proceed to train the model on our vectorized training data and predict on our testing data.
*Metrics*
The following metrics have been considered:
Accuracy : Measures accuracy of prediction.
Precision : Out of all the reviews that were identified as positive, how many were actually positive?
Recall : Out of all the positive reviews, how many were correctly identified as positive?
F1 Score : Mean of precision and recall.

Model 2: Pytorch LSTM
Pipeline: Data cleaning, tokenisation (bert), custom dataset/dataloaders creation, model training + eval
Cleaning:
We follow the same approach as we did for the logical regression model.
Tokenisation (example in code):
We need to split the text into words so we can process this numerically. In the scikit-learn model, this was done automatically for us.




