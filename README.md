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

The next part of preprocessing is to split the data into training and testing datasets. In my approach, I've used an 80-20 split, where 80 percent is training and 
20 percent is testing. 

We also use stratify to maintain the same proportion of each label in both the training and testing sets.

*Vectorisation*

Models can't understand regular English : they understand numbers. To convert our data to numbers, we use the TF - IDF vectorisor provided by scikit learn, 
emitting common words like 'the', 'is', etc. and setting max features as 5000 to limit vocabulary size.

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

<img width="569" height="455" alt="image" src="https://github.com/user-attachments/assets/7fa97a22-8dba-4019-b3f2-711ebfab1b2f" />


Model 2: Pytorch LSTM

Pipeline: Data cleaning, tokenisation (bert), custom dataset/dataloaders creation, model training + eval

Cleaning:

We follow the same approach as we did for the logical regression model.

Tokenisation (example in code):

We need to split the text into words so we can process this numerically. In the scikit-learn model, this was done automatically for us.

Custom Dataset:

Our custom dataset converts raw text into numerical tokens and organizes labels for efficient batching.

reviews: List of movie reviews (text).

labels: List of corresponding sentiment labels (0 for negative, 1 for positive).

tokenizer: A tokenizer (bert-base-uncased) to convert text into token IDs.

max_length: Maximum number of tokens per review. Longer reviews are truncated, shorter reviews are padded.

Inside the constructor:

Reviews and labels are converted to Python lists for easy indexing.

Length (__len__):

Returns the total number of reviews in the dataset.

This allows PyTorch to know how many samples are available

Get item (__getitem__):

Accesses one review and its label by index idx.

Converts the review to a string and the label to a PyTorch tensor.

Tokenizes the review using the tokenizer:

Adds padding to reach max_length.

Truncates reviews that are too long.

Returns PyTorch tensors.

Removes extra dimensions using .squeeze() to ensure the shapes are compatible with the LSTM.

Returns a dictionary with:

input_ids: Numeric IDs of tokens in the review.

attention_mask: Binary mask indicating which tokens are real vs padding.

label: The sentiment label as a tensor

*Initialisation of Dataloader*

for training, we need to feed our custom data set in batches, along with shuffling so the model does not learn the data order. also, as we will see in the training loop later, we can iterate over these batches efficiently. this is achieved using dataloader.

*Initialisation of Model*
 We will use LSTM (Long Short Term Memory) which is a type of RNN , controls what to remember and what to forget, good for sentiment analysis (better than regular rnn).
 Architecture overview: Embedding layer, LSTM layer, attention masking, fully connnected layer and output (squeezed tensor of shape batch_size)

 *Instantiation of parameters*
 increased embedded layer dimensions and hidden layer dimensions from defaults to 200 and 256 respectively for increased accuracy.

 *Loss function and optimizer*
 Binary Cross-Entropy with Logits loss function is used because our data is binary classification (positive vs negative sentiment).
 
 Optimizer: Adam
 
 Adam is a popular gradient-based optimizer that adapts learning rates for each parameter.

 Training and Evaluation

The model trains for a maximum of 6 epochs, using early stopping if validation loss doesn’t improve for 2 consecutive epochs.

During each epoch:

The Lmodel processes batches of reviews, and weights are updated using Adam to minimize the loss.

Validation is performed to track loss and accuracy on unseen data.

The best model (lowest validation loss) is saved automatically as best_model.pt.

After training, metrics including Accuracy, Precision, Recall, F1-score, and a confusion matrix are computed to evaluate performance.


