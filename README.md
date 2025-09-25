IMDB Dataset: Sentiment Analysis

**IMPORTANT NOTE**: the final version of the code is model2.ipynb, which runs the training loop for the pytotch lstm twice for a total of (6+3) epochs for 88.04 accuracy, but the initial version, model.ipynb, has higher accuracy for the Pytorch LSTM model (87.9) with one training run (6 epochs). The final bar graph comparison of all three models (LR, LSTM, LinearSVC) uses the metrics of the latest training run of the LSTM in model2.


original lstm metrics:

<img width="664" height="204" alt="image" src="https://github.com/user-attachments/assets/ae69ae9d-f9da-4c7c-a8e7-31613692d25d" /> can be checked in model1.ipynb



This project trains sentiment analysis models on the **IMDB movie reviews dataset**.  
The models predicts whether a review is **positive** or **negative**.  
Models Used: Logistic Regression (Scikit-Learn), LSTM (Pytorch), LinearSVC (Scikit-learn)

**Approach** 
Will compare three models, a pytorch LSTM model, Logistic regression model from scikit learn, and linear svc from scikit learn, compare all metrics and plot learning curves and confusion matrixes.

Aim: To compare the metrics of the two different models and plot the confusion matrix and training/test performance plots.

**Logistic Regression**:

This model works well with large datasets and provides solid results.

We start with data preprocessing : plotting sentiment-wise and analysing the output.

<img width="589" height="432" alt="image" src="https://github.com/user-attachments/assets/927747cb-4c0b-48cd-961a-645c208e74f4" />

As we can see, the data is balanced. This makes things a lot simpler for us, it makes training more stable.

The next part of preprocessing is to split the data into training and testing datasets. In my approach, I've used an 80-20 split, where 80 percent is training and 
20 percent is testing. 

We also use stratify to maintain the same proportion of each label in both the training and testing sets.

**Vectorisation**

Models can't understand regular English : they understand numbers. To convert our data to numbers, we use the TF - IDF vectorisor provided by scikit learn, 
emitting common words like 'the', 'is', etc. and setting max features as 5000 to limit vocabulary size.

We then fit_transform for the training set so the model learns the vocabulary and plain transform for the testing set so it doesn't.

**Model Instantiation**

We are using the Logistic Regression model with 500 maximum iterations (to ensure weights are optimised).

We then proceed to train the model on our vectorized training data and predict on our testing data.

**Metrics**

The following metrics have been considered:

Accuracy : Measures accuracy of prediction.

Precision : Out of all the reviews that were identified as positive, how many were actually positive?

Recall : Out of all the positive reviews, how many were correctly identified as positive?

F1 Score : Mean of precision and recall.

  Model                       Accuracy  Precision  Recall   F1
  
  Logistic Regression (Raw)    0.8899   0.883835  0.8978  0.890763
  
  Logistic Regression (Cleaned) 0.8907   0.883716  0.8998  0.891686

Confusion Matrix: shows how many predictions were right, how many were wrong

<img width="569" height="455" alt="image" src="https://github.com/user-attachments/assets/7fa97a22-8dba-4019-b3f2-711ebfab1b2f" />

Learning Curve: graph that shows how the model’s performance changes with the size of the training data.

<img width="576" height="455" alt="image" src="https://github.com/user-attachments/assets/2602a214-856f-48e7-bb4d-3dccf0d28ecc" />



**Model 2: Pytorch LSTM**

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

**Initialisation of Dataloader**

for training, we need to feed our custom data set in batches, along with shuffling so the model does not learn the data order. also, as we will see in the training loop later, we can iterate over these batches efficiently. this is achieved using dataloader.

**Initialisation of Model**

 We will use LSTM (Long Short Term Memory) which is a type of RNN , controls what to remember and what to forget, good for sentiment analysis (better than regular rnn).
 Architecture overview: Embedding layer, LSTM layer, attention masking, fully connnected layer and output (squeezed tensor of shape batch_size)

 **Instantiation of parameters**
 
 increased embedded layer dimensions and hidden layer dimensions from defaults to 200 and 256 respectively for increased accuracy.

 **Loss function and optimizer**
 
 Binary Cross-Entropy with Logits loss function is used because our data is binary classification (positive vs negative sentiment).
 
 Optimizer: Adam
 
 Adam is a popular gradient-based optimizer that adapts learning rates for each parameter.

 Training and Evaluation

The model trains for a maximum of 6 epochs, using early stopping if validation loss doesn’t improve for 2 consecutive epochs.

During each epoch:

The model processes batches of reviews, and weights are updated using Adam to minimize the loss.

Validation is performed to track loss and accuracy on unseen data.

The best model (lowest validation loss) is saved automatically as best_model.pt.

After training, metrics including Accuracy, Precision, Recall, F1-score, and a confusion matrix are computed to evaluate performance.

Confusion Matrix:

<img width="548" height="432" alt="image" src="https://github.com/user-attachments/assets/40231fe2-4002-4b18-a4a8-3aede869c200" />

**IMPORTANT NOTE**: the final version of the code is modelnew.ipynb, but the initial version, model.ipynb, has higher accuracy for this model (87.9) which wasnt replicated in the new notebook.


**Model 3: Linear SVC**

Linear svc, based on the supervised machine learning algorithm used for classification tasks (SVM), is the next model we are using because it is efficient and effective for text classification, perfect for our imdb dataset. We train it and test it on our cleaned data and track the same metrics as before.

Accuracy: 0.8802

Precision: 0.8748028391167192

Recall: 0.8874

F1 Score: 0.881056393963463

**Confusion Matrix:**

<img width="569" height="455" alt="image" src="https://github.com/user-attachments/assets/1a06bf02-49bd-4181-8258-1176dc095c10" />

**Learning Curve:**

<img width="700" height="470" alt="image" src="https://github.com/user-attachments/assets/a42ca3b4-52c6-4e28-a0db-c375b4d4eb31" />

**All 3 Models Metrics Comparison:**

<img width="846" height="528" alt="image" src="https://github.com/user-attachments/assets/f10b98ef-363e-4966-955e-b8a36c29c735" />


From the chart, all three models perform similarly across most metrics, with slight variations. Logistic Regression achieves high accuracy and F1-score, the LSTM shows strong precision but slightly lower recall, and LinearSVC performs consistently across metrics, highlighting that all models are effective for sentiment classification on this dataset.

Ensemble:

To wrap things up, we combine the predictions of the pytorch LSTM and the scikit learn LR  to improve sentiment classification. We collect the probabilities using sigmoid function on the lstm outputs and for the scikit learn LR, we use scikit learn's predict-proba method. we then combine the probabilities from both models by averaging. 

Ensemble Metrics (RNN + LR):

Accuracy: 0.8906

Precision: 0.8737083811710677

Recall: 0.9132

F1 Score: 0.8930177977703893

Confusion Matrix:

<img width="452" height="393" alt="image" src="https://github.com/user-attachments/assets/8f85541d-d697-4dc6-86dd-90c5a831da4a" />



