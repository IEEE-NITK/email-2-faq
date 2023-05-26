# Email-2-FAQ

### Mentors

- Aryan Amit Barsainyan
- Ashish Bharath
- Amandeep Singh

### Members

- Bharadwaja M Chittrapragada
- Karan Kumar Bhagat
- Mohammad Aadil Shabier
- Tejashree Chaudhari

[GitHub Repo](https://github.com/IEEE-NITK/email-2-faq)

#### Acknowledgements

Special thanks to Pranav DV, Rakshit P, Nishant Nayak and the seniors for guiding and reviewing us during the project.

### Abstract

In today's fast-paced business environment, as more and more customers turn to online platforms to interact with businesses, the volume of email inquiries has increased significantly and these queries remain largely unanswered. Generation of FAQs from email repositories can simplify the task. However, manual generation of FAQs by experts is a time consuming and strenous job. An Email-2-FAQ generation system can help address this challenge by automating the process of generating frequently asked questions from email messages. Such a system can extract relevant information from emails, identify common themes and topics, and generate corresponding FAQs that can be easily accessed by customers. Deploying a Deep Learning model for Auto Email-2-FAQ generation would lead to faster response times and reduced wait times for customers, improving the overall user experience.This model is based on the F-Gen framework and it consists of three subsystems wherein email queries are fed as input and the model returns FAQs as outputs.
\end{abstract}

__Index Terms-BERT, FAQ, FG, F-Gen, FGG, GPU, LSTM , QC, RNN, Word2Vec__

### INTRODUCTION

This project implements an approach for generating Frequently Asked Questions (FAQs) from emails using the F-Gen framework. Our approach involves leveraging the strengths of natural language processing and deep learning techniques to automatically extract relevant information from emails and generate corresponding FAQs. Having a brief list of FAQs enables customer service officials to promptly answer them. The F-Gen framework in explained in detail, including its architecture and various components.The system is composed of three interconnected subsystems arranged in a sequential manner: the QC (Query Classifier) subsystem, the FGG (FAQ Group Generator) subsystem, and the FG (FAQ Generator) subsystem. Our implementation demonstrates that the F-Gen framework can generate high-quality FAQs with a high level of accuracy and efficiency.

#### A. Literature Survey

As part of our literature survey, we analyzed several research papers, among which the paper titled "A deep learning based end-to-end system (F-Gen) for automated email FAQ generation "[6] stood out as the most insightful. This paper outlines the F-Gen framework for Email-2-FAQ generation, which comprises three interconnected subsystems, namely QC, FGG, and FG. Our project involved implementing this framework and all its subsystems.Additionally, we relied on the research presented in "Domain Adapted Word Embeddings for Improved Sentiment Classification "[10] to better understand and implement the Query Classifier subsystem, which utilizes a BERT classifier. We also referred to the research paper titled "Siamese Recurrent Architectures for Learning Sentence Similarity " [9] to implement the Siamese LSTM coupled with a Manhattan distance measuring metric for comparing semantic similarities between sentence pairs, as part of the FAQ Group Generator subsystem.Finally, we utilized the RoBERTa summarizer to implement the FAQ Generator subsystem. Our extensive research and implementation of these techniques enabled us to build an efficient and effective email FAQ generation system.

### METHODOLOGY

#### A. Learning Phase

To supplement our research, we completed several courses that provided a comprehensive understanding of the underlying concepts and techniques.Specifically, we completed three courses as part of the Deep Learning Specialization offered by DeepLearning.AI on Coursera .

The first course, "Neural Networks and Deep Learning" [1] provided a foundational understanding of deep learning principles. This included a detailed exploration of various types of neural networks and their respective architectures, as well as the fundamentals of forward propagation, backpropagation, gradient descent algorithms, and loss and cost functions. Second course was: "Improving Deep Neural Networks: Hyperparameter Tuning, Regularization and Optimization"[2] which provided us with practical techniques to improve the performance of deep neural networks. Through this coursework, we learned how to fine-tune hyperparameters, employ regularization techniques, and optimize the architecture of deep neural networks.

Additionally, we completed the fifth course in the deep learning specialization, titled "Sequence Modeling" [3] This coursework provided an in-depth exploration of various types of Recurrent Neural Networks (RNNs) and their implementation. Furthermore, we learned about the backpropagation through time algorithm and its application in sequence modeling. Additionally, we delved into the concept of word embeddings, which is a technique to represent words as numerical vectors using models like Word2Vec and GloVe. In our coursework, we gained a comprehensive understanding of the advantages and disadvantages of using LSTM (Long Short Term Memory) networks versus GRU (Gated Recurrent Units) for text comparison. Our analysis led us to select LSTMs as the preferred choice due to their superior ability to handle long-range dependencies in sequential data. Furthermore, we found that their more complex structure provides additional flexibility and accuracy . By contrast, while GRUs are simpler and computationally efficient, they are not suitable for applications where long-term dependencies are crucial. Overall, our selection of LSTMs was based on careful consideration of the specific requirements of the task, and our findings support their use for effective and robust text comparison.

To get a deeper understanding of transformer based models we completed "Introduction - Hugging Face Course ". [4] Through this course we covered a range of topics related to transformers, including their architecture, pre-training, finetuning, and deployment and also learnt how to use the popular Hugging Face library to work with transformers in real-world applications.

After careful consideration, we opted to utilize the Torch library in Python for our project. This decision was primarily driven by the advantages it offers in terms of debugging and prototyping, owing to the user-friendly PyTorch interface. Additionally, PyTorch integrates seamlessly with other Python libraries, such as NumPy, further facilitating development. Furthermore, its efficient access to state-of-the-art pre-trained language models, such as BERT, makes it an optimal choice for our project's requirements. To learn the PyTorch framework, we enrolled for a series of video lectures titled "PyTorch - Python Deep Learning Neural Network API" [5] offered by deeplizard.

#### B. Technologies used

This project has been coded using Python 3.9 and google colaboratory notebooks. Python modules and libraries including numpy,pandas ,matplotlib, keras, nltk, gensim ,intertool ,tensorflow ,datasets ,transformers, sumy ,typing ,dataclasses,re, email, torch, seaborn, warnings, os, model ,config ,train, dataloader, sklearn,tqdm ,logging ,datetime and time were used for this project.

### IMPLEMENTATION

#### A. Dataset

Our original intention was to acquire email query data from companies, however, due to customer privacy concerns and laws governing the export of customer data, companies were reluctant to share this information with us.

So, we developed a python script that utilizes ChatGPT to auto generate email queries and store them in a text file. The script generates new email queries for a given set of topics by utilizing the OpenAI ChatCompletion API. It creates a dictionary of topics and their corresponding queries, and then uses the ChatCompletion API to generate new queries based on each reference query. This way a total of 10,000 queries were generated .Subsequently, we trained our model on this generated data utilizing a GPU to execute parallel computations, thus achieving improved efficiency and accuracy in our model.

#### B. QC - Subsystem

The Query Classifier Subsystem is implemented using a BERT (Bidirectional Encoder Representations from Transformers) classifier. BERT is a powerful pre-trained language model that captures the contextual meaning of language by training on large amounts of text data. Steps used to build a BERT classifier include:

1) Data Preprocessing: The text data was cleaned, tokenized, and converted into a format that could be input to the BERT model. This involved removing any unnecessary characters, converting the text to lowercase, splitting the text into individual words (tokens), and converting the tokens into numerical vectors using word embeddings.

2) BERT Fine-tuning: The BERT model is first initialized with pre-trained weights, and then the weights are finetuned on the dataset using backpropagation. The fine-tuning process involves minimizing a loss function that measures the difference between the predicted categories and the ground truth labels in the dataset.

3) Label Prediction: Once the BERT model is trained, it can be used to classify new text data into different categories. The text data is first tokenized and then passed through the BERT model to obtain a feature representation of the text. This feature representation is then passed through a classification layer that maps it to a probability distribution over the different categories.

4) Thresholding: A thresholding mechanism is applied to the output probabilities to determine the final category label. Text data is classified into a category if the probability of belonging to that category is higher than a specified threshold.

Therefore, the QC subsystem uses BERT to effectively classify user queries by analyzing the language used, identifying the intent behind the query, and accurately categorizing it into the appropriate category. Additionally, BERT has a large 

![](https://cdn.mathpix.com/cropped/2023_05_24_46958b531822ead168b7g-3.jpg?height=786&width=808&top_left_y=190&top_left_x=192)

Fig. 1. BERT Classifier Architecture

capacity to learn from data, which enables it to continuously improve its accuracy and performance over time.

#### FGG - Subsystem

FAQ Group Generator subsystem selects a subset of relevant FAQs from a larger pool of FAQs and groups them into appropriate clusters based on the user's query.

Various quantitative distance measuring techniques like Cosine similarity, Levenshtein distance, Jaccard, Manhattan distance and transformer based techniques were tested and compared to compute the distance or similarity between two or more data points. Manhattan distance technique had the highest accuracy so it was deployed along with a Siamese LSTM(Lon Short Term Memory) network to compare semantic similarity between two sentences.

The Siamese MaLSTM architecture consists of two identical LSTMs that share the same weights. Each LSTM encodes one of the input sequences into a fixed-length vector representation using Word2Vec. Word2Vec is an unsupervised learning algorithm used to learn distributed representations of words in a text corpus. It represents words as vectors in a highdimensional space, where similar words are represented by vectors that are close together. It uses a neural network to learn these representations by analyzing the context in which words appear in a text corpus. The two vector representations are then fed into a similarity function, such as the Manhattan distance, to compute a similarity score between the two inputs. The Manhattan distance technique measures the distance between two points in a two-dimensional space. The Manhattan distance between two points $(x 1, y 1)$ and (x2, $y 2)$ is given by:

$$
\text { ManhattanDistance }=|x 1-x 2|+|y 1-y 2|
$$

The numerical scale displays the similarity score, based on which email queries are clustered together. High-scoring pairs of sentences are grouped, while low-scoring ones are not. The FGG subsystem is instrumental in refining the scope of email queries for the final FAQ generation subsystem. Ultimately, this plays a pivotal role in enhancing the system's accuracy and efficiency.


| Query 1    | Query 2 | Similarity |
| -------- | ------- |---------|
| Could you please reset my Progress Report?  |  have forgotten my password for progress    |0.8269431  |
| When will the written test and interview be? | When will the written test and interview be    |    0.999     |
| Should I need to fill and submit the registration form?  |  Please reply if any issue in my application    |      0.8504299 |


Fig. 2. Similarity comparison using Siamese LSTM

#### FG Subsystem

A single F-A-Q is generated from multiple queries that had been grouped together by second subsystem. This is carried out using RoBERTa(Robustly Optimized BERT Pretraining Approach) Summarizer, Sumy and Gensim Summarizer.RoBERTa summarizer is a pre-trained transformer based model while Sumy and Gensim are python libraries that include algorithms for automatic text summarization. The summarizer achieves this by encoding the input text into a sequence of vectors, which are then processed through multiple layers of attention-based neural network architecture. These layers enable the model to capture the contextual relationships between words and sentences in the text and extract the most salient information. Once the summarizer has identified the most important information in the queries, it generates a summary of the email-FAQ by selecting and rephrasing the most important sentences. This summary is then used to create a single FAQ that provides clear and concise answers to the queries that were identified in the email.

### Result And Conclusion

Our Email-2-FAQ generation project based on the F-Gen framework provides an efficient and effective way to convert customer queries received via email into a concise and informative FAQ. It is a valuable tool for businesses that want to improve their customer support services. By automating the process of generating FAQs from emails, businesses can save time and resources, while improving customer satisfaction and engagement.

### REFERENCES

[1] Neural Networks and Deep Learning

[2] Improving Deep Neural Networks: Hyperparameter Tuning, Regularization and Optimization

[3] Sequence Models

[4] Introduction - Hugging Face Course [5] PyTorch - Python Deep Learning Neural Network API - deeplizard

[6] A deep learning based end-to-end system (F-Gen) for automated email FAQ generation - ScienceDirect

[7] A Framework for Automatic Generation of FAQs from Email Repositories - IEEE Conference Publication - IEEE Xplore

[8] [1810.04805] BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding (arxiv.org)

[9] Siamese Recurrent Architectures for Learning Sentence Similarity Proceedings of the AAAI Conference on Artificial Intelligence

[10] . [1805.04576] Domain Adapted Word Embeddings for Improved Sentiment Classification (arxiv.org)