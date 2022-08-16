# ADS 509: Amazon Customer Reviews Sentiment Analysis Based on Streaming Devices


# Collaborators 

- [Jimmy Nguyen](https://github.com/jimmy-nguyen-data-science)
- [Juliet Sieland-Harris](https://github.com/JSielandHarris)

# Installation


To clone this repository onto your device, use the commands below:

	1. git init
	2. git clone https://github.com/jimmy-nguyen-data-science/Capstone-Project-Amazon-Product-Reviews.git

### Dependencies

To download the required packages as a local environment, please do the following:

 	1. pip install -r requirements.txt


# Background and Purpose 

As society continues to progress digitally, consumers are increasingly relying on e-commerce shopping websites to make
purchases. The convenience of ordering an itemto ship directly to the consumer without leaving home can be overshadowed by misleading product descriptions and photos. Many consumers use product reviews from customers who have bought the item in the past to aid in decision making, but with potentially 1,000s of
reviews, determining customer sentiment can be a challenge. A machine learning model trained on customer reviews can alleviate this challenge by efficiently classifying the sentiment of large-scale reviews. With data scraped from the Amazon shopping website for four different streaming
devices, the objective of this research is to obtain a machine learning model with high accuracy, precision, and recall. Different sampling techniques balance the dominant positive class of reviews. The final model was a hyper-tuned support vector machine model and deployed in the open-source application framework, Streamlit.


![live demo part 1](https://github.com/jimmy-nguyen-data-science/Capstone-Project-Amazon-Product-Reviews/blob/main/Image/live%20demo.PNG)
![live demo part 2](https://github.com/jimmy-nguyen-data-science/Capstone-Project-Amazon-Product-Reviews/blob/main/Image/live%20demo%20part%202.PNG)
![live demo part 3](https://github.com/jimmy-nguyen-data-science/Capstone-Project-Amazon-Product-Reviews/blob/main/Image/live%20demo%20part%203.PNG)


# Data Science Objectives

The data science objective is to provide an in-depth analysis on streaming devices in 2022. This includes insights from sentiment analysis where we can monitor trends and observe overall opinion towards the brand and product.  We will aim to achieve the highest accuracy score as the performance metric and deploy it as a web-application where it can classifiy positive or negative ratings based on the user’s review.


# Methodology

- Problem statement and justification for the proposed approach
- EDA (graphical and non-graphical representations of relationships between the response variable and predictor variables)
- Data wrangling and pre-processing (handling of missing values, tokenization, removing stopwords, etc.)
- Data splitting (training, test sets, over-sampling vs. under-sampling techniques)
- Model strategies (Baseline/Dummy Classifier, SVM, and Random Forest)
- Validation and testing (model tuning and evaluation)
- Results and final model selection (performance measures, etc.)
- Discussion and conclusions (address the problem statement and suggestions that could go beyond the scope of the project.)

# Data Visualization 

![Product and Product Ratings](https://github.com/jimmy-nguyen-data-science/Capstone-Project-Amazon-Product-Reviews/blob/main/Image/star%20ratings%20by%20product.png)
![Bi-gram Word Cloud](https://github.com/jimmy-nguyen-data-science/Capstone-Project-Amazon-Product-Reviews/blob/main/Image/word%20cloud%20-%20top%20100%20bigrams.PNG)


# Technologies
- Python, R, 
- Microsoft Powerpoint
- Microsoft Word

# Conclusion

Overall, the final model using the SVM algorithm with its tuned hyper-parameters on the one-sided selection sampling data performs the best with the highest accuracy of 94.69%, whereas its baseline criteria were 88.93%. The reasoning behind this performance metric selection is due to the balanced weights of positive and negative
reviews equally. For validation, an unseen sample from a negative review is selected for testing each model. The SVM model classifies the review accurately with a probability of 98.9% negative, but the random forest model classifies it incorrectly with 53.5% positive and 46.5% negative. Due to this result, only the SVM model
is considered the final model for deployment.


# Next Steps

A more simplified approach without a class imbalance problem is to collect more data. Currently, the training data scraped in this project is predominantly positive reviews. A step to progression is collecting more customer reviews based on other streaming devices from Amazon.com. This larger sample should include
ratings from the neutral class too. Thus, expanding into a multi-class classification problem. Alternatively, a potential change in direction can be classifying the product rating based on the number of stars out of five for a product instead.

# Reference

Fernández, A., García, S., Galar, M., Prati, R. C., Krawczyk, B., & Herrera, F. (2018). Learning from imbalanced data sets (1st ed.). Springer. 1–6. 
https://doi.org/10.1109/ICIRD.2018.8376299 
  
Haque, T. U., Saber, N. N, & Shah, F. M. (2018). Sentiment analysis on large-scale Amazon product reviews. 2018 IEEE International Conference on Innovative Research   and Development (ICIRD), Bangkok, Thailand, 1–6. https://doi.org/10.1109/ICIRD.2018.8376299 

Kokje, R. C., & Chouhan, G. S. (2020). A supervised learning technique for classifying Amazon product reviews based on buyers sentiments. International Journal of     Computer Applications, 175(38), 36–41. https://doi.org/10.5120/ijca2020920956        

Rathor, A. S., Agarwal, A., & Dimri, P. (2018). Comparative study of machine learning approaches for Amazon reviews. Procedia Computer Science, 132, 1552–1561.       https://doi.org/10.1016/j.procs.2018.05.119 
  
Shrestha, N., & Nasoz, F. (2019). Deep learning sentiment analysis of Amazon.com reviews and ratings. International Journal on Soft Computing, Artificial Intelligence and Applications, 8(1), 01–15. https://doi.org/10.5121/ijscai.2019.8101 
  
Srujan, K. S., Nikhil, S. S., Raghav Rao, H.,  Karthik, K., Harish, B. S., & Keerthi Kumar, H. M. (2018). Classification of Amazon book reviews based on sentiment analysis. Information Systems Design and Intelligent Applications, 401–411. https://doi.org/10.1007/978-981-10-7512-4_40  
  
Wang, K., Tian, J., Zheng, C., Yang, H., Ren, J., Li, C., Han, Q., & Zhang, Y. (2021). Improving risk identification of adverse outcomes in chronic heart failure  using SMOTE+ENN and machine learning. Risk Management Healthcare Policy, 2021(14), 2453–2463. https://doi.org/10.2147/RMHP.S310295 


