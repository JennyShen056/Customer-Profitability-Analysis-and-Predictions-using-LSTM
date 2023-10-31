# Customer-Profitability-Analysis-and-Predictions-using-LSTM

*Jenny (Yiran) Shen, Panjie Peng*

*Spring 2022*

## Documents

The final project report can [be found here.](https://github.com/JennyShen056/Customer-Profitability-Analysis-and-Predictions-using-LSTM/blob/main/report_chinese_ver.pdf)

## Abstract

The paper studies and analyzes the dynamic prediction of customer profitability over time. By collecting a real transaction dataset from a UK retail store, we use the Recency, Frequency, Monetary (RFM) model to measure customer profitability and accordingly generate a monthly RFM time series for each customer of the enterprise. At each time point, using k-means clustering and comparing the profitability of different categories, customers' RFM is divided into high, medium, or low groups. By counting the number of customers with different profitabilities as the window period changes, it was found that as the considered time period gets longer, the proportion of customers with different profitabilities remains basically stable. In addition, clustering analysis also provides a dynamic change process of each customer's profitability by labeling different customers in different window periods, providing data for the next step of using time series machine learning models to predict future customer profitability.

To further target customers for marketing, we trained a Recurrent Neural Network model and found that this machine learning model predicts the profitability of retail store customers with high accuracy. Businesses can use this predicted data for targeted marketing. For customers who are about to churn, they can use promotions to prevent their loss, and for customers whose profitability will increase in the future, they can use sales tactics to further enhance their profitability and so on.

## Data

The dataset used in this article comes from an online retailer registered in the UK (Chen et al., 2012). The dataset contains 11 variables, and the attributes and specific meanings of the variables are shown in Table 1. It includes all transactions that occurred from 2009 to 2011. From December 2009 to December 2011, a total of 53,628 valid transactions were generated, involving purchases of a total of 5,305 products by 5,943 consumers.

## Exploratory Data Analysis

Based on the RFM values at the end of each time period, relevant customers were divided into five groups (clusters) using the k-means clustering algorithm. Each cluster contains a group of customers with similar RFM values. Then, these RFM values are aggregated to determine a unique single-value RFM score for all customers within the same cluster.

## Model Results

We take the data from the last timestamp as the test set and use the data from the previous time periods as the training set, resulting in the training curve shown in the figure below. At the same time, we validate the model's performance on the test set, and the training accuracy obtained is: 0.9665. The confusion matrix is shown in the following figure:

<img src="/visualization_output/confusion matrix.png" width="500"/>

<img src="/visualization_output/training loss of LSTM.png" width="500"/>

From above, it can be seen that using LSTM for prediction can achieve a very high accuracy rate. Therefore, using recurrent neural networks can precisely predict customer profitability.

## References:
Ale, L., Zhang, N., Wu, H., Chen, D., & Han, T. (2019). Online Proactive Caching in Mobile Edge Computing Using Bidirectional Deep Recurrent Neural Netwoek. IEEE Internet of Things Journal, (6), 5520-5530. 

Chen, D., Sain, S.K., & Guo, K. (2012). Data Mining for the Online Retail Industry: A Case Study of RFM Model-Based Customer Segmentation Using Data Mining. Journal of Database Marketing and Customer Strategy Management, (19), 197-208.

Chen, D., Guo, K., & Ubakanma, G. (2015). Predicting Customer Profitability Over Time Based on RFM Time Series. International Journal of Business Forecasting and Marketing Intelligence, (2), 1-18.

Chen, D., Guo, K., & Li, B. (2019). Predicting Customer Profitability Dynamically Over Time: An Experimental Comparative Study, Based on RFM Time Series. 24TH Iberoamerican Congress on Pattern Recognition, Cuba, 28-31.

Januszewski, F. (2011). Possible Applications of Instruments of Measurement of the Customer Value in the operations of Logistcis Companies. Scientific Journal of Logistics, (7), 17-25. 

Manaswi, N.K. (2018). RNN and LSTM. Deep Learning with Application Using Python, Apress. 
