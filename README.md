# Customer-Profitability-Analysis-and-Predictions-using-LSTM
*Using Convolutional Neural Networks to Detect and Analyze Human Emotions from Facial Images*

*Panjie Peng, Jenny (Yiran) Shen*

*Spring 2023*

## Documents

**If you want to skip right to the end, [click here for the video presentation](https://youtu.be/l5mb75gyGYE)**

The project proposal can [be found here.](https://github.com/wafiakmal/awesome-ml-group/blob/main/40_docs/IDS705%20Project%20Proposal.pdf)

The final project report can [be found here.](https://github.com/wafiakmal/awesome-ml-group/blob/c9ddff28e2eda9721bee869baeae76d9c15e6385/40_docs/IDS705%20Final%20Report.pdf)

## Abstract

The paper studies and analyzes the dynamic prediction of customer profitability over time. By collecting a real transaction dataset from a UK retail store, we use the Recency, Frequency, Monetary (RFM) model to measure customer profitability and accordingly generate a monthly RFM time series for each customer of the enterprise. At each time point, using k-means clustering and comparing the profitability of different categories, customers' RFM is divided into high, medium, or low groups. By counting the number of customers with different profitabilities as the window period changes, it was found that as the considered time period gets longer, the proportion of customers with different profitabilities remains basically stable. In addition, clustering analysis also provides a dynamic change process of each customer's profitability by labeling different customers in different window periods, providing data for the next step of using time series machine learning models to predict future customer profitability.

To further target customers for marketing, we trained a Recurrent Neural Network model and found that this machine learning model predicts the profitability of retail store customers with high accuracy. Businesses can use this predicted data for targeted marketing. For customers who are about to churn, they can use promotions to prevent their loss, and for customers whose profitability will increase in the future, they can use sales tactics to further enhance their profitability and so on.

## Project Overview

The onset of the COVID-19 Pandemic introduced new wrinkles to identifying emotions from observing faces. As much of the world began to socially distance or adopt facemasks as a standard practice, questions about the possible impacts on human interaction and emotional inference naturally emerged. While this challenge has already existed in certain cultures where the wearing of items that obstruct a part of a person’s face might be tradition, the COVID-19 pandemic brought this question to a whole new scale.

One of the most notable areas where this question extends today is the more prevalent use of webcams and virtual meetings. Everyone has experienced a virtual meeting where a colleague’s face might be partially blocked, poorly positioned in the view pane, or obstructed by other issues. Reading the audience’s emotions can be challenging in these environments, a critical task in human communication. With this in mind, this project will seek to gain a greater understanding of where computer vision might be able to augment human interpretation when a portion of the face is obstructed and where a model’s limitations are. 

Below is a represnetation of our experimental workflow:

<img src="/20_intermediate_files/ML Project Flowchart.jpg" width="800"/>


## Data

The dataset used in this article comes from an online retailer registered in the UK (Chen et al., 2012). The dataset contains 11 variables, and the attributes and specific meanings of the variables are shown in Table 1. It includes all transactions that occurred from 2009 to 2011. From December 2009 to December 2011, a total of 53,628 valid transactions were generated, involving purchases of a total of 5,305 products by 5,943 consumers.

## Exploratory Data Analysis

Our exploratory data analysis began by checking these emotional categories for balance and visual sampling to ensure that the author assigned the categories correctly. The visualization of the category balance shows some skew, while the visual inspection found multiple instances that the team considered either an erroneous assignment or a potentially duplicated image. A team member conducted an image-by-image review of the dataset to correct the erroneous categorization in the base dataset. This resulted in a slight alteration to the emotional categories, as the overall size of the dataset outweighed the small but concerning number of misclassified images. In total, we identified 1,506 images that had at least one exact copy. Anecdotally, we later observed cases where a face was present in more than one image, but the aspect ratio, zoom, or other features were altered very slightly. These cases escaped our inspection because they were not identical copies of one another. The figure below presents the dataset before and after these steps were taken.

<img src="/20_intermediate_files/ML EDA.png" width="800"/>

## Model Results

Below are the results of our model evaluations. We find that both models perform reasonably well, with some struggles in specific emotions:

<img src="/20_intermediate_files/ML VGG.png" width="700"/>

<img src="/20_intermediate_files/ML Resnet.png" width="700"/>

## Saliency Maps

Once the model evaluation was complete, we generated saliency maps of the outcomes to compare the areas of importance for detecting specific emotions in our images. Below is a comparison of the saliency map for predictions created from all facial expressions determined by VGG-16 and ResNet50:

<img src="/20_intermediate_files/ML Saliency Maps.png" width="600"/>

## References:
Ale, L., Zhang, N., Wu, H., Chen, D., & Han, T. (2019). Online Proactive Caching in Mobile Edge Computing Using Bidirectional Deep Recurrent Neural Netwoek. IEEE Internet of Things Journal, (6), 5520-5530. 

Chen, D., Sain, S.K., & Guo, K. (2012). Data Mining for the Online Retail Industry: A Case Study of RFM Model-Based Customer Segmentation Using Data Mining. Journal of Database Marketing and Customer Strategy Management, (19), 197-208.

Chen, D., Guo, K., & Ubakanma, G. (2015). Predicting Customer Profitability Over Time Based on RFM Time Series. International Journal of Business Forecasting and Marketing Intelligence, (2), 1-18.

Chen, D., Guo, K., & Li, B. (2019). Predicting Customer Profitability Dynamically Over Time: An Experimental Comparative Study, Based on RFM Time Series. 24TH Iberoamerican Congress on Pattern Recognition, Cuba, 28-31.

Januszewski, F. (2011). Possible Applications of Instruments of Measurement of the Customer Value in the operations of Logistcis Companies. Scientific Journal of Logistics, (7), 17-25. 

Manaswi, N.K. (2018). RNN and LSTM. Deep Learning with Application Using Python, Apress. 
![image](https://github.com/JennyShen056/Customer-Profitability-Analysis-and-Predictions-using-LSTM/assets/112578566/fc948f1b-0a6b-4107-becb-4a5f80861388)
