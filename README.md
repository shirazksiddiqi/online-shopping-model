# online-shopping-model
This is the report that was submitted of the findings and conclusion of the model.
This effort was by 5 students for the course Data Mining for Business Intelligence.

Problem description and the current state of the domain
Problem Statement
To predict if the customer will buy from a franchise merchant (e.g. BestBuy/Walmart etc.) or a local merchant based on purchasing preferences.

As consumers we all have faced the dilemma of where to buy a particular product from. Each one of us has a different purchasing pattern that revolves around our comfort and priority.
Many merchants offer different types of purchasing options and assistance. Based on these a consumer can make the decision of purchasing from a particular merchant.

Some of the different purchasing preferences are:
Price – The cheapest price overall for a particular product.
Availability – If a particular merchant has that product available
Shipping – If the merchant can deliver and when is the soonest delivery
Discount/Promotions – If there is an ongoing or upcoming sale on the item

Understanding consumer behavior and aiding their search for a better deal, some organizations help by consolidating the data.
Examples of some of these organizations are:

PriceOye.pk
This website gathers information from other websites about a particular electronic product. The information includes selling price and shipping details other than the link to the seller’s website.

The Verge
Recent Black Friday sale was a good source of gathering and understanding data on consumer behavior. We see that even after the sale consumers wanted to know which merchants are giving the best deal.

An article by The Verge helps the consumers find merchants that match their purchasing preferences as shown in figure 2.
We can see that the merchants are clearly mentioned with the deal they are providing.

Out of the merchants that have these offers and deals, they can be divided in two types.
1 – Franchise Merchants
2 – Local Merchants

Franchise merchants are the ones that are owned by a company and not a person. Most of them are on the stock exchange market and have stores everywhere in the United States.
Local merchants are owned by one owner or partners and operate in certain area and buy from the manufacturers to sell in their stores, both online and offline.

As both types of merchants base their sales and marketing by taking into customer behavior, both have offers that are attractive.

After considering the practical value, our project aims to find how based on historical data, that given certain behaviors, where did the customer purchase an electronic product from, a franchise merchant or a local merchant.

Dataset description: origin, data points, variables 

Dataset Source: Our dataset contains information about Electronic Products being sold in the market. It has been taken from Datafiniti. 
Datafiniti helps businesses become data driven by offering easy access to a variety of high-quality, comprehensive data sets. The clients of Datafiniti include Fortune 500 companies to startups such as Nickelodeon. Hence the dataset source can be concluded to be a viable source.
The dataset used as part of this project has been obtained from Datafinity [URL: https://datafiniti.co ]. The data describes the many attributes used in this project concerning e-commerce websites and offline retail stores selling electronic goods. 
Data Description:
14,592 data points are being used for mining the data. The dataset was originally spread across 26 attributes. The dependent variable (output) is modeled around the merchant name, from where the user would acquire electronic goods. After cleaning, reducing and transforming the data, we model the dependent variable using 11 independent variables. These 11 independent variables comprise of the categorical nominal variable, Boolean variable, and numeric ratio scale as shown in table 1.


	Name of Attribute	Type	Description
1	AmountMax	Numeric	The maximum selling price of a product.
2	AmountMin	Numeric	The minimum selling price of a product.
3	Condition	Categorical	The condition of the product that its being sold as. It can be new, refurbished and used.
4	Online.offline	Boolean	If the product was sold online or in store
5	Shipping	Boolean	If the product was shipped or not
6	TimeSold	Numeric	Number of times products has seen to be sold
7	Weight	Numeric	The weight in pounds of the product
8	Brand	Categorical	Name of brands of the products
9	Availability	Boolean	If the product is available or not for purchase.
10	Currency	Categorical	The currency required to purchase the product.
11	IsSale	Boolean	If the product is on sale or not.
12	Merchant	Categorical	The merchant from where the product is available


Data preprocessing activities and results:
The data from the original source does not have missing values but needs processing before it can be used for data mining. The following are the activities and actions used on the different variables/attributes.

To run correlation, we converted all values to numerical.

1 - Data Integration 
Entity Identification
Same entities that mean the same thing for that particular variable.

Availability
True = Yes, In Stock, Limited Stock, 32 available, 7 available etc
False = No, Sold, Discontinued etc
For Correlation
True = 1
False = 0

Condition
New = Brand New, New other etc
Used = Pre-owned
Refurbished = Manufacturer refurbished, Seller refurbished
For Correlation
New = 1
Used = 2
Refurbished = 3

Shipping
Applicable = Free, Expedited, Standard, Free Shipping etc
Not Applicable = Blank
For Correlation
Applicable = 1
Not Applicable = 0

Merchant
BestBuy and Walmart
Others = accessory.net, Adorama, afdsolutions, memoryC, luckystore etc (8975 rows)
For Correlation
BestBuy & Walmart = 1
Others = 0

Weight
All values were converted from other units to ‘pounds. 
The mathematical conversation was as follows:
1 Ounce(oz) = 0.0625 Pounds
1 Kg = 2.2 Pounds
1 Gram = 0.0022 Pounds

Duplicate Information
Weight
The variable Weight has numerical values followed by the unit in characters but the unit is not uniform. Some values have been recorded in ounces and other are in pounds.
Even in those two categories, ounces are written as ‘oz’ and in whole words. Pounds are written as ‘lbs’ or ‘lb’ other than proper word. In the next stage we will use different functions to homogenize this variable.

Manufacturer/Brand
These two variables have exactly the same entities. They both give the same information about the manufacturer of the product which can also be labeled as brand.

2 - Data Reduction

Irrelevant Features
Id - Website ID associated with this review
No verification needed of the review
Prices.sourceURLs - Source of where the price was taken from
No verification needed of the price
ASINS - Amazon identifier for products
Cross platform product prices are used for the project not specific amazon
Categories - Keywords used for product search
All products are electronics so specific keywords are not needed
DateAdded - The date the product was added to the database
Only sale related information is relevant
DateUpdated - The most recent date the product was updated in the database 
Only sale related information is relevant
EAN - European Identification Number
The project focuses on US sales only
ImageURLs - The link for the product image
The project does not involve any image processing
Keys - Internal identifiers for Datafiniti
A classification not needed by us
ManufacturerNumber - Manufacturer Id number
Similar to manufacturer name
Name - Product Name
No variation in product name so brand is enough for identifying product
SourceURLs - Different website URLs where the product was seen
Only sale related information is relevant
UPC - Universal Product Code
The project focuses on US sales only
PrimaryCategories - 
All products are electronics so specific item category is not needed

From the original data, some attributes were transformed to add more insight to the data and make them more meaningful for our project.

3- Data Transformation
Feature Construction
Prices.online/offline 
This variable was created from Prices.shipping which had information about the shipping price. The products that weren’t shipped were bought in a physical store or picked up from a store location.
If the product was shipped, then it was an online transaction and if no shipping information was provided then it was an offline transaction.

Prices.sellcount from prices.dateSeen
The variable prices.dateSeen had the information about the dates that the product was recorded to be sold. Using actual dates was of no use so using a function that counted comma separated values, Prices.sellcount was created, a new variable that had the count of times sold. This is measures of frequency, a branch of descriptive statistics.

Average Price
The average taken of variables amountMax and amountMin results in a new feature called Average Price. This will help us to relate the price to the product better and the attribute ‘IsSale’ already gives us the information if the product was on sale or not.

Transformed Data
 After integration and transformation, the following attributes can be used for data mining as shown in table 2.

	Name of Attribute	Transformed Values	Numeric Value (Correlation)
1	AmountMax	1-26871	1-26871
2	AmountMin	1-26871	1-26871
3	AveragePrice	1-26871	1-26871
4	Condition	New, Used, Refurbished 	1, 2,3 
5	Online.offline	True, False	1, 0
6	Shipping	Applicable, Not Applicable	1, 0 
7	SellCount	1-198	1-198
8	Weight	0.01-350.00 lbs	0.01-350.00 lbs
9	Brand	309 Different	309 Different
10	Availability	True, False	1, 0
11	Currency	CAD, EUR, GBP, SGD, USD	1, 2, 3, 4, 5
12	IsSale	True, False	1, 0
13	Merchant	BestBuy & Walmart, Others	1, 0

Incorporation of previous rounds of feedback by instructor, TA, and class mates.

Dataset alteration:
The first and foremost major feedback we received was on our dataset. Our first dataset, which we had decided to work on was Crime Records in Boston for 2014-2019. But the primary drawback of this dataset was lack of sufficient categorical variables to perform any predictive modelling algorithms to get an efficient result. Thus, following this feedback, we considered changing the dataset to our current one.

Calculating correlation between variables:
For our stage 1, we used Naive Bayes algorithm which gave us a very good accuracy of 72.2%. But we missed out on checking the correlation between our independent variables. Finding correlation is very important as it will help us identify the strength between variables which gives us an idea to select the independent variables to perform our algorithm. Thus, we decided to work on this feedback for our stage 2 and stage 3.

Problem statement relation with real life problems:
Another major feedback we got was in regard to our problem statement. Our problem statement seemed to be very vague in terms of comprehending what we wanted to show through our prediction. Thus, we worked on our problem statement and provided some examples where our problem statement was relevant. We also altered a few things in our problem statement so that it became more comprehensible.
Categorical variable and Numeric conversion:
We received feedback related to our categorical variables and their conversions. For each of our categorical variable, for ease of application of algorithm, we converted them all to numerical values and ran our algorithm. We didn’t realize that by using numerical values, we were assigning weights to them, thus causing deviation from our results. Therefore, we converted them back to their original values and then run the algorithm.
 
Results of individual projects of team members and how they influenced the team project.

1.RITOBAN	
2.SHIRAZ	
3.SARISKA	
4.AMIT	
5.DEVI

DEPENDENT VARIABLE
1.merchant
2.merchant
3.merchant
4.availability	
5.Product condition

INDEPENDENT 
VARIABLES
1.Availability, condition, isSale, sellcount, online/offline
2.Condition, online/offline, Shipping, sellcount, weight, brand	
3.Availability, condition, sale, shipping, online/offline	
4.Condition, sale, shipping, sellcount, merchant	
5.Merchant, Average Selling Price, Online vs Offline, On Sale, Sell count

PREPROCESSING 
DATA
1.Categorical to numerical transformation.
2.Categorical to numerical transformation	
3.General statistical preprocessing	
4.Categorical to numerical transformationand data reduction
5.Categorical to numerical transformation and data reduction

ALGORITHM USED
1.Decision Tree	
2.Naïve Bayes, SVM	
3.SVM, Neural Network
4.Naïve Bayes	SVM,
5.Decision Tree
ACCURACY
1.94.76%
2.69.03%, 69.26%	
3.~73%, ~68%
4.80.22%
5.85.54%,86.60%

The above table summarizes our individual algorithm applications on various independent and dependent variables. Each individual used different sets of independent variables and tried various algorithms on them to find the accuracy each algorithm gave as compared to the baseline accuracy.

Each one of us saw major accuracy changes, between 20% to 40% as compared to baseline accuracy. This helped us understand which variables were to be used and which were to be discarded for our final stage 3 predictive modelling.

Rationale behind the choice of algorithm and variables for Stage 3

STAGE 3 ALGORITHM
For are STAGE 3 project presentation we chose DECISION TREE over the other predictive models. The reasons for our choice of algorithm is because Decision Trees are 

1.	Simple
2.	Takes lesser training time
3.	Could be visualized easily

Decision tree is more flexible and easier compared to Naïve Bayes. It takes lesser processing time compared to Neural Networks. Decision trees are better when there is large set of categorical values in training data. Our data set has more categorical variables and most of the variables we chose to model for a STAGE 3 were also categorical.

STAGE 3 VARIABLES

For our STAGE 3 project presentation we chose the following variables to run our algorithm.

Dependent variable: Merchant
Independent variable:  online/ offline, Availability, Average Selling price, Shipping, On sale, Product condition.

After two stages of project exploration we decided that the above-mentioned Independent variables most impact from a business perspective. We also understood that running a correlation is a very important process of a data preprocessing activity.

Using correlation, we get the following insights.
1.	One or multiple attributes depend on another attribute or a cause for another attribute.
2.	One or multiple attributes are associated with other attributes.

Hence data and feature correlation are an important part of a data preprocessing activity for the following reasons.
•	Correlation can help in predicting one attribute from another (Great way to impute missing values).
•	Correlation can (sometimes) indicate the presence of a causal relationship.
•	Correlation is used as a basic quantity for many modelling techniques

We ran the correlation among our variables and understood that the variables which we decided would make the most impact on our prediction were the most positively correlated variables. Below is the R code and our findings post running a Feature correlation.

R Code for Correlation :
data <- read.csv("Trial2.csv", header=TRUE)
# Read the data file that consists of numerical variables into data.
cor(data)
# Find correlation between variables

Results and Interpretation
After running the R code for the intended algorithm, we find that customers prefer to buy electronic products through flagship merchants such as Walmart and Best Buy. 
The R code to implement the algorithm is shown below:

library(C50)  
#The library that needs to be imported to implement C5.0. model which is a part of Decision Tree Algorithm
phase3<-read.csv("Final_Sheetv7.csv")  
# Read the CSV file into phase3.
summary(phase3)  
# Gives summary of the dataset, including counts of factor variables and mean, median and quadrant data of numeric variables.
phase3$prices.merchant <- as.factor(phase3$prices.merchant) 
#  The dependant variable(Merchant) should also be converted into factor in order to implement C5.0
sample_size <- floor (0.7 * nrow(phase3)) 
#  Calculate sample size as 70% of the entire dataset
training_index <- sample(nrow(phase3),size = sample_size) 
#  Compute an index, to calculate train and test dataset
train <- phase3[training_index,] 
#  Training Dataset 
test <- phase3[-training_index,] 
#  Testing Dataset
predictors <- c('prices.isAvailable', 'prices.condition', 'prices.isSale','prices.online.offline', 'prices.shipping','brand','prices.average') 
#  Chose the predictor variables after proper analysis
model <- C5.0.default(x = train[,predictors], y = train$prices.merchant) 
# Run the model on the training dataset
summary(model) 
# Check to see attribute usage
plot(model) 
# Plot the decision tree

The summary of the model shows that the most used independent variables are prices.condition and prices.online.offline which are being used a 100% of the time. Other variables being used are prices.isavailable, prices.issale,prices.average,prices.shipping as shown in the figure 10. 


The decision tree obtained after executing the above R code snippet is as shown:
As we see from the decision tree, node 4,8 and 15 are the only nodes which gives us a decision that an individual must be buying the electronic products from the local vendors, instead of going to flagship stores such as Walmart or best buy.

Node 4 : When the product is not available online and the condition of the product is new or used.
Node 8 : When the product is present online, and the condition is new. However, the product pricing is lower compared to the other products i.e. they are not expensive.
Node 15 : When the product is present online, condition is new, the price is high, shipping is available, and the product is on sale.

Our trained model is now tested with the 30 percent test dataset. We achieve ~ 74.5 % accuracy as seen in figure 


pred<- predict(model,newdata=test) 
# Predict the dependent variable in the test dataset using the model learning
evaluation<- cbind(test, pred) 
# Bind the predicted variable in above step with test
evaluation$correct <- ifelse(evaluation$prices.merchant == evaluation$pred,1,0) 
# If the output is predicted correctly, assign 1 else 0
sum(evaluation$correct)/nrow(evaluation) 
# Calculate Accuracy

Recommendations:
With the distribution shown in the decision tree, we can see that most transactions are online hence having a good e-commerce website that helps the consumer make the purchase is necessary for a successful sale.
Most consumers want a new product and are willing to pay more compared to used or refurbished. So for BestBuy/Walmart the best chances for a sale is product present online, new, high price and available for in-store purchase, and if the local stores can also give that offer, the consumer has the upper hand.

Evaluation
For the evaluation of the model, we have used the following metrics:
Sensitivity/Recall (also called the true positive rate, the recall, or probability of detection in some fields) measures the proportion of actual positives that are correctly identified as such (e.g., the percentage of sick people who are correctly identified as having the condition).
Specificity (also called the true negative rate) measures the proportion of actual negatives that are correctly identified as such (e.g., the percentage of healthy people who are correctly identified as not having the condition).
Along with this , we also calculate FPR and FNR based on sensitivity and specificity.

The R code used to calculate these metrics in R is as shown here:

table(evaluation$prices.merchant, evaluation$pred) 
# Confusion Matrix
tpr_sensitivity<- sum(evaluation$pred == '0' & evaluation$prices.merchant == '0') / sum(evaluation$prices.merchant == '0')
# Calculating sensitivity
tnr_specificity<-sum(evaluation$pred == '1' & evaluation$prices.merchant == '1')/ sum(evaluation$prices.merchant == '1')
# Calculating specificity
FPR<-1 - tnr_specificity
# Finding false positive rate
FNR<-1 - tpr_sensitivity
# Finding false negative rate
dt_precision<-sum(evaluation$prices.merchant == '0' & evaluation$pred == '0') / sum(evaluation$pred == '0')
# Calculating precision
dt_recall<-sum(evaluation$prices.merchant == '0' & evaluation$pred == '0') / sum(evaluation$prices.merchant == '0')
# Calculating the recall value
F<-2 * dt_precision * dt_recall / (dt_precision + dt_recall)
# Calculating the F score

From execution we get the following values for the metrics:

tpr_sensitivity : 77.12%
tnr_specificity : 71.19%
fpr : 28.80%
fnr : 22.87%
precision : 78.03%
recall : 77.12%
f-score : 77.57%

We also evaluate the model using AUC - ROC curve, which is a performance measurement for classification problem at various thresholds settings. ROC is a probability curve and AUC represent degree or measure of separability. It tells how much model is capable of distinguishing between classes. Higher the AUC, better the model is at predicting. The code used in R is as shown here:

library(pROC)
# Library used to plot the roc
reg <- glm(prices.merchant ~ prices.isAvailable+prices.isSale+prices.online.offline+prices.shipping+brand+prices.average+prices.condition , data = train, family = binomial() )
summary(reg)
# Summarizing dataframe reg
evaluation <- test
evaluation$prob <- predict(reg, newdata = evaluation, type = "response")
g <- roc(evaluation$prices.merchant ~ evaluation$prob, data = evaluation)
plot(g)
# Plotting the roc curve
auc(g)
# Calculating the area under the curve

From the AUC-ROC curve, we find the area under the curve to be 0.806 which is closer to 1, hence showing that the model can predict the dependent variable based on the independent variables to a large extent.

References:
https://wikipedia.com
https://towardsdatascience.com
https://priceoye.pk
https://www.theverge.com
https://towardsdatascience.com/
