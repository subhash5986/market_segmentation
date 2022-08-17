#!/usr/bin/env python
# coding: utf-8

# # SUBHASH RAJ 

# ## Ocean Engineering

# ## ROLL.NO : 213040063

# ## Data Collection 

# ### I have collected the data first. For this case, i take the data from UCI Machine Learning called Online Retail dataset. The dataset itself is a transactional data that contains transactions from December 1st 2010 until December 9th 2011 for a UK-based online retail. Each row represents the transaction that occurs. It includes the product name, quantity, price, and other columns that represents ID. The size of dataset is (541909,8). In this case, we haven’t used all of the rows. Instead, we have sample 10000 rows from the dataset, and we assumed that as the whole transactions that the customers do.

# ## Use Case of Customer Segmentation

# ### Segmentation using K-Means clustering algorithm
# ###    Suppose that we have a company that selling some of the product, and you want to know how well does the selling performance of the product.
# 
# ###    we have the data that can we analyze, but what kind of analysis that we can do?
# 
# ###    Well, we can segment customers based on their buying behaviour on the market.
# 
# ###    Keep in mind that the data is really huge, and we can not analyze it using a bare eye. We have to use machine learning algorithms and the power of computing for it.
# 
# ###    This project will show how to cluster customers on segments based on their behaviour using the K-Means algorithm in Python. This project will help on how to do customer segmentation step-by-step from preparing the data to cluster it. WE hve used here RFM model i.e. recency frequency and monetary model.

# ## 1. Gather the data

# In[7]:


import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import numpy as np


# In[16]:


retail_dataframe= pd.read_excel(r'C:\Users\HP\Downloads\dataset.xlsx')


# In[22]:


## checking the shape of the dataframe
reatil_dataframe.shape


# In[17]:


## checkimg the head of the dataframe
retail_dataframe.head()


# In[19]:


##checking the null values in the column 
retail_dataframe.isnull().any()


# In[20]:


## checking the number of rows which have null values
reatil_dataframe.isnull().sum()


# In[24]:


## Dropping the rows which have na values 
df_fix=retail_dataframe.dropna()


# In[25]:


df_fix.head()


# ## Data Preprocessing 

# ### Create RFM Table: here we due some feature engineering in which we try to extract three features namely recency(how recently products are bought) frequency(how often products are bought) and monetary value(what is the value spent by each customer)

# In[26]:


# Convert to show date only
from datetime import datetime
df_fix["InvoiceDate"] = df_fix["InvoiceDate"].dt.date

# Create TotalSum colummn
df_fix["TotalSum"] = df_fix["Quantity"] * df_fix["UnitPrice"] #total sum is the product of unit price and no of units

# Create date variable that records recency
import datetime
snapshot_date = max(df_fix.InvoiceDate) + datetime.timedelta(days=1)

# Aggregate data by each customer
customers = df_fix.groupby(['CustomerID']).agg({
    'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
    'InvoiceNo': 'count',
    'TotalSum': 'sum'})

# Rename columns
customers.rename(columns = {'InvoiceDate': 'Recency',
                            'InvoiceNo': 'Frequency',
                            'TotalSum': 'MonetaryValue'}, inplace=True)


# In[27]:


customers.head()


# ## Manage Skewness:we convert the given data into a non skewed data by using box-cox transformation

# In[28]:


fig, ax = plt.subplots(1, 3, figsize=(15,3))
sns.distplot(customers['Recency'], ax=ax[0])
sns.distplot(customers['Frequency'], ax=ax[1])
sns.distplot(customers['MonetaryValue'], ax=ax[2])
plt.tight_layout()
plt.show()


# In[29]:


from scipy import stats
def analyze_skewness(x):
    fig, ax = plt.subplots(2, 2, figsize=(5,5))
    sns.distplot(customers[x], ax=ax[0,0])
    sns.distplot(np.log(customers[x]), ax=ax[0,1])
    sns.distplot(np.sqrt(customers[x]), ax=ax[1,0])
    sns.distplot(stats.boxcox(customers[x])[0], ax=ax[1,1])
    plt.tight_layout()
    plt.show()
    
    print(customers[x].skew().round(2))
    print(np.log(customers[x]).skew().round(2))
    print(np.sqrt(customers[x]).skew().round(2))
    print(pd.Series(stats.boxcox(customers[x])[0]).skew().round(2))


# In[30]:


# analyze_skewness('Recency')
analyze_skewness('Frequency')


# In[31]:


fig, ax = plt.subplots(1, 2, figsize=(10,3))
sns.distplot(customers['MonetaryValue'], ax=ax[0])
sns.distplot(np.cbrt(customers['MonetaryValue']), ax=ax[1])
plt.show()
print(customers['MonetaryValue'].skew().round(2))
print(np.cbrt(customers['MonetaryValue']).skew().round(2))


# In[33]:



# Set the Numbers
customers_fix = pd.DataFrame()
customers_fix["Recency"] = stats.boxcox(customers['Recency'])[0]
customers_fix["Frequency"] = stats.boxcox(customers['Frequency'])[0]
customers_fix["MonetaryValue"] = pd.Series(np.cbrt(customers['MonetaryValue'])).values
customers_fix.tail()


# ## Centering and Scaling Variables: we convert the data to have the same mean and variance.We have to normalize it. To normalize, we can use StandardScaler object from scikit-learn library to do it. The code will look like this.

# In[35]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(customers_fix)
customers_normalized = scaler.transform(customers_fix)
print(customers_normalized.mean(axis = 0).round(2))
print(customers_normalized.std(axis = 0).round(2))


# In[36]:


pd.DataFrame(customers_normalized).head()


# # Modelling

# ### Choose k-number:To make segmentation from the data, we can use the K-Means algorithm to do this.K-Means algorithm is an unsupervised learning algorithm that uses the geometrical principle to determine which cluster belongs to the data. By determine each centroid, we calculate the distance to each centroid. Each data belongs to a centroid if it has the smallest distance from the other. It repeats until the next total of the distance doesn’t have significant changes than before.

# In[38]:



from sklearn.cluster import KMeans

sse = {}
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(customers_normalized)
    sse[k] = kmeans.inertia_ # SSE to closest cluster centroid

plt.title('The Elbow Method')
plt.xlabel('k')
plt.ylabel('SSE')
sns.pointplot(x=list(sse.keys()), y=list(sse.values()))
plt.show()


# In[39]:


model = KMeans(n_clusters=3, random_state=42)
model.fit(customers_normalized)
model.labels_.shape


# ## Based on our observation, the k-value of 3 is the best hyperparameter for our model because the next k-value tend to have a linear trend. Therefore, our best model for the data is K-Means with the number of clusters is 3.

# In[40]:


## cluster Anlaysis
customers["Cluster"] = model.labels_
customers.head()


# In[41]:


customers.groupby('Cluster').agg({
    'Recency':'mean',
    'Frequency':'mean',
    'MonetaryValue':['mean', 'count']}).round(1)


# In[42]:


## plots
df_normalized = pd.DataFrame(customers_normalized, columns=['Recency', 'Frequency', 'MonetaryValue'])
df_normalized['ID'] = customers.index
df_normalized['Cluster'] = model.labels_
df_normalized.head()


# In[43]:


# Melt The Data
df_nor_melt = pd.melt(df_normalized.reset_index(),
                      id_vars=['ID', 'Cluster'],
                      value_vars=['Recency','Frequency','MonetaryValue'],
                      var_name='Attribute',
                      value_name='Value')
df_nor_melt.head()


# In[44]:


## plots of the dist
sns.lineplot('Attribute', 'Value', hue='Cluster', data=df_nor_melt)


# In[45]:


customers.groupby('Cluster').agg({
    'Recency':'mean',
    'Frequency':'mean',
    'MonetaryValue':['mean', 'count']}).round(1)


# ### cluster 2 is frequent, spend more, and they buy the product recently. Therefore, it could be the cluster of a ###loyal customer. Then, the cluster 0 is less frequent, less to spend, but they buy the product recently. Therefore, it could be the cluster of ###new customer. Finally, the cluster 1 is less frequent, less to spend, and they buy the product at the old time. Therefore, it could be the cluster of ###saturated or old customers.

# In[46]:


cluster_avg = customers.groupby('Cluster').mean()
population_avg = customers.mean()
relative_imp = cluster_avg / population_avg - 1
relative_imp


# # Conclusion

# ### The customer segmentation is really necessary for knowing what characteristics that exist on each customer. Which customer loyal customer, new customer and churned customers is clearly segmented by using K-Means clustering. The project has shown to you how to implement it using Python. We infer that cluster 2 is frequent, spend more, and they buy the product recently. Therefore, it could be the cluster of a loyal customer. Then, the cluster 0 is less frequent, less to spend, but they buy the product recently. Therefore, it could be the cluster of new customers. Finally, the cluster 1 is less frequent, less to spend, and they buy the product at the old time. Therefore, it could be the cluster of churned customers. SUGGESTIONS:- 1) Since 0 is our new customer segmentation we can try to convert them into loyal customer by providing them the attractive offers and discount to maximize our profit. 2) cluster 1 is our churned out customer that means either they are saturated or not interested in our shop so theres not much scope for improvement in this segment. 3) cluster 3 is our loyal and most important cluster and keeping and increasing this cluster is the most important thing for the shop.

# ## References
# https://www.geeksforgeeks.org/elbow-method-for-optimal-value-of-k-in-kmeans/ https://www.marketingprofs.com/tutorials/snakeplot.asp reference for snake plot