#!/usr/bin/env python
# coding: utf-8

# **Pre request** 

# In[1]:


pip install snowflake-connector-python


# In[2]:


pip install numpy


# In[3]:


pip install snowflake-sqlalchemy


# In[4]:


pip install "snowflake-connector-python[pandas]"


# In[5]:


pip install jupyter_scheduler


# In[6]:


pip install numpy


# In[7]:


pip install jupyterlab-scheduler


# **1.pip install snowflake-connector-python**
# 
# **2.pip install snowflake-sqlalchemy**
# 
# **3.pip install "snowflake-connector-python[pandas]"**

# In[8]:


pip install pandas


# In[9]:


pip install pandas-profiling


# In[10]:


import numpy as np
import pandas as pd
import ydata_profiling
import matplotlib.pyplot as plt 
import getpass
import seaborn as sns 
get_ipython().run_line_magic('matplotlib', 'inline')
import snowflake.connector


# **Connecting to Snowflake**

# In[14]:


conn = snowflake.connector.connect(
        user = 'MJNEETHA23DS',
        password = getpass.getpass('Your Snowflake Password: '),
        ##password='MJneetha23',
        ##  account = https://uolykwb-gi26713.snowflakecomputing.com
        account = 'uolykwb-gi26713',
        database='RETAILS',
        schema='PUBLIC',
        warehouse='COMPUTE_WH',
  ) 


# In[15]:


cur = conn.cursor()


# In[16]:


select_demographic_RAW = 'SELECT * FROM demographic_RAW'
select_CAMPAIGN_DESC_RAW = 'SELECT * FROM CAMPAIGN_DESC_RAW'
select_CAMPAIGN_RAW = 'SELECT * FROM CAMPAIGN_RAW'
select_PRODUCT_RAW = 'SELECT * FROM PRODUCT_RAW'
select_COUPON_RAW = 'SELECT * FROM COUPON_RAW'
select_COUPON_REDEMPT_RAW = 'SELECT * FROM COUPON_REDEMPT_RAW'
select_TRANSACTION_RAW = 'SELECT * FROM TRANSACTION_RAW'


# In[17]:


cur.execute(select_demographic_RAW)
demographic_RAW = cur.fetch_pandas_all()


# In[18]:


cur.execute(select_CAMPAIGN_DESC_RAW)
CAMPAIGN_DESC_RAW = cur.fetch_pandas_all()


# In[19]:


cur.execute(select_CAMPAIGN_RAW)
CAMPAIGN_RAW = cur.fetch_pandas_all()


# In[20]:


cur.execute(select_PRODUCT_RAW)
PRODUCT_RAW = cur.fetch_pandas_all()


# In[21]:


cur.execute(select_COUPON_RAW)
COUPON_RAW = cur.fetch_pandas_all()


# In[22]:


cur.execute(select_COUPON_REDEMPT_RAW)
COUPON_REDEMPT_RAW = cur.fetch_pandas_all()


# In[23]:


cur.execute(select_TRANSACTION_RAW)
TRANSACTION_RAW = cur.fetch_pandas_all()


# In[24]:


cur.close()
conn.close()


# In[25]:


demographic_RAW.head(5)


# In[26]:


CAMPAIGN_DESC_RAW.head(5)


# In[27]:


CAMPAIGN_RAW.head(5)


# In[28]:


PRODUCT_RAW.head(5)


# In[29]:


COUPON_RAW.head(5)


# In[30]:


COUPON_REDEMPT_RAW.head(5)


# In[31]:


TRANSACTION_RAW.head(5)


# In[32]:


TRANSACTION_RAW.dtypes


# In[33]:


CAMPAIGN_DESC_RAW.shape


# In[34]:


COUPON_REDEMPT_RAW.shape


# In[35]:


COUPON_RAW.shape


# In[36]:


demographic_RAW.shape


# In[37]:


PRODUCT_RAW.shape


# In[38]:


TRANSACTION_RAW.shape


# In[39]:


CAMPAIGN_DESC_RAW.isnull().sum() #no cleaning is required


# In[40]:


CAMPAIGN_DESC_RAW.describe()


# In[41]:


PRODUCT_RAW.isnull().sum()


# In[42]:


PRODUCT_RAW.describe()


# In[43]:


demographic_RAW.isnull().sum()


# In[44]:


demographic_RAW.describe()


# In[45]:


COUPON_RAW.isnull().sum()


# In[46]:


demographic_RAW.describe()


# In[47]:


COUPON_REDEMPT_RAW.isnull().sum()


# In[48]:


COUPON_REDEMPT_RAW.describe()


# In[49]:


TRANSACTION_RAW.isnull().sum()


# In[50]:


TRANSACTION_RAW.describe()


# **Data Modifications**

# In[51]:


from datetime import datetime,timedelta


# In[52]:


start_date = pd.to_datetime('2020-01-01')


# In[53]:


start_date


# In[54]:


TRANSACTION_RAW.head(20)


# start_date_yyyy_mm_dd = start_date.strftime('%Y-%m-%d')

# In[55]:


TRANSACTION_RAW['Date'] = start_date + pd.to_timedelta(TRANSACTION_RAW['DAY'],unit='D')


# In[56]:


TRANSACTION_RAW['Date'].head(20)


# In[57]:


CAMPAIGN_DESC_RAW['Start_date']= start_date + pd.to_timedelta(CAMPAIGN_DESC_RAW['START_DAY'],unit='D')


# In[58]:


CAMPAIGN_DESC_RAW['End_date']=start_date + pd.to_timedelta(CAMPAIGN_DESC_RAW['END_DAY'],unit='D')


# In[59]:


CAMPAIGN_DESC_RAW.head(10)


# In[60]:


CAMPAIGN_DESC_RAW['Campaign_Duration'] = CAMPAIGN_DESC_RAW['END_DAY'] - CAMPAIGN_DESC_RAW['START_DAY']


# In[61]:


CAMPAIGN_DESC_RAW.head(20)


# In[62]:


COUPON_REDEMPT_RAW['Date'] = start_date + pd.to_timedelta(COUPON_REDEMPT_RAW['DAY'],unit='D')


# In[63]:


COUPON_REDEMPT_RAW.head(10)


# In[64]:


TRANSACTION_RAW['Date'].max()


# In[65]:


CAMPAIGN_DESC_RAW['End_date'].max()


# In[66]:


COUPON_REDEMPT_RAW['Date'].max()


# **Understanding the dataset**

# In[67]:


demographic_RAW.shape


# In[68]:


demographic_RAW.columns


# In[69]:


demographic_RAW.isnull().sum()


# In[70]:


demographic_RAW.dtypes


# In[71]:


demographic_RAW['AGE_DESC'].value_counts()


# In[72]:


demographic_RAW['HOUSEHOLD_SIZE_DESC'].value_counts()


# In[73]:


CAMPAIGN_DESC_RAW.shape


# In[74]:


CAMPAIGN_DESC_RAW.isnull().sum()


# In[75]:


CAMPAIGN_DESC_RAW.dtypes


# In[76]:


## CAMPAIGN_DESC_RAW['durations_days']=CAMPAIGN_DESC_RAW['End_date'] - CAMPAIGN_DESC_RAW['Start_date']


# In[77]:


CAMPAIGN_DESC_RAW.head(4)


# In[78]:


CAMPAIGN_DESC_RAW['Campaign_Duration'].mean()


# In[79]:


CAMPAIGN_DESC_RAW.dtypes


# **The Average Campaign Duration is 46.6 days**

# In[80]:


plt.figure(figsize=(15,5))
sns.barplot(x='CAMPAIGN',y='Campaign_Duration',data = CAMPAIGN_DESC_RAW)


# **Campaign 15 Lasted more than 160 days**

# In[81]:


CAMPAIGN_DESC_RAW.groupby('DESCRIPTION').aggregate({'CAMPAIGN':'count','Campaign_Duration':'mean'})


# **There have been 19 type B campaigns, whose average length was 38 days. In comparison, there has been 6 type C campaigns of 75 days on average.**

# In[82]:


CAMPAIGN_DESC_RAW['Start_month'] = CAMPAIGN_DESC_RAW['Start_date'].dt.strftime('%m')


# In[83]:


CAMPAIGN_DESC_RAW['End_month'] = CAMPAIGN_DESC_RAW['End_date'].dt.strftime('%m')


# In[84]:


CAMPAIGN_DESC_RAW['Start_Year'] = CAMPAIGN_DESC_RAW['Start_date'].dt.strftime('%Y')


# In[85]:


CAMPAIGN_DESC_RAW['End_Year'] = CAMPAIGN_DESC_RAW['End_date'].dt.strftime('%Y')


# In[86]:


CAMPAIGN_DESC_RAW.head(4)


# In[87]:


CAMPAIGN_DESC_RAW.dtypes


# In[88]:


CAMPAIGN_RAW.shape


# In[89]:


CAMPAIGN_RAW.columns


# In[90]:


CAMPAIGN_RAW.isnull().sum()


# In[91]:


CAMPAIGN_RAW['HOUSEHOLD_KEY'].nunique()


# In[92]:


TRANSACTION_RAW['HOUSEHOLD_KEY'].nunique()


# **There are 1584 households have participed to the campaign, leaving 916 households who never participated to any campaign.**

# In[93]:


CAMPAIGN_RAW.dtypes


# In[94]:


CAMPAIGN_RAW.groupby('HOUSEHOLD_KEY')['CAMPAIGN'].count()


# In[95]:


plt.figure(figsize=(15,5))
CAMPAIGN_RAW.groupby('CAMPAIGN')['HOUSEHOLD_KEY'].count().plot.bar()
plt.ylabel('Number of Households Reached To')


# **In Campaing 18 maximum number of households are participated.**

# In[96]:


COUPON_RAW.shape


# In[97]:


COUPON_RAW.columns


# In[98]:


COUPON_RAW['COUPON_UPC'].nunique()


# In[99]:


COUPON_RAW.isnull().sum()


# In[100]:


COUPON_RAW.dtypes


# In[101]:


Coupon_Given=COUPON_RAW.groupby("CAMPAIGN").aggregate(Total_product = ('PRODUCT_ID','nunique'),
                                                    Total_Coupon_Given = ('COUPON_UPC','nunique'))


# In[102]:


Coupon_Given.sort_values(by='Total_product',ascending=False).head(10)


# In[103]:


Coupon_Given.head(10)


# In[104]:


Coupon_Given = Coupon_Given.merge(right = CAMPAIGN_DESC_RAW,on='CAMPAIGN',how='left')


# In[105]:


Coupon_Given.head(10)


# In[106]:


Coupon_Given.columns


# In[107]:


Coupon_Given.loc[:,('CAMPAIGN','Total_product','Total_Coupon_Given','Start_Year','End_Year','Start_month','End_month','DESCRIPTION','Campaign_Duration')].sort_values(by='Total_product',
                    ascending=False).head(10)


# In[108]:


COUPON_RAW.head()


# In[109]:


PRODUCT_RAW.head()


# **campaign 13,18,8 are the one with most product in them.**

# In[110]:


coupon_product = COUPON_RAW.merge(right=PRODUCT_RAW,on='PRODUCT_ID',how='left')


# In[111]:


coupon_product.head(5)


# In[112]:


coupon_product.isnull().sum()


# **Top 10 product on which the Coupon has been applied**

# In[113]:


coupon_product['COMMODITY_DESC'].value_counts().head(10)


# **Most prominent products among coupons created are bathroom products such as hair care and makeup.**

# In[114]:


COUPON_REDEMPT_RAW.shape


# In[115]:


COUPON_REDEMPT_RAW.columns


# In[116]:


COUPON_REDEMPT_RAW['COUPON_UPC'].nunique()


# **Only 556 Coupon has been used out of 1135**

# In[117]:


COUPON_REDEMPT_RAW.groupby('CAMPAIGN').agg(total_coupon_reedm=('COUPON_UPC','nunique')).sort_values(by='total_coupon_reedm',ascending=False).plot.bar()


# In[118]:


Coupon_redeem = COUPON_REDEMPT_RAW.groupby('CAMPAIGN').agg(total_coupon_reedm=('COUPON_UPC','nunique'))


# In[119]:


Coupon_redeem.sort_values(by = 'total_coupon_reedm',ascending=False)


# In[120]:


Coupon_Given.head()


# In[121]:


Coupon_redeem.head(3)


# In[122]:


Coupon_redeem = Coupon_redeem.merge(right = Coupon_Given,on='CAMPAIGN',how='left')


# In[123]:


Coupon_redeem.head()


# In[124]:


Coupon_redeem['Coupon_redeem_rate']=(Coupon_redeem['total_coupon_reedm']/Coupon_redeem['Total_Coupon_Given'])*100


# In[125]:


Coupon_redeem.head().sort_values(by = 'Coupon_redeem_rate',ascending = False)


# In[126]:


plt.figure(figsize=(15,5))
sns.barplot(x='CAMPAIGN',y='Coupon_redeem_rate',data=Coupon_redeem)


# In[127]:


TRANSACTION_RAW.shape


# In[128]:


TRANSACTION_RAW.columns


# In[129]:


TRANSACTION_RAW.isnull().sum()


# In[130]:


TRANSACTION_RAW['BASKET_ID'].count()


# In[131]:


TRANSACTION_RAW['BASKET_ID'].nunique()


# In[132]:


TRANSACTION_RAW['HOUSEHOLD_KEY'].nunique()


# In[133]:


trnx_bucket =TRANSACTION_RAW.groupby('BASKET_ID').aggregate({'SALES_VALUE':'sum','COUPON_DISC':'sum','COUPON_MATCH_DISC':'sum'})


# In[134]:


trnx_bucket.head()


# In[135]:


trnx_bucket['Use_coupon'] = trnx_bucket['COUPON_DISC']!=0


# In[136]:


trnx_bucket['Use_coupon'].value_counts()


# In[137]:


trnx_bucket.sort_values('SALES_VALUE',ascending=False).head(10)


# In[138]:


round(trnx_bucket['SALES_VALUE'].mean(),2)


# The average basket value is $29.14

# In[139]:


plt.figure(figsize=(25,5))
sns.boxplot(x=trnx_bucket['SALES_VALUE'])
plt.title('Basket value boxplot', fontsize = 20)


# In[140]:


trnx_bucket.groupby('Use_coupon').aggregate( sales_mean=('SALES_VALUE','mean'),
                                             COUPON_DISC_mean =('COUPON_DISC','mean'),
                                             COUPON_MATCH_DISC=('COUPON_MATCH_DISC','mean'),
                                             No_coupon  =('SALES_VALUE','count'))


# The average basket value without a coupon is $26.79.
# 
# The average basket value with a coupon is $68.21.
# 
# The average discount generated by coupons is $2.98.

# **It shows that customeer purchase more product when coupon is given to them**

# In[141]:


trnx_desc = TRANSACTION_RAW.merge(right= trnx_bucket,on='BASKET_ID',how='left')


# In[142]:


trnx_desc.head()


# In[143]:


trnx_desc= trnx_desc.merge(right=PRODUCT_RAW,on='PRODUCT_ID',how='left')


# In[144]:


trnx_desc.head(3)


# In[145]:


trnx_desc.drop(['SALES_VALUE_y','COUPON_DISC_y','COUPON_MATCH_DISC_y'],axis=1,inplace=True)


# In[146]:


trnx_desc.head(10)


# In[147]:


COMMODITY_Coupon = trnx_desc.groupby('COMMODITY_DESC').aggregate(total_quantity=('QUANTITY','count'),
                                             Use_coupon=('Use_coupon','sum'),
                                             Coupon=('COUPON_DISC_x','sum'))


# In[148]:


COMMODITY_Coupon.head(10)


# In[149]:


COMMODITY_Coupon['Coupon%']=round((COMMODITY_Coupon['Use_coupon']/COMMODITY_Coupon['total_quantity'])*100,2)


# In[150]:


COMMODITY_Coupon.sort_values('Coupon%',ascending=False).head(30)


# **While the most prominents products for which coupons are created are haircare and makeup products, coupons are mostly used on drinks, cigarettes, diapers, etc. Bathroom products are not even among the top 10**

# In[151]:


TRANSACTION_RAW.groupby(['HOUSEHOLD_KEY','WEEK_NO','DAY']).aggregate({'SALES_VALUE':'sum','RETAIL_DISC':'sum',
                                                                  'COUPON_DISC' :'sum','COUPON_MATCH_DISC':'sum'})


# In[152]:


TRANSACTION_RAW.head()


# In[153]:


TRANSACTION_RAW.groupby(TRANSACTION_RAW['Date'].dt.year).aggregate({'SALES_VALUE':'sum','RETAIL_DISC':'sum',
                                                                   'COUPON_DISC':'sum'})


# **Sales Value increases as Retail Discount and Coupon Discount increases**

# **Droping the columns**

# In[154]:


from datetime import datetime


# In[155]:


CAMPAIGN_DESC_RAW.drop(['START_DAY','END_DAY'],axis=1,inplace=True)


# In[156]:


CAMPAIGN_DESC_RAW.head(3)


# In[157]:


CAMPAIGN_DESC_RAW['Start_date'] = pd.to_datetime(CAMPAIGN_DESC_RAW['Start_date']).apply(lambda x: x.date())


# In[158]:


type(CAMPAIGN_DESC_RAW['Start_date'])


# In[159]:


CAMPAIGN_DESC_RAW['Start_date'] 


# In[160]:


CAMPAIGN_DESC_RAW['End_date'] = pd.to_datetime(CAMPAIGN_DESC_RAW['End_date']).apply(lambda x: x.date())


# In[161]:


type(CAMPAIGN_DESC_RAW['End_date'])


# In[162]:


CAMPAIGN_DESC_RAW.dtypes


# In[163]:


COUPON_REDEMPT_RAW.drop(['DAY'],axis=1,inplace=True)


# In[164]:


COUPON_REDEMPT_RAW.head(3)


# In[165]:


COUPON_REDEMPT_RAW['Date']=pd.to_datetime(COUPON_REDEMPT_RAW['Date']).apply(lambda x: x.date())


# In[166]:


COUPON_REDEMPT_RAW.dtypes


# In[167]:


TRANSACTION_RAW.drop(['DAY','WEEK_NO'],axis=1,inplace=True)


# In[168]:


TRANSACTION_RAW.head()


# In[169]:


TRANSACTION_RAW['Date']=pd.to_datetime(TRANSACTION_RAW['Date']).apply(lambda x: x.date())


# In[170]:


TRANSACTION_RAW.dtypes


# In[171]:


TRANSACTION_RAW.head(10)


# **Now loading the table to Db**

# In[172]:


from sqlalchemy import create_engine
from sqlalchemy.engine import URL
import snowflake.connector as snowCtx
from snowflake.connector.pandas_tools import write_pandas
import pandas as pd
import getpass


# In[173]:


conn = snowflake.connector.connect(
     user = 'MJNEETHA23DS',
     ##password = getpass.getpass('Your Snowflake Password: '),
     password='MJneetha23',
     ##  account = https://uolykwb-gi26713.snowflakecomputing.com
     account = 'uolykwb-gi26713',
     database='RETAILS',
     schema='PUBLIC',
     warehouse='COMPUTE_WH',
  ) 


# In[174]:


cur=conn.cursor()


# In[175]:


cur.execute(''' 
CREATE OR REPLACE TABLE COUPON_REDEMPT_NEW
(HOUSEHOLD_KEY NUMBER(38,0),
COUPON_UPC NUMBER(38,0),
CAMPAIGN NUMBER(38,0),
Date Date
)''')


# In[176]:


cur.execute(''' 
CREATE OR REPLACE TABLE CAMPIGN_DESC_NEW
(DESCRIPTION VARCHAR(10),
CAMPAIGN NUMBER(38,0),
Start_date date,
End_date date,
durations_days NUMBER(38,0),
Start_month VARCHAR(10),
End_month VARCHAR(10))''')


# In[177]:


cur.execute('''CREATE OR REPLACE TABLE TRANSACTION_NEW
(HOUSEHOLD_KEY NUMBER(38,0),
BASKET_ID NUMBER(38,0),
PRODUCT_ID NUMBER(38,0),
QUANTITY NUMBER(38,0),
SALES_VALUE FLOAT,
STORE_ID NUMBER(38,0),
RETAIL_DISC FLOAT,
TRANS_TIME NUMBER(38,0),
COUPON_DISC FLOAT,
COUPON_MATCH_DISC FLOAT,
Date Date
)
''')


# In[178]:


CAMPAIGN_DESC_RAW.head() ## cleaned raw dataframe


# In[179]:


cur.execute('''
CREATE OR REPLACE TABLE CAMPAIGN_DESC_NEW
(DESCRIPTION VARCHAR(10),
CAMPAIGN NUMBER(38,0),
Start_date date,
End_date  date,
Campaign_Duration NUMBER(38,0),
Start_month VARCHAR(10),
End_month VARCHAR(10),
Start_Year INT,
End_Year INT)''')


# In[180]:


success, nchunks, nrows, _ = write_pandas(conn, CAMPAIGN_DESC_RAW,'CAMPAIGN_DESC_NEW',quote_identifiers=False)
print(str(success)+','+str(nchunks)+','+str(nrows))


# In[181]:


success, nchunks, nrows, _ = write_pandas(conn, COUPON_REDEMPT_RAW,'COUPON_REDEMPT_NEW',quote_identifiers=False)
print(str(success)+','+str(nchunks)+','+str(nrows))


# In[182]:


success, nchunks, nrows, _ = write_pandas(conn,TRANSACTION_RAW ,'TRANSACTION_NEW',quote_identifiers=False)
print(str(success)+','+str(nchunks)+','+str(nrows))


# In[183]:


cur.close()
conn.close()


# In[184]:


##CAMPAIGN_DESC_CLEANED = CAMPAIGN_DESC_RAW.copy()


# In[ ]:


##from sqlalchemy import create_engine


# In[ ]:


##connection_string = f"snowflake://{user}:{password}@{account}/{database}/{schema}?warehouse={warehouse}"


# In[ ]:


##engine = create_engine(connection_string)


# In[ ]:


##cur = conn.cursor()


# In[ ]:


##CAMPAIGN_DESC_RAW.to_sql(con=engine,name="CAMPAIGN_DESC_New",if_exists="append",index=False)


# In[ ]:


##COUPON_REDEMPT_RAW.to_sql(name="COUPON_REDEMPT_New",con=engine,if_exists="replace",index=False)


# In[ ]:


##TRANSACTION_RAW.to_sql(name="TRANSACTION_New",con=engine,if_exists="replace",index=False)


# In[ ]:


##TRANSACTION_RAW.shape


# In[ ]:


##TRANSACTION_RAW.to_csv('TRANSACTION.csv',index=False)


# In[ ]:


##chunk_size = 10000
##for chunk in pd.read_csv("TRANSACTION.csv", chunksize=chunk_size):
    ##chunk.to_sql('TRANSACTION_New',con= engine,if_exists='append', index=False)


# In[ ]:


##chunk_size = 16000


# In[ ]:


##chunks = [TRANSACTION_RAW[i:i+chunk_size] for i in range(0, len(TRANSACTION_RAW), chunk_size)]


# In[ ]:


##for chunk in chunks:
    ##chunk.to_sql(name="TRANSACTION_New", con=engine, if_exists='append', index=False)
    


# In[ ]:


##engine.dispose()


# In[ ]:


##programmingerror: (snowflake.connector.errors.programmingerror) 001042 (xx000): 
        ##sql compilation error: compilation memory exhausted
    
##(snowflake.connector.errors.ProgrammingError) 001795 (42601): SQL compilation error: error line 1 at position 187
##maximum number of expressions in a list exceeded, expected at most 16,384, got 100,000    


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




