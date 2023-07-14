# 这是文章Rossman药店药品销售预测的代码Predicting Sales- Rossman by Sourabh Pattanshetty


# Importing-Libraries-Required-for-the-Analysis
import numpy as np
import pandas as pd, datetime
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from time import time
import os
from math import sqrt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import itertools
import statsmodels.api as sm
from statsmodels.tsa.stattools import acf,pacf
from statsmodels.tsa.arima_model import  ARIMA
from sklearn import model_selection
from sklearn.metrics import mean_squared_error, r2_score
from pandas import DataFrame
import xgboost as xgb
from fbprophet import Prophet
import warnings
warnings.filterwarnings('ignore')


# Exploratory-Data-Analysis
# Importing store data
store = pd.read_csv('../input/sales-data-prediction/store.csv')
store.head()

# Importing Sales data
Sales = pd.read_csv('../input/sales-data-prediction/train.csv', index_col='Date', parse_dates = True)
Sales.head()


# Checking Sales data
Sales.head(5).append(Sales.tail(5))

Sales.shape


# Creating Alternate Features to work on
Sales['Year'] = Sales.index.year
Sales['Month'] = Sales.index.month
Sales['Day'] = Sales.index.day
Sales['WeekOfYear'] = Sales.index.weekofyear
Sales['SalePerCustomer'] = Sales['Sales']/Sales['Customers']


# Checking data when the stores were closed
Sales_store_closed = Sales[(Sales.Open == 0)]
Sales_store_closed.head()

# Checking days when the stores were closed
Sales_store_closed.hist('DayOfWeek');


# Checking whether there was a school holiday when the store was closed
Sales_store_closed['SchoolHoliday'].value_counts().plot(kind='bar');


# Checking whether there was a state holiday when the store was closed
Sales_store_closed['StateHoliday'].value_counts().plot(kind='bar');


# Checking missing values in Sales set - no missing value
Sales.isnull().sum()

# No. of days with closed stores
Sales[(Sales.Open == 0)].shape[0]


# No. of days when store was opened but zero sales - might be because of external factors or refurbishmnent
Sales[(Sales.Open == 1) & (Sales.Sales == 0)].shape[0]


# Checking store data
store.head()

#缺失数据分析与归算Missing Data Analysis and Imputation

# Checking missing values in store data 
store.isnull().sum()

#只有3个观察到“竞争距离”丢失。这可能是因为有人没有在系统中输入信息。用中间值替换这些缺失的值是安全的。我们不能用同样的方法处理竞争的存在(月份和年份) ，因为它没有意义。最好用0(即最近推出的)来代替。我们还将推销中缺少的值归结为0，因为没有关于推销的信息可用。

# Replacing missing values for Competiton distance with median
store['CompetitionDistance'].fillna(store['CompetitionDistance'].median(), inplace=True)

# No info about other columns - so replcae by 0
store.fillna(0, inplace=True)

# Joining the tables
Sales_store_joined = pd.merge(Sales, store, on='Store', how='inner')
Sales_store_joined.head()


# Distribution of sales and customers across store types
Sales_store_joined.groupby('StoreType')['Customers', 'Sales', 'SalePerCustomer'].sum().sort_values('Sales', ascending=False)

#让我们看看哪些商店关门了或者没有销售。
# Closed and zero-sales obseravtions
Sales_store_joined[(Sales_store_joined.Open ==0) | (Sales_store_joined.Sales==0)].shape

#因此，我们有172,871个观察，当商店关闭或零销售。为了进行数据分析，我们可以删除这些行，但我们仍然可以保留它们用于预测建模，因为我们的模型将能够理解其背后的趋势
# Open & Sales >0 stores
Sales_store_joined_open = Sales_store_joined[~((Sales_store_joined.Open ==0) | (Sales_store_joined.Sales==0))]

#离群值分析Outlier Analysis
plt.figure(figsize=(20,6))
features = ['CompetitionDistance','Sales','Customers','SalePerCustomer']
for i in enumerate(features):
    plt.subplot(1,4,i[0]+1)
    sns.distplot(Sales_store_joined_open[i[1]])
    plt.title(i[1])
plt.show()  


plt.figure(figsize=(20,6))
features = ['CompetitionDistance','Sales','Customers','SalePerCustomer']
for i in enumerate(features):
    plt.subplot(2,2,i[0]+1)
    sns.boxplot(Sales_store_joined_open[i[1]])
    plt.title(i[1])
plt.show() 

#Competition Distance
percentiles = Sales_store_joined_open['CompetitionDistance'].quantile([0.01,0.99]).values
Sales_store_joined_open['CompetitionDistance'][Sales_store_joined_open['CompetitionDistance'] <= percentiles[0]] = percentiles[0]
Sales_store_joined_open['CompetitionDistance'][Sales_store_joined_open['CompetitionDistance'] >= percentiles[1]] = percentiles[1]

#Sales
percentiles = Sales_store_joined_open['Sales'].quantile([0.01,0.99]).values
Sales_store_joined_open['Sales'][Sales_store_joined_open['Sales'] <= percentiles[0]] = percentiles[0]
Sales_store_joined_open['Sales'][Sales_store_joined_open['Sales'] >= percentiles[1]] = percentiles[1]

#Customers
percentiles = Sales_store_joined_open['Customers'].quantile([0.01,0.99]).values
Sales_store_joined_open['Customers'][Sales_store_joined_open['Customers'] <= percentiles[0]] = percentiles[0]
Sales_store_joined_open['Customers'][Sales_store_joined_open['Customers'] >= percentiles[1]] = percentiles[1]

#SalePerCustomer
percentiles = Sales_store_joined_open['SalePerCustomer'].quantile([0.01,0.99]).values
Sales_store_joined_open['SalePerCustomer'][Sales_store_joined_open['SalePerCustomer'] <= percentiles[0]] = percentiles[0]
Sales_store_joined_open['SalePerCustomer'][Sales_store_joined_open['SalePerCustomer'] >= percentiles[1]] = percentiles[1]


plt.figure(figsize=(20,6))
features = ['CompetitionDistance','Sales','Customers','SalePerCustomer']
for i in enumerate(features):
    plt.subplot(1,4,i[0]+1)
    sns.distplot(Sales_store_joined_open[i[1]])
    plt.title(i[1])
plt.show()  


plt.figure(figsize=(20,6))
features = ['CompetitionDistance','Sales','Customers','SalePerCustomer']
for i in enumerate(features):
    plt.subplot(2,2,i[0]+1)
    sns.boxplot(Sales_store_joined_open[i[1]])
    plt.title(i[1])
plt.show()  

#Correlation analysis相关分析
# Correlation
plt.figure(figsize = (20, 10))
sns.heatmap(Sales_store_joined.corr(), annot = True);
#我们可以看到销售量和顾客访问商店之间有很强的正相关性。我们还可以观察到正在进行的促销活动(Promo = 1)与客户数量之间的正相关关系。


#Effect of Promotional events on  Sales促销活动对销售的影响

# Sales trend over the months
sns.factorplot(data = Sales_store_joined_open, x ="Month", y = "Sales", 
               col = 'Promo', # per store type in cols
               hue = 'Promo2',
               row = "Year"
             );
 
#从上述趋势我们可以看出，销售往往在11月和12月飙升。因此，数据中存在季节性因素。

# Sales trend over days
sns.factorplot(data = Sales_store_joined_open, x = "DayOfWeek", y = "Sales", hue = "Promo");


#我们可以从这种趋势中看出，周末(即周六和周日)没有促销活动，这是有道理的，因为商店希望在人们做家务的时候获得最大利润。由于人们在周末购物，销售往往在周日增加。我们还可以看到，最大的销售发生在星期一，当有促销优惠。

#EDA 结论

A)最畅销和拥挤的 StoreType 是 A。

B) StoreType B 每个客户的销售额最高。

C)顾客倾向于在星期一有持续促销活动的时候购买更多的商品，在星期四/星期五根本没有促销活动的时候购买更多的商品。

第二次促销似乎对销售额的增长没有什么帮助。


#Standardize the sales and number of customers variables before modelling在建模前对销售和客户数量变量进行标准化

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
num_col = ['Sales','Customers','SalePerCustomer']
Sales_store_joined_open[num_col] = scaler.fit_transform(Sales_store_joined_open[num_col])
Sales_store_joined_open.head()


#Time Series Analysis & Predictive Modelling时间序列分析与预测模型

#对于时间序列分析，我们将考虑来自每个存储类型 a、 b、 c、 d 的一个存储，它将表示它们各自的组。使用重新采样的方法将数据从几天缩减到几周，以便更清楚地看到当前的趋势，这也是有意义的。

pd.plotting.register_matplotlib_converters()

# Data Preparation: input should be float type
Sales['Sales'] = Sales['Sales'] * 1.0

# Assigning one store from each category
sales_a = Sales[Sales.Store == 2]['Sales']
sales_b = Sales[Sales.Store == 85]['Sales'].sort_index(ascending = True) 
sales_c = Sales[Sales.Store == 1]['Sales']
sales_d = Sales[Sales.Store == 13]['Sales']

f, (ax1, ax2, ax3, ax4) = plt.subplots(4, figsize = (12, 13))

# Trend
sales_a.resample('W').sum().plot(ax = ax1)
sales_b.resample('W').sum().plot(ax = ax2)
sales_c.resample('W').sum().plot(ax = ax3)
sales_d.resample('W').sum().plot(ax = ax4);


#从上面的图中我们可以看到，StoreType A 和 C 的销售趋向于在年底(圣诞季节)达到高峰，然后在假期之后下降。我们无法在 StoreTypeD 中看到类似的趋势，因为该时间段内没有数据可用(商店关闭)。

#Checking The Stationarity in the Time Series
#检验时间序列的平稳性
#时间序列的平稳性

为了使用时间序列预测模型，我们需要确保我们的时间序列数据是平稳的，即常数均值，常数方差和常数协方差与时间。

有2种方法可以检验时间序列的平稳性 a)滚动平均值: 可视化 b) Dicky-Fuller 检验: 统计检验

滚动平均值: 时间序列模型的滚动分析通常用于评估模型随时间的稳定性。该窗口每周滚动一次(在数据上滑动) ，平均值每周取一次。滚动统计是一个可视化测试，在这里我们可以比较原始数据和滚动数据，并检查数据是否是平稳的。

2) Dicky-Fuller 检验: 该检验为我们提供了统计数据，如 p 值，以了解我们是否可以拒绝零假设。零假设是数据不是平稳的，另一个假设是数据是平稳的。如果 p 值小于临界值(例如0.5) ，我们将拒绝零假设，并说数据是平稳的。

# Function to test the stationarity
def test_stationarity(timeseries):
    
    # Determing rolling statistics
    roll_mean = timeseries.rolling(window=7).mean()
    roll_std = timeseries.rolling(window=7).std()

    # Plotting rolling statistics:
    orig = plt.plot(timeseries.resample('W').mean(), color='blue',label='Original')
    mean = plt.plot(roll_mean.resample('W').mean(), color='red', label='Rolling Mean')
    std = plt.plot(roll_std.resample('W').mean(), color='green', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.show(block=False)
    
    # Performing Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    result = adfuller(timeseries, autolag='AIC')
    print('ASales_store_joined_open Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
           print(key, value)
           
           
# Testing stationarity of store type a
test_stationarity(sales_a)


#Testing stationarity of store type b
test_stationarity(sales_b)

#Testing stationarity of store type b
test_stationarity(sales_c)

#Testing stationarity of store type d
test_stationarity(sales_d)

#我们可以从上面的图和统计检验中看出，平均值和变异值并没有随着时间的变化而发生很大的变化，即它们是不变的。因此，我们不需要执行任何转换(当时间序列不是平稳的时候需要)。

#Determining if Time Series has Trend and Seasonality
#确定时间序列是否具有趋势性和季节性


# Plotting seasonality and trend
def plot_timeseries(sales,StoreType):

    fig, axes = plt.subplots(2, 1, sharex=True, sharey=False)
    fig.set_figheight(10)
    fig.set_figwidth(15)

    decomposition= seasonal_decompose(sales, model = 'additive',freq=365)

    estimated_trend = decomposition.trend
    estimated_seasonal = decomposition.seasonal
    estimated_residual = decomposition.resid
    
    axes[1].plot(estimated_seasonal, 'g', label='Seasonality')
    axes[1].legend(loc='upper left');
    
    axes[0].plot(estimated_trend, label='Trend')
    axes[0].legend(loc='upper left');

    plt.title('Decomposition Plots')
    
    # Plotting seasonality and trend for store type a
plot_timeseries(sales_a,'a')

# Plotting seasonality and trend for store type b
plot_timeseries(sales_b,'b')

# Plotting seasonality and trend for store type c
plot_timeseries(sales_c,'c')

# Plotting seasonality and trend for store type d
plot_timeseries(sales_d,'d')


#从上面的图表中，我们可以看出我们的数据中存在着季节性和趋势性。因此，我们将使用考虑这两个因素的预测模型。例如，SARIMAX 和 Prophet。

#Cointegration - Johansen Test
#协整-约翰森检验

#如果两个时间序列同时运动，则可以认为它们是协整的。这意味着它们随着时间的推移受到约束，因此处于平衡状态。

from statsmodels.tsa.vector_ar.vecm import coint_johansen
#Cointegration - Johansen Test
"""
    Johansen cointegration test of the cointegration rank of a VECM

    Parameters
    ----------
    endog : array_like (nobs_tot x neqs)
        Data to test
    det_order : int
        * -1 - no deterministic terms - model1
        * 0 - constant term - model3
        * 1 - linear trend
    k_ar_diff : int, nonnegative
        Number of lagged differences in the model.
"""

def joh_output(res):
    output = pd.DataFrame([res.lr2,res.lr1],
                          index=['max_eig_stat',"trace_stat"])
    print(output.T,'\n')
    print("Critical values(90%, 95%, 99%) of max_eig_stat\n",res.cvm,'\n')
    print("Critical values(90%, 95%, 99%) of trace_stat\n",res.cvt,'\n')
    

joh_model = coint_johansen(Sales_store_joined_open[['Sales','Promo2','Customers','SalePerCustomer']],-1,1) # k_ar_diff +1 = K
joh_output(joh_model)

#注意，痕迹检验和最大特征检验在每个阶段的检验统计量都大于5% 显著性水平的检验统计量。所以矩阵的秩是2 = > 无余积分 = > 差分变量对这4个级数来说是足够的。


#Forecasting a Time Series#预测时间序列

#评估指标:

#测量回归(连续变量)模型的性能有两种常用的度量方法，即 MAE 和 RMSE。

#平均绝对误差: 它是预测值和观测值之间绝对差的平均值。

#平方平均数误差: 它是预测值和观测值之间的平方差的平均值的平方根。

#MAE 更容易理解和解释，但是 RMSE 在不希望出现大错误的情况下工作得很好。这是因为误差在平均之前被平方，从而惩罚了较大的误差。在我们的案例中，RMSE 非常适合，因为我们希望以最小的误差(即惩罚高误差)来预测销售，这样就可以正确地管理库存。

#因此，让我们选择 RMSE 作为度量模型性能的指标。


# SARIMA (Seasonal Autoregressive Integrated Moving Average):
#SARIMA(季节性自回归综合移动平均):
#为了使用这个模型，我们首先需要找出 p，d 和 q.p 的值表示自回归项的数目-因变量的滞后。Q 表示预测方程中移动平均项落后预测误差的个数。D 表示非季节性差异的数量。

利用自相关函数(ACF)和偏自相关(PACF)图求 p、 d 和 q 的值。

ACF-测量时间序列与滞后版本本身之间的相关性。PACF-测量时间序列与滞后版本本身之间的相关性，但在消除了已经通过介入比较解释的变化之后。

P 值是 PACF 的 x 轴上的值，在这里图第一次穿过上面的置信区间。Q 值是 ACF 的 x 轴上的值，在这里图第一次穿过上面的置信区间。

现在，让我们绘制这些图表。


# Autocorrelation function to make ACF and PACF graphs
def auto_corr(sales):
    lag_acf = acf(sales,nlags=30)
    lag_pacf = pacf(sales,nlags=20,method='ols')
  
    plt.subplot(121)
    plt.plot(lag_acf)
    plt.axhline(y=0,linestyle='--',color ='red')
    plt.axhline(y=1.96/np.sqrt(len(sales_a)),linestyle='--',color ='red')
    plt.axhline(y=-1.96/np.sqrt(len(sales_a)),linestyle='--',color ='red')
    plt.title('ACF')
    
    plt.subplot(122)
    plt.plot(lag_pacf)
    plt.axhline(y=0,linestyle='--',color ='red')
    plt.axhline(y=1.96/np.sqrt(len(sales_a)),linestyle='--',color ='red')
    plt.axhline(y=-1.96/np.sqrt(len(sales_a)),linestyle='--',color ='red')
    plt.title('PACF')
    
# ACF and PACF for store type a
auto_corr(sales_a)

# ACF and PACF for store type c
auto_corr(sales_c)

# ACF and PACF for store type d
auto_corr(sales_d)

#上面的图表表明 p = 2和 q = 2，但是让我们做一个网格搜索，看看 p、 q 和 d 的组合给出了最低的赤池信息量准则(AIC，它告诉我们给定一组数据的统计模型的质量。最佳模型使用最少数量的特性来适应数据。

#如果我们要预测每个商店的销售，我们需要考虑整个数据集，而不是每个类别的一个商店。为了理解时间序列数据，我们使用了每个类别的一个存储，但是从现在开始，我们将使用整个数据集进行建模。

# Summing sales on per week basis
Sales_arima = Sales.resample("W").mean() 
Sales_arima = Sales_arima[["Sales"]]
Sales_arima.plot();

#超参数调整 ARIMA 模型

#如上所述，我们有三个参数(p，d 和 q)的 SARIMA 模型。因此，为了选择这些参数的最佳组合，我们将使用网格搜索。最佳的参数组合将得到最低的 AIC 评分。

# Define the p, d and q parameters to take any value between 0 and 3
p = d = q = range(0, 2)

# Generate all different combinations of p, q and q triplets
pdq = list(itertools.product(p, d, q))

# Generate all different combinations of seasonal p, q and q triplets
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]

print('Examples of parameter combinations for Seasonal ARIMA: ')
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))

#现在，让我们遍历这些组合，看看哪一个得到的 AIC 分数最低。

# Determing p,d,q combinations with AIC scores.
for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(Sales_arima,
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)

            results = mod.fit()

            print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
        except:
            continue
            
            
#从上面的网格搜索可以看出，我们的最佳参数组合是 ARIMA (1,1,1) x (0,1,1,12)12-AIC: 1806.2981906704658。让我们在模型中使用它。

#拟合模型-使用上面调整的超参数
# Fitting the data to SARIMA model 
model_sarima = sm.tsa.statespace.SARIMAX(Sales_arima,
                                order=(1, 1, 1),
                                seasonal_order=(0, 1, 1, 12),
                                enforce_stationarity=False,
                                enforce_invertibility=False)

results_sarima = model_sarima.fit()

print(results_sarima.summary().tables[1])

# Checking diagnostic plots
results_sarima.plot_diagnostics(figsize=(10, 10))
plt.show()

#我们可以从上面的“直方图加估计密度”图中看到，我们的 KDE (核密度估计)图紧跟 N (0,1)正态分布图。正态 Q-Q 曲线表明，残差的有序分布服从与正态分布相似的分布。因此，我们的模型似乎是相当不错的。

#标准化残差图表明，该区域不存在明显的季节性变化趋势，并由相关图(自相关)图证实了这一点。相关图告诉我们，时间序列残差与其自身的滞后版本相关性很低。

#最终 ARIMA 解决方案:

#我们使用网格搜索尝试了不同的参数组合，找到了最佳参数: ARIMA (1,1,1) x (0,1,1,12)12-AIC: 1806.29。

#Sales Prediction for the next 6 weeks
#未来6周的销售预测

# Model Prediction and validation 
# Predictions are performed for the 11th Jan' 2015 onwards of the Sales data.
from math import sqrt
pred = results_sarima.get_prediction(start=pd.to_datetime('2015-01-11'), dynamic = False) 

# Get confidence intervals of forecasts
pred_ci = pred.conf_int() 

ax = Sales_arima["2014":].plot(label = "observed", figsize=(15, 7))
pred.predicted_mean.plot(ax = ax, label = "One-step ahead Forecast", alpha = 1)
ax.fill_between(pred_ci.index, 
                pred_ci.iloc[:, 0], 
                pred_ci.iloc[:, 1], 
                color = "k", alpha = 0.05)

ax.set_xlabel("Date")
ax.set_ylabel("Sales")

plt.legend
plt.show()

Sales_arima_forecasted = pred.predicted_mean
Sales_arima_truth = Sales_arima["2015-01-11":]

# Calculating the error
rms_arima = sqrt(mean_squared_error(Sales_arima_truth, Sales_arima_forecasted))
print("Root Mean Squared Error: ", rms_arima)
