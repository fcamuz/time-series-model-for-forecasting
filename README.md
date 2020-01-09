
![Download Dataset](https://github.com/fcamuz/time-series-model-for-forecasting/blob/master/images/Slide1.png)

# Time Series Model for Real Estate Investment Company

This model is for suggesting the best 5 zipcodes to invest for a Real Estate Investment Company. Conditions are listed below. 

- Company would like to invest in Texas
- The range of the investment is not defined, it could be any amount
- Company would like to do investment for at least 3 years span. 

## Dataset

Zillow dataset which includes 22 years of monthly data for average real estate price in each zip code. The data is available from 1996 – 2018. 

![Download Dataset](https://github.com/fcamuz/time-series-model-for-forecasting/blob/master/zillow_data.csv)

## Methodology 

I have used following aproach to complete this project
- EDA
- ARIMA Modeling 
- Facebook Prophet Modelling 

## Presentation 

You may watch the presentation for non-technical audience from the link below.

https://www.youtube.com/watch?v=M_ue4RZJmTQ

Link for the pdf is below.

![Real_Estate_Investment.pdf](https://github.com/fcamuz/time-series-model-for-forecasting/blob/master/Real_Estate_Investment.pdf)
 
 ## Table of Contents
- Step 1: Load the Data
- Step 2: Feature Engineering
- Step 3: EDA and Visualization
- Step 4. Investigating the components of time series data for TOP 10 Zipcodes
- Step 5: ARIMA Modelling
- Step 6: Performence of ARIMA Model
- Step 7: Modelling with Facebook Prophet
- Step 8: Conclusion

After dealing with null values, I have created a criteria.I used price increase and the increase rate (percentage of the increase) when filtering the data. I cross refenced them by creating a new column as "criteria" that stores increase rate X price increase for the last 22 years.

```python
df_full['increase'] =(df_full['2018-04'] - df_full['1996-04'])  #calculate price change and the percentage of the increase
df_full['increase_rate'] = df_full['increase']/df_full['1996-04'] 
df_full['criteria']= df_full['increase']*df_full['increase_rate']
```

I have created a set of functions to convert dataframe to time series. I will be using this one several times. 

```python
def drop_cols(df):
    ts=df
    for col in ts.columns:
        if col in ('RegionID', 'CountyName','Metro','SizeRank','State','increase','increase_rate','criteria'):
            ts=ts.drop(col, axis=1)
    return ts
        

def convert_to_ts (df):   
    ts=drop_cols(df)  #drop columns other than price data
    ts=ts.T     # Transpose the dataframe
    ts.index=pd.to_datetime(ts.index)  #change index type to timestamp
    ts=ts.astype(int)  
    return ts
```
## Several findings from EDA 

### Top 10 states in US for average price increase

![Download Dataset](https://github.com/fcamuz/time-series-model-for-forecasting/blob/master/images/download1.png)

### Boxplot for top 10 states' increase rates

![Download Dataset](https://github.com/fcamuz/time-series-model-for-forecasting/blob/master/images/download2.png)



![Download Dataset](https://github.com/fcamuz/time-series-model-for-forecasting/blob/master/images/download3.png)

![Download Dataset](https://github.com/fcamuz/time-series-model-for-forecasting/blob/master/images/download4.png)

![Download Dataset](https://github.com/fcamuz/time-series-model-for-forecasting/blob/master/images/download5.png)

![Download Dataset](https://github.com/fcamuz/time-series-model-for-forecasting/blob/master/images/download6.png)


# Step 4: Investigating the components of time series data for TOP 10 Zipcodes

Visualisation can give us a great deal of information about time series. 
I will use plots to check; 
- trend, seasonality
- detrending with .diff()
- autocorrelation
- partial autocorrelation
- partial autocorrelation after detrending

```python
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
def rolmean_rolstd_plot(df, zipcode):
    # check rolmean , rolstd plots for trend
    rolmean = df.rolling(window = 4).mean()
    rolstd = df.rolling(window = 4).std()
    fig = plt.figure(figsize=(12,4))
    orig = plt.plot(df, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='green', label='Rolling Std')
    plt.legend(loc='best')
    plt.title(zipcode+' Parameters: Rolling Mean & Standard Deviation')
    plt.show(block=False)
    
    # check first degree diff() plot for detrending
    ts_lag_1 = df.diff(1).plot(legend=False, label='1 lag difference',linewidth=1, c='r')
    plt.xlabel('1 lag difference')
    rcParams['figure.figsize'] = 10,1
    
    #ckeck Autocorrelation , Partial Autocorrelation plots
    ts_acf_plot = plot_acf(df, title='ACF')
    ts_pacf_plot = plot_pacf(df, title='PACF')

for col in ts_top10_zip.columns:
     rolmean_rolstd_plot(ts[col],str(col))
```

With this I got parameters visualizations for each zipcode like one below

![parameters ](https://github.com/fcamuz/time-series-model-for-forecasting/blob/master/images/params.png)

### Dickey Fuller Test for Stationarity Check

```python
    for i in ts.columns:
    X = ts[i]
    result = adfuller(X)
    print(i)
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))
```

None of the columns in the time series data is stationary. So I will use ARIMA Model since it can handle non-stationary data. I will do grid search to figure out best combinations for the necessary parameters.
 
 # Step 5:ARIMA Modelling

 I have done modelling with Arima and filtered the forecasted data by the growth rate. I used top 5 zipcodes series, their best parameter combinations to run the model one more time and create plot for the historical data and forcasted data. It also shades the predictions confidence interval. At the bottom of each plot, there is a detailed report for that zipcode. Last set of plots are the plot diagnostics with results calculated in the model. I got this report like one below for 5 zipcodes


![parameters ](https://github.com/fcamuz/time-series-model-for-forecasting/blob/master/images/download7.png)

Zipcode : 75205
City :  University Park
State :  TX
Investment($) : 1268600
********
Predicted price in 3 years : 1475245.0
Total Increase in 3 years : 206644.6278622318
Increase rate in 3 years (%) : 0.16
Possible range : 1269124.0  -  1681366.0
********
Predicted price in 5 years : 1614455.0
Total Increase in 5 years : 345855.4514563428
Increase rate in 5 years (%) : 0.27
Possible range : 1231511.0  -  1997400.0

![parameters ](https://github.com/fcamuz/time-series-model-for-forecasting/blob/master/images/download8.png)

### 3 Assumptions for the model have met

From the diognostics plot, we can see if the assumptions met to make sure the model works properly.Residuals resamble white noise and it's KDE has normal distribution with mean 0 and standard deviation of 1.Also, the qq-plot shows that the ordered distribution of residuals mostly follows the linear trend of the samples taken from a standard normal distribution These indicate that the residuals are normally distributed. Correlogram shows that the time series residuals have low correlation with lagged versions of itself.

According to the plots above, all columns seem met the assumptions. KDE histogram and Q-Q plot indicates some outliers in a few zipcodes, that makes the confidence interval wider.But data seem too be ok for this model mostly.

# Step 6: Performence of ARIMA Model

I visually checked the performence of ARIMA forecasting for selected Zipcode. 

```python
from sklearn.metrics import mean_squared_error
pdq_new=(3,1,1)
pdqs_new=(3,1,1,12)

for i in zipcodes:
    X = ts[i]
    size = int(len(X) * 0.66)
    train, test = X[0:size], X[size:len(X)]
    history = [x for x in train]
    predictions = list()

    for t in range(len(test)):
        model = sm.tsa.statespace.SARIMAX(history,
                                        order=pdq_new,
                                        seasonal_order=pdqs_new,
                                        enforce_stationarity=False,
                                        enforce_invertibility=False)
        model_fit = model.fit()
        output = model_fit.forecast()
        yhat = output[0]
        predictions.append(yhat)
        obs = test[t]
        history.append(obs)
        #print('predicted=%f, expected=%f' % (yhat, obs))

    error = np.sqrt(mean_squared_error(test, predictions))
    print("Zipcode :", i )
    print('Test RMSE: %.3f' % error)
    print('Standart Deviation:' , np.std(test))
    print(model_fit.summary())
    
    plt.figure(figsize=(14,5))
    #plt.plot(train, label='Train')
    plt.plot(test, color='red', label="Test")
    plt.plot(test.index, predictions, color='blue', label="Predicted", alpha=0.5)
    plt.legend()
    plt.show()  
```
![parameters ](https://github.com/fcamuz/time-series-model-for-forecasting/blob/master/images/download9.png)

```
Zipcode : 78705
Test MSE: 2113.006
Standart Deviation: 44787.79589387354
                                 Statespace Model Results                                 
==========================================================================================
Dep. Variable:                                  y   No. Observations:                  264
Model:             SARIMAX(3, 1, 1)x(3, 1, 1, 12)   Log Likelihood               -1828.146
Date:                            Fri, 25 Oct 2019   AIC                           3674.293
Time:                                    11:48:57   BIC                           3704.502
Sample:                                         0   HQIC                          3686.503
                                            - 264                                         
Covariance Type:                              opg                                         
==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
ar.L1          1.1298      0.136      8.315      0.000       0.863       1.396
ar.L2         -0.5720      0.159     -3.609      0.000      -0.883      -0.261
ar.L3          0.2962      0.074      3.977      0.000       0.150       0.442
ma.L1          0.4870      0.122      4.008      0.000       0.249       0.725
ar.S.L12      -1.2305      0.333     -3.691      0.000      -1.884      -0.577
ar.S.L24      -0.7305      0.290     -2.520      0.012      -1.299      -0.162
ar.S.L36      -0.1116      0.121     -0.925      0.355      -0.348       0.125
ma.S.L12       0.2809      0.334      0.840      0.401      -0.375       0.937
sigma2      2.087e+06   1.55e+05     13.464      0.000    1.78e+06    2.39e+06
===================================================================================
Ljung-Box (Q):                       77.11   Jarque-Bera (JB):               173.33
Prob(Q):                              0.00   Prob(JB):                         0.00
Heteroskedasticity (H):               7.45   Skew:                             0.27
Prob(H) (two-sided):                  0.00   Kurtosis:                         7.40
===================================================================================

Warnings:
[1] Covariance matrix calculated using the outer product of gradients (complex-step).
```


### Result Tables

Report for forecasting 3 years

![parameters ](https://github.com/fcamuz/time-series-model-for-forecasting/blob/master/images/report1.png)

Report for forecasting 5 years

![parameters ](https://github.com/fcamuz/time-series-model-for-forecasting/blob/master/images/report2.png)


# Step 7: Modelling with Facebook Prophet

```python
from fbprophet import Prophet as proph
def prophet_prediction(df, topzips):
    dscol = [x for x in df.ds]
    
    values = []
    
    for i in topzips:
        ycol = [x for x in df[i]]
        
        ts = pd.DataFrame()
        ts['ds'] = dscol
        ts['y'] = ycol
        
        Model = proph(interval_width=0.95)
        Model.fit(ts)
        
        future_dates = Model.make_future_dataframe(periods=36, freq='MS')
        forecast = Model.predict(future_dates)
        
        forecasted_data=forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
        
        increase_3_yrs = round(forecast['yhat'][300] - forecast['yhat'][264], 2)
        increase_rate_3 = round(increase_3_yrs / forecast['yhat'][264], 2)
                                                        
        values.append((i, df[i][-1], increase_3_yrs, increase_rate_3))
        
        
        Model.plot(forecast, uncertainty=True)
        plt.title(i)
        plt.ylabel('Real Estate Value')
        plt.xlabel('Year')
        Model.plot_components(forecast)
        plt.title(i)
        plt.ylabel('Real Estate Value')
        plt.xlabel('Yearly Trend');
        
        
       
        print("Zipcode :" , i)
        print("Investment($) :", df[i][-1] )
        print("********")
        print("Predicted price in 3 years :",round(forecast['yhat'][300] ))
        print('Increase rate in 3 years (%) :',  increase_rate_3 )
        print("-----------------------------------------------------------------------------")    
        
    return values , forecasted_data
```
```
Zipcode : 75205
Investment($) : 1268600
********
Predicted price in 3 years : 1534000.0
Increase rate in 3 years (%) : 0.2
-----------------------------------------------------------------------------
```

![parameters ](https://github.com/fcamuz/time-series-model-for-forecasting/blob/master/images/download10.png)


# Step 8: Conclusion

![parameters ](https://github.com/fcamuz/time-series-model-for-forecasting/blob/master/images/Slide5.png)

  - **The best zipcodes to invest in Texas are 75205, 78703, 78752, 78705, 78702**
  
  - **The best range for investment is 100K-500K in terms of the rate of price increase. These are mostly the affordable residential houses**
  
  - **78703, 78752, 78705, 78702 are in Austin, the capitol city of Texas. It is not surprising because Austin has been one of the fastest growing city in US. Austin Metro Area has the fastest growing real estate market in Texas. All major high tech companies are moving there including Google and Apple. DELL and IBM has been born in Austin as well. It is already been called Silicon Hills for quite a while. Beacuse of the numerous high tech companies openning branches in Austin last 10 years, the increasing trend is expected to be long term. So, the longer investment horizon could be rewarding, despite the increased risk on the model's forecast.**
  
  - **75205 is in University Park. It is a decidedly white-collar city, with fully 97.43% of the workforce employed in white-collar jobs, well above the national average. University Park home prices are not only among the most expensive in Texas, but University Park real estate also consistently ranks among the most expensive in America.**
  
  