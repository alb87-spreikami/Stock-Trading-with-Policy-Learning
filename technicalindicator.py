# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 13:16:16 2024

@author: ANGGORO.BUDIONO
"""

import pandas as pd
import pandas_ta as ta
import datetime as dt 
import yfinance as yf
import numpy as np



class TechnicalData():
    def __init__(self,ticker,s_date,
                 e_date=dt.date.today()):
        self.df=yf.download(ticker,start=s_date,end=e_date)
        self.df_market=yf.download('^JKSE',start=s_date,end=e_date)
        self.df_price=self.df['Close']
          
    def ad(self): #Advance-Decline Volume Percent
        #breadth indicator that measures the percentage of Net Advancing 
        #Volume for a particular group of stocks
        return self.df_market.ta.ad()
    
    def ma(self,span=20): #moving average
        return self.df_price.rolling(span).mean()
    
    def macd(self): #moving average convergen divergen 12-26
        #to help investors identify price trends, measure trend momentum, 
        #and identify market entry points for buying or selling
        return self.df.ta.macd() 
        
    def adx(self): # average directional index
        # to determine the strength of a price trend.
        # Trading in the direction of a strong trend reduces risk 
        # and increases profit potential.
        return self.df.ta.adx()
    
    def rsi(self): #relative strength index
        #measures the speed and magnitude of a security's recent price 
        #changes to evaluate overvalued or undervalued conditions 
        #in the price of that security.
        return self.df.ta.rsi()    
    
    def stochastic(self): #stochastic 
        #used to generate overbought and oversold trading signals.
        return self.df.ta.stoch()
    
    def bollinger(self): #bolingerbands
        #helps gauge the volatility of stocks to determine 
        #if they are over- or undervalued
        return self.df.ta.bbands()
    
    def ad_line(self): #advance-decline line
        #plots the difference between the number of advancing and 
        #declining stocks on a daily basis.
        return self.df.ta.ad()
    
    def merge_dataframe(self):
        df=pd.DataFrame()
        df['Market']=self.ad()
        df['Price']=self.df_price
        df['MA']=self.ma()
        df[['MACD','MACDh','MACDs']]=self.macd()
        df[['ADX','DMP','DMN']]=self.adx()
        df['RSI']=self.rsi()
        df[['STOCHk','STOCHd']]=self.stochastic()
        df[['BBL','BBM','BBU','BBB','BBP']]=self.bollinger()
        df['AD_LINE']=self.ad_line()
        return df.dropna(axis=0)

    @property
    def normalize(self):
        df=self.merge_dataframe()
        mean=df.mean(axis=0)
        std_dev=df.std(axis=0)
        
        df_norm=pd.DataFrame()
        for col in df.columns:
            df_norm[col]=(df[col]-mean[col])/df[col].std()
        
        return {'data':df_norm,
                'info_mean':pd.DataFrame(mean,columns=['info_mean']),
                'info_std':pd.DataFrame(std_dev,columns=['info_std'])}
    @property
    def date(self):
        df=self.normalize['data'].index
        nump=np.array(df)
        return nump.astype(dtype='datetime64[D]')
    
    @property
    def price(self):
        index=self.merge_dataframe().index
        df=self.df_price.loc[index]
        return np.array(df)
       
    
    def __call__(self):
        df=self.normalize['data']
        return np.array(df)
    
def roll_window(a,b,c,window):
    # a--> data set
    # b--> data date
    w=window
    shape=a.shape
    shape=(shape[0]-window,window,shape[1])
    dataset=np.zeros(shape=shape)
    
    for i in range (len(dataset)):
        dataset[i,:,:]=a[i:i+w,:]
    
    datadate=b[-(shape[0]):]
    
    dataprice=c[-(shape[0]):]
    
    return dataset, datadate, dataprice


if __name__=='__main__':
        
    data=TechnicalData(ticker="ADRO.JK",
                       s_date=dt.date(2020,1,1))
    data_set=data()
    data_time=data.date 
    data_price=data.price
    data_mean=data.normalize['info_mean']
    data_std=data.normalize['info_std']
      