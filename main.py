import streamlit as st
import pandas as pd
import os, json
import pandas as pd
import seaborn as sns
import joblib
import numpy as np
import matplotlib.pyplot as plt
import ast
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.cluster import DBSCAN
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from langchain.llms import OpenAI
import os
from langchain.chat_models import ChatOpenAI
import subprocess
from langchain.schema import HumanMessage, SystemMessage, AIMessage
os.environ["OPEN_API_KEY"] = "YOUR_API_KEY_HERE"
import random
from langchain.prompts import SystemMessagePromptTemplate, PromptTemplate
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, r2_score, recall_score, confusion_matrix,f1_score, classification_report,precision_score

st.title("FinXP")


llm = OpenAI(openai_api_key=os.environ["OPEN_API_KEY"], temperature=0.6)
chatllm = ChatOpenAI(openai_api_key=os.environ["OPEN_API_KEY"], temperature=0.6, model='gpt-3.5-turbo')
pt = PromptTemplate(input_variables = ['ata','tf','sg','se','st','ti'],
                                          template = """
                                          You are an Expenditure Monitoring Agent. You are given a few variables:
                                          Average Transaction Amount Av = {ata}
                                          Transaction Frequency = {tf}
                                          Amount Spent on Groceries = {sg}
                                          Amount Spent on Entertainment = {se}
                                          Amount Spent on Travel = {st}
                                          Total Income = {ti}. Give a short 3-5 line answer about how this user can
                                          improve their spending habits and how to better improve thier expenditure.
                                          Directly generate suggestions and ideas. The problem statement is optimising
                                          expenditures. Amount spent should not increase (preferrably the user should get similar
                                          experience for lesser expenditure).
                                          """)
systemtemplate = SystemMessagePromptTemplate(prompt=pt)

hm = "Give me suggestions about my expenditure. Use numerical suggestions for the mentioned parameters" 
def generate(chatllm,systemmessage,hm):
    ans = chatllm([
    SystemMessage(content=systemmessage.content),
    HumanMessage(content=hm)
    ])
    return ans









def trans(df):
  bdf = df.select_dtypes(include = ['bool'])
  bdf = bdf.applymap(lambda x: 1 if x>0 else 0)
  df.RetirementSaving = bdf['RetirementSaving']
  df.HouseBuying = bdf['HouseBuying']
  df.EmergencyFund = bdf['EmergencyFund']
  df = df.drop(["CustomerID"],axis=1)
  ndf = df.select_dtypes(include = ['float','int'])
  ncol = ndf.columns
  mapping_risk = {'Low': 0, 'Medium': 0.5, 'High': 1}
  mapping_inv_dur = {'Long-term': 1, 'Short-term': 0}
  mapping_gender = {'F': 1, 'M': 0}

  # Replace values
  df['RiskTolerance'] = df['RiskTolerance'].map(mapping_risk)
  df['InvestmentDuration'] = df['InvestmentDuration'].map(mapping_inv_dur)
  df['Gender']=df['Gender'].map(mapping_gender)
  df['PreferredInvestmentType_Bonds'] = 0
  df['PreferredInvestmentType_Mutual Funds'] = 0
  df['PreferredInvestmentType_Real Estate'] = 0
  df['PreferredInvestmentType_Stocks'] = 0
  x = df['PreferredInvestmentType']
  sx = 'PreferredInvestmentType_'+x
  df[sx] = 1
  df = df.drop(['PreferredInvestmentType'],axis=1)
  df['Occupation_Clerk'] = 0
  df['Occupation_Doctor'] = 0
  df['Occupation_Engineer'] = 0
  df['Occupation_Lawyer'] = 0
  df['Occupation_Nurse'] = 0
  df['Occupation_Teacher'] = 0
  y = df['Occupation']
  sy = 'Occupation_'+y
  df[sy] = 1
  df = df.drop(["Occupation"],axis=1)

  invcol = ['Age', 'Gender',
        'IncomeInvested', 'InvestmentDuration', 'RiskTolerance',
        'RetirementSaving', 'HouseBuying', 'EmergencyFund',
        'TotalIncome', 'PreferredInvestmentType_Bonds',
        'PreferredInvestmentType_Mutual Funds',
        'PreferredInvestmentType_Real Estate',
        'PreferredInvestmentType_Stocks']
  spcol = ['Age', 'Gender', 'AvgTransactionAmount', 'TransactionFrequency',
        'SpendingOnGroceries', 'SpendingOnEntertainment', 'SpendingOnTravel',
          'TotalIncome', 'Occupation_Clerk', 'Occupation_Doctor',
        'Occupation_Engineer', 'Occupation_Lawyer', 'Occupation_Nurse',
        'Occupation_Teacher']
  regcol = ['Age', 'Gender', 'AvgTransactionAmount', 'TransactionFrequency',
       'SpendingOnGroceries', 'SpendingOnEntertainment', 'SpendingOnTravel',
       'IncomeInvested', 'InvestmentDuration', 'RiskTolerance',
       'RetirementSaving', 'HouseBuying', 'EmergencyFund',
       'TotalIncome', 'Occupation_Clerk', 'Occupation_Doctor',
       'Occupation_Engineer', 'Occupation_Lawyer', 'Occupation_Nurse',
       'Occupation_Teacher', 'PreferredInvestmentType_Bonds',
       'PreferredInvestmentType_Mutual Funds',
       'PreferredInvestmentType_Real Estate',
       'PreferredInvestmentType_Stocks']
  
  rdf = df[regcol]
  idf = df[invcol]
  sdf = df[spcol]
  return rdf,idf,sdf



df = pd.read_csv("https://raw.githubusercontent.com/PranayaLV/Capstone/main/Finance/Final%20Data/Financial_Data_OG.csv")
#invkm = joblib.load("https://raw.githubusercontent.com/PranayaLV/Capstone/blob/main/Finance/Models/invmodel.joblib")
#spdkm = joblib.load("https://raw.githubusercontent/PranayaLV/Capstone/blob/main/Finance/Models/spdmodel.joblib")
#regmodel = joblib.load("https://raw.githubusercontent/PranayaLV/Capstone/blob/main/Finance/Models/regmodel.joblib")
invkm = joblib.load("C:/Users/sajjatarun.teja.lv/Downloads/invmodel.joblib")
spdkm = joblib.load("C:/Users/sajjatarun.teja.lv/Downloads/spdmodel.joblib")
regmodel = joblib.load("C:/Users/sajjatarun.teja.lv/Downloads/regmodel.joblib")
existdf = pd.read_csv("https://raw.githubusercontent.com/PranayaLV/Capstone/main/Finance/Final%20Data/main_with_clusters_invs.csv")
investments = pd.read_csv("https://raw.githubusercontent.com/PranayaLV/Capstone/main/Finance/Final%20Data/Investments_with_cat.csv")
simdf = pd.read_csv("https://raw.githubusercontent.com/PranayaLV/Capstone/main/Finance/Final%20Data/simdf.csv")
main_data = pd.read_csv("https://raw.githubusercontent.com/PranayaLV/Capstone/main/Finance/Final%20Data/main_with_clusters_invs.csv")
main_data["CurrentInvestments"] = main_data["CurrentInvestments"].apply(ast.literal_eval)
  
def sim_inv(inv_id):
  sim_values = pd.DataFrame(simdf.loc[inv_id].sort_values(ascending=False))
  top_5_similar_inv = list(sim_values.iloc[1:6].index)  # Exclude self, which will have similarity=1
  return top_5_similar_inv
def similar_ids_to_customer(customer_id):
  value = main_data.loc[main_data['CustomerID'] == customer_id, 'CurrentInvestments'].values[0]
  inv_freq = dict()
  lst = []
  for i in value:
    temp = sim_inv(i)
    for x in temp:
      if x not in value:
        if x in inv_freq:
          inv_freq[x] = inv_freq[x] + 1
        else:
          inv_freq[x] = 1
  sorted_inv_freq = sorted(inv_freq.items(), key=lambda a: a[1], reverse=True)
  top_3 = sorted_inv_freq[4:7]
  for investment, frequency in top_3:
    lst.append(investment)
  return lst
idf = pd.read_csv("https://raw.githubusercontent.com/PranayaLV/Capstone/main/Finance/Final%20Data/idf.csv")
idf["CurrentInvestments"] = main_data["CurrentInvestments"]
idf["CustomerID"] = main_data["CustomerID"]
idf.drop("Unnamed: 0",axis=1,inplace=True)

def most_popular_in_cluster(cluster):
  df = idf[idf["Invest_class"] == cluster]
  df_pop = df["CurrentInvestments"]
  inv_freq = dict()
  lst = []
  for inv in df_pop:
    for i in inv:
      if i in inv_freq:
        inv_freq[i] = inv_freq[i] + 1
      else:
        inv_freq[i] = 1
  sorted_inv_freq = sorted(inv_freq.items(), key=lambda a: a[1], reverse=True)
  top_3 = sorted_inv_freq[:5]
  for investment, frequency in top_3:
    print(investment,frequency)
  return top_3

def cluster_sim(cluster):
  new_idfp = idf[idf["Invest_class"]==cluster]
  customer_ids = new_idfp["CustomerID"]
#   store_inv = new_idfp[["CustomerID","CurrentInvestments"]]
  new_idfp = new_idfp.drop(columns = ["Invest_class","CustomerID","CurrentInvestments"])
  scaler = StandardScaler()
  scaled_df = pd.DataFrame(scaler.fit_transform(new_idfp), columns=new_idfp.columns)
  temp = scaled_df.drop(columns = ["Age","Gender","PreferredInvestmentType_Stocks"])
  encoded_vectors = temp.to_numpy()
  similarity_matrix = cosine_similarity(encoded_vectors)
  similarity_df = pd.DataFrame(similarity_matrix, index=customer_ids, columns=customer_ids)
  return similarity_df

idf_0 = cluster_sim(0)
idf_1 = cluster_sim(1)
idf_2 = cluster_sim(2)

# Function to find top 5 most similar users for each customer
def find_top_similar_users(customer_id):
  cluster = idf.loc[idf['CustomerID'] == customer_id, 'Invest_class'].values[0]
  current_user_investments = idf.loc[idf['CustomerID'] == customer_id, 'CurrentInvestments'].values[0]
  # current_user_investments = set(current_user_investments)
  if cluster == 0:
    similarity_df = idf_0
  elif cluster == 1:
    similarity_df = idf_1
  elif  cluster == 2:
    similarity_df = idf_2

  sim_values = pd.DataFrame(similarity_df.loc[customer_id].sort_values(ascending=False))
  top_5_similar_users = list(sim_values.iloc[1:6].index)  # Exclude self, which will have similarity=1
  inv_freq = dict()
  lst = []
  for user_id in top_5_similar_users:
    investments = idf.loc[idf['CustomerID'] == user_id, 'CurrentInvestments'].values[0]
    for i in investments:
      if i not in current_user_investments:
        if i in inv_freq.keys():
          inv_freq[i] = inv_freq[i] + 1
        else:
          inv_freq[i] = 1
  sorted_inv_freq = sorted(inv_freq.items(), key=lambda a: a[1], reverse=True)
  top_3 = sorted_inv_freq[:3]
  for investment, frequency in top_3:
    lst.append(investment)
  return lst

def dspc(car,c):
  res = ''
  if car == 'i':
    if c==0:
      res = 'Low Risk Investor'
    elif c==1:
      res = 'High Risk Investor'
    elif c==2:
      res = 'Dynamic Investor'

  elif car == 's':
    if c==0:
      res = 'Lavish Spender'
    elif c==1:
      res = 'Moderate Spender'
    elif c==2:
      res = 'Low Spender'
  
  return res





def cistr(df,c):
  xx = []
  a = df[c]
  for x in a:
    x = x[1:-1]
    x = x.split(", ")
    x = [int(i) for i in x]
    xx.append(x)
  df[c] = xx
  return df
def getnames(ids):
   idnames = []
   for i in ids:
    idnames.append(investments.loc[int(i),"Investment"])
   return idnames

option = st.radio("Choose an option:", ("Existing User", "New User"))
if option == "Existing User":
    cid = st.text_input("Enter the Customer ID")
    if cid:  # Check if cid is not empty

        filtered_df = df[df['CustomerID'] == int(cid)]
        st.dataframe(filtered_df)
        fdf2 = existdf[existdf["CustomerID"] == int(cid)]
        col1, col2, col3 = st.columns(3)
        col1.metric("Credit Score",fdf2['CreditScore'])
        col2.metric("Investment Cluster",dspc('i',int(fdf2['Invest_class'])))
        col3.metric("Spending Cluster",dspc('s',int(fdf2['Spend_class'])))
        fdf2 = cistr(fdf2,"CurrentInvestments")
        ivids = []
        for i in fdf2['CurrentInvestments'].values[0]:
            ivids.append(i)
        ivnames = getnames(ivids)
        st.subheader("Current Investments")
        st.dataframe(ivnames)
        st.subheader("Investments Similar to your choices")
        sivids = similar_ids_to_customer(int(cid))
        sinames = getnames(sivids)
        st.dataframe(sinames)
        st.subheader("People have also Invested in")
        suivids = find_top_similar_users(int(cid))
        sunames = getnames(suivids)
        st.dataframe(sunames)
        st.subheader("Most popular Investments in you Segment")
        ct = most_popular_in_cluster(int(fdf2['Invest_class']))
        ctnames = getnames([i[0] for i in ct])
        st.dataframe(ctnames)
        bt = st.button('Generate')
        if bt:
          systemmessage = systemtemplate.format(ata = fdf2['AvgTransactionAmount'],
                                                tf = fdf2['TransactionFrequency'],
                                                sg = fdf2['SpendingOnGroceries'],
                                                se = fdf2['SpendingOnEntertainment'],
                                                st = fdf2['SpendingOnTravel'],
                                                ti = fdf2['TotalIncome'])
          ans = generate(chatllm,systemmessage,hm)
          st.write(ans.content)
        



        


elif option == "New User":
    form1 = st.form(key='my_form')
    cid = len(df)+1
    usrage = form1.number_input("Please Enter your Age")
    gender_mapping = {"Male": "M", "Female": "F"}
    usrgender_display = form1.radio("Please select your Gender", ["Male", "Female"])
    usrgender = gender_mapping[usrgender_display]
    usroccupation = form1.selectbox("Select your Occupation",
                                    ('Doctor', 'Teacher', 'Clerk', 'Nurse', 'Lawyer', 'Engineer'))
    usravgtransactionamount = form1.number_input("Please Enter your Average Transaction Amount")
    usrtransactionfrequency = form1.number_input("Please Enter your Transaction Frequency")
    usrspendingongroceries = form1.number_input("Please Enter your Expenditure on Groceries")
    usrspendingonentertainment = form1.number_input("Please Enter your Expenditure on Entertainment")
    usrspendingontravel = form1.number_input("Please Enter your Expenditure on Travel")
    usrprefferedinvestment = form1.selectbox("Select your Preferred Investement Type",
                                            ('Real Estate', 'Mutual Funds', 'Bonds', 'Stocks'))
    usrincomeinvested = form1.number_input("Please the amount of income you've invested")
    usrinvestmentduration = form1.selectbox("Select your Preferred Investement Duration",
                                            ('Long-term', 'Short-term'))
    usrrisktolerance = form1.radio("Please select your Risk Tolerance",["High","Low","Medium"])
    form1.write("Toggle the funds if you do have them")
    usrretirementsaving = form1.toggle("Retirement Saving")
    usrhousebuying = form1.toggle("House Fund")
    usremergencyfund = form1.toggle("Emergency Fund")
    usrtotalincome = form1.number_input("Please Enter your Total Income")

    submit_button = form1.form_submit_button(label='Submit')
    newusr = pd.DataFrame({})
    if submit_button:  # Check if submit button is clicked and cid is not empty
        newusr = pd.DataFrame({
        'CustomerID': cid,
        'Age': [usrage], 
        'Gender': [usrgender], 
        'Occupation': [usroccupation], 
        'AvgTransactionAmount': [usravgtransactionamount],
        'TransactionFrequency': [usrtransactionfrequency], 
        'SpendingOnGroceries': [usrspendingongroceries],
        'SpendingOnEntertainment': [usrspendingonentertainment], 
        'SpendingOnTravel': [usrspendingontravel],
        'PreferredInvestmentType': [usrprefferedinvestment], 
        'IncomeInvested': [usrincomeinvested], 
        'InvestmentDuration': [usrinvestmentduration],
        'RiskTolerance': [usrrisktolerance], 
        'RetirementSaving': [usrretirementsaving], 
        'HouseBuying': [usrhousebuying], 
        'EmergencyFund': [usremergencyfund],
        'TotalIncome': [usrtotalincome]})
        st.dataframe(newusr)
        rrdf,iidf,ssdf = trans(newusr)
        cscore = regmodel.predict(rrdf)[0]
        iidf.insert(8,'CreditScore',cscore)
        ssdf.insert(14,'CreditScore',cscore)
        iclass = invkm.predict(iidf)[0]
        sclass = spdkm.predict(ssdf)[0]

        newusr['CreditScore'] = int(cscore)
        newusr['Invest_class'] = iclass
        newusr['Spend_class'] = sclass
        col1, col2, col3 = st.columns(3)
        col1.metric("Credit Score",newusr['CreditScore'])
        col2.metric("Investment Cluster",dspc('i',int(newusr['Invest_class'])))
        col3.metric("Spending Cluster",dspc('s',int(newusr['Spend_class'])))
        st.subheader("Most popular Investments in you Segment")
        ct = most_popular_in_cluster(int(newusr['Invest_class']))
        ctnames = getnames([i[0] for i in ct])
        st.dataframe(ctnames)
        systemmessage = systemtemplate.format(ata = newusr['AvgTransactionAmount'],
                                                tf = newusr['TransactionFrequency'],
                                                sg = newusr['SpendingOnGroceries'],
                                                se = newusr['SpendingOnEntertainment'],
                                                st = newusr['SpendingOnTravel'],
                                                ti = newusr['TotalIncome'])
        ans = generate(chatllm,systemmessage,hm)
        st.write(ans.content)
        

        









  