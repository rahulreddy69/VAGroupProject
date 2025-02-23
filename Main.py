import altair as alt
import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
import base64

@st.cache(allow_output_mutation=True)
def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_png(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = '''
    <style>
    .stApp {
        background-image: url("data:image/png;base64,%s");
        background-size: cover;
    }
    </style>
    ''' % bin_str
    
    st.markdown(page_bg_img, unsafe_allow_html=True)
    return

set_png('background.png')

def check_pwd():
    main_title = '<p style="font-family:fantasy; color:Black; font-size: 55px; text-align: center;">Profitability at Movie Making</p>'
    st.markdown(main_title, unsafe_allow_html=True)

    def password_entered():
        if st.session_state["password"] == st.secrets["password"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.text_input(
            "Login:", type="password", on_change=password_entered, key="password"
        )
        return False
    elif not st.session_state["password_correct"]:
        st.text_input(
            "Login:", type="password", on_change=password_entered, key="password"
        )
        st.error("Password incorrect")
        return False
    else:
        return True

if check_pwd():
    ##########################################################################################################################
    #Import Datasets
    ###########################################################################################################################
    release_dates_df = pd.read_csv('movie_release_dates.csv', index_col=0)
    theaters_df = pd.read_csv('movie_theater_data.csv', index_col=0)
    movie_awards_df = pd.read_csv('movie_awards.csv', index_col=0)
    actors_table_df = pd.read_csv('Actors_Table.csv')
    directors_table_df = pd.read_csv('Directors_Table.csv')
    base_df = pd.read_csv('IMDb_base.csv')
    budgets_df = pd.read_csv('IMDb_budgets.csv')

    ##########################################################################################################################
    #General
    ##########################################################################################################################
    budgets_df = budgets_df[budgets_df['Domestic Gross'] !=0]

    budgets_df['Production Budget'] = (((2020-budgets_df['Year'])*.045)+1)* budgets_df['Production Budget']
    budgets_df['Worldwide Gross'] = (((2020-budgets_df['Year'])*.045)+1)* budgets_df['Worldwide Gross']
    budgets_df['Domestic Gross'] = (((2020-budgets_df['Year'])*.045)+1)* budgets_df['Domestic Gross']

    budgets_df['Net Profit'] = budgets_df['Worldwide Gross'] - budgets_df['Production Budget']
    budgets_df['Profit Margin'] = budgets_df['Net Profit'] / budgets_df['Worldwide Gross']

    budgets_df['Genre'] = budgets_df['Genre'].str.split(', ')
    budgets_df1 = budgets_df['Genre'].apply(pd.Series)
    budgets_df2 = pd.merge(budgets_df, budgets_df1, right_index = True, left_index = True)
    budgets_df3 = budgets_df2.drop(['Genre'], axis = 1)
    genre_df = budgets_df3.melt(id_vars=['Movie', 'Year'], value_vars=[0, 1, 2] ,var_name = ['X'])
    genre_df = pd.merge(genre_df, budgets_df)
    genre_df = genre_df.drop(['Genre', 'X'], axis=1)
    genre_df = genre_df.drop_duplicates()
    genre_df = genre_df.rename(columns={'value': 'Genre'})
    genre_df = genre_df.dropna()

    ##########################################################################################################################
    #question 1
    ##########################################################################################################################

    budgets_df['Profit'] = budgets_df['Worldwide Gross'] - budgets_df['Production Budget']

    budgets_df['Adjusted_Budget'] = ((((2020-budgets_df["Year"])*.047)+1)*
                                        budgets_df['Production Budget'])
    budgets_df['Adjusted_Profit'] = (((2020-budgets_df['Year'])*.047)+1)* budgets_df['Profit']

    profitable_df = budgets_df.loc[budgets_df['Profit'] > 0]
    pranked_df = profitable_df.sort_values(by=['Adjusted_Profit'], ascending=False)
    pranked_df.reset_index(inplace=True)
    #pranked_df.head()

    def ques_1():
        plt.figure(figsize=(15,12))
        sns.barplot(x=pranked_df.loc[0:25, 'Movie'],y=pranked_df.loc[0:25, 'Adjusted_Profit'], 
                color='mediumspringgreen', label='Profit', ci=None)
        sns.barplot(x=pranked_df.loc[0:25, 'Movie'],y=pranked_df.loc[0:25, 'Adjusted_Budget'], 
                color='black', label='Budget', ci=None)
        plt.xlabel('Movie', fontsize=13)
        plt.title("Budgets and Profits for the top 25 Most Profitable Movies", fontsize=15)
        plt.ylabel('Adjusted Profit ', fontsize=13)
        plt.xticks(rotation=34, horizontalalignment='right', fontsize=13)
        plt.legend(fontsize=13)
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()

    #############################################################################################################################
    #Question2
    #############################################################################################################################
    m_genre = genre_df.groupby('Genre', as_index=False)['Movie'].count().sort_values(by='Movie', ascending=False)
    p_genre = genre_df.groupby('Genre', as_index=False)['Net Profit', 'Profit Margin'].median().sort_values(by='Net Profit', ascending=False)

    per_genre = genre_df.groupby(['Genre'],  as_index=False)['Net Profit'].sum().sort_values(by='Net Profit', ascending=False)
    per_genre['Percentage Movie Total of Net Profit'] = (per_genre['Net Profit']/per_genre['Net Profit'].sum()*100).round(2)

    def q2_1():
        
        plt.figure(figsize=(19,9))
        a1 = sns.barplot(x=m_genre['Genre'], y=m_genre['Movie'])
        a1.set(title='Movie Counting by Genre')
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()

    def q2_2():
        
        plt.figure(figsize=(20,8))
        a2 = sns.barplot(x=p_genre['Net Profit'], y=p_genre['Genre'])
        a2.set(xlabel='Net Profit in Hundreds of Millions', ylabel='Genre', title='Net Profit  By Genre')
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()

    def q2_3():
        plt.figure(figsize=(20,9))
        ax12 = sns.barplot(x=p_genre['Profit Margin'], y=p_genre['Genre'])
        ax12.set(xlabel='Profit Margin', ylabel='Genre', title='Profit Margin of the Movie By Genre')
        plt.xlim(0.4, 0.86)
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()


    def q2_4():
        plt.figure(figsize=(20,9))
        ax3 = sns.barplot(x=per_genre['Genre'], y=per_genre['Percentage Movie Total of Net Profit'])
        ax3.set(title='Percentage of the Movie Profit By Genre')
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()

    #############################################################################################################################
    #Question3
    #############################################################################################################################
    budgets_df['Release Date'] = pd.to_datetime(budgets_df['Release Date'])

    dateData =  [x.strftime('%B') for x in budgets_df['Release Date']]
    budgets_df['Month'] = dateData

    m_month = budgets_df.groupby(['Month'], as_index=False)['Movie'].count().sort_values(by='Movie', ascending=False)
    p_month = budgets_df.groupby('Month', as_index=False)['Net Profit', 'Profit Margin'].median().sort_values(by='Net Profit', ascending=False)

    def q3_1():
        plt.figure(figsize=(15,8))
        a4 = sns.countplot(x=budgets_df['Month'],
                    order=['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'])
        a4.set(ylabel='Count', title='Movie Count By Release Month')
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()

    def q3_2():
        plt.figure(figsize=(15,8))
        a5 = sns.barplot(x=p_month['Month'], y=p_month['Net Profit'], 
                    order=['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'])
        a5.set(ylabel='Net Profit(Tens of Millions)', title='Movie Median Net Profit By Release Month')
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()

    def q3_3():
        plt.figure(figsize=(15,8))
        a6 = sns.barplot(x=p_month['Month'], y=p_month['Profit Margin'], 
                    order=['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'])
        a6.set(ylabel='Profit Margin', title='Profit Margin of the Movie By Release Month')
        plt.ylim(0.5, 0.9)
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()
    #############################################################################################################################
    #Question4
    #############################################################################################################################
    actors_table_df['Production Budget'] = (((2020-actors_table_df['Year'])*.0322)+1)*actors_table_df['Production Budget']
    actors_table_df['Worldwide Gross'] = (((2020-actors_table_df['Year'])*.0322)+1)*actors_table_df['Worldwide Gross']
    actors_table_df['Domestic Gross'] = (((2020-actors_table_df['Year'])*.0322)+1)*actors_table_df['Domestic Gross']
    actors_table_df['Net Profit'] = actors_table_df['Worldwide Gross'] - actors_table_df['Production Budget']
    actors_table_df['Profit Margin'] = actors_table_df['Net Profit'] / actors_table_df['Worldwide Gross']

    counts = actors_table_df['value'].value_counts()
    alist = counts[counts >= 10].index.tolist()
    actors_table_df = actors_table_df[actors_table_df['value'].isin(alist)]


    atotal = actors_table_df.groupby(['value'],  as_index=False)['Net Profit'].mean().sort_values(by='Net Profit', ascending=False)
    atotal['VAR'] = (atotal['Net Profit']/atotal['Net Profit'].mean())
    top_25actors = atotal.head(25)

    directors_table_df['Production Budget'] = (((2020-directors_table_df['Year'])*.0322)+1)*directors_table_df['Production Budget']
    directors_table_df['Worldwide Gross'] = (((2020-directors_table_df['Year'])*.0322)+1)*directors_table_df['Worldwide Gross']
    directors_table_df['Domestic Gross'] = (((2020-directors_table_df['Year'])*.0322)+1)*directors_table_df['Domestic Gross']

    directors_table_df['Net Profit'] = directors_table_df['Worldwide Gross'] - directors_table_df['Production Budget']
    directors_table_df['Profit Margin'] = directors_table_df['Net Profit'] / directors_table_df['Worldwide Gross']
    dcounts = directors_table_df['value'].value_counts()
    dlist = dcounts[dcounts >= 5].index.tolist()
    directors_table_df = directors_table_df[directors_table_df['value'].isin(dlist)]
    dtotal = directors_table_df.groupby(['value'],  as_index=False)['Net Profit'].mean().sort_values(by='Net Profit', ascending=False)
    dtotal['VAR'] = (dtotal['Net Profit']/atotal['Net Profit'].mean())
    top_25directors = dtotal.head(25)
    #top_25directors

    def q4_1():
        plt.figure(figsize=(15,8))
        a7 = sns.barplot(x=top_25actors['VAR'], y=top_25actors['value'])
        a7.axvline(1, ls='-', color='black', linewidth=4)
        a7.set(ylabel='Actor', title='Average Compared with VAR by Actor')
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()

    def q4_2():
        plt.figure(figsize=(15,8))
        ax7 = sns.barplot(x=top_25directors['VAR'], y=top_25directors['value'])
        ax7.axvline(1, ls='-', color='black', linewidth=3)
        ax7.set(ylabel='Director', title='VAR By Director Compared to Average')
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()
    #############################################################################################################################
    #Question5
    #############################################################################################################################
    budgets_df.set_index(['Movie','Year'], inplace=True)
    movie_awards_df.set_index(['film_name', 'film_year'], inplace=True)

    awards_and_budgets = budgets_df.join(movie_awards_df, how='inner', on=['Movie', 'Year'])

    nominated_df = awards_and_budgets.loc[awards_and_budgets['Profit'] > 0]

    def q5_1():
        plt.figure(figsize=(15,8))
        sns.boxplot(x='Adjusted_Budget', data=nominated_df, showfliers=False, color='powderblue')
        sns.stripplot(x='Adjusted_Budget', data=nominated_df)
        plt.ticklabel_format(axis='x', style='sci', scilimits=(6,6))
        plt.xticks(fontsize=13)
        plt.xlabel('Movie Budgets Adjusted for Inflation ', fontsize=13)
        plt.title('Distribution of Movie Budgets for Profitable Oscar Nominated Movies', fontsize=14)
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()
    #############################################################################################################################
    #Question6
    #############################################################################################################################

    def q6_1():
        plt.figure(figsize=(14,7))
        ax15 = sns.regplot(x='Runtime', y='Adjusted_Profit', data=budgets_df)
        plt.xlabel('Movie Runtime', fontsize=12)
        plt.ylabel('Net Profit (In Billions)', fontsize=12)
        plt.title('Correlation Between Net Profit and Runtime', fontsize=14)
        plt.savefig('CorrProfitRuntime')
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()

    #############################################################################################################################
    #Question7
    #############################################################################################################################
    
    filter_df = pranked_df.loc[(pranked_df['Profit Margin'] >= 0.75) & 
                         (pranked_df['Adjusted_Budget'] > 38676000)]
    def q7_1():
        ax2 = sns.lmplot(x='Adjusted_Budget', y='Profit Margin', data=filter_df, height=7, aspect=2)
        plt.xlabel('Adjusted Budget (Millions of Dollars)', fontsize=12)
        plt.ticklabel_format(axis='x', style='sci', scilimits=(6,6))
        plt.ylabel('Profit Margin', fontsize=12)
        plt.title('Adjusted Budget vs Profit Margin', fontsize=14)
        plt.savefig('BudgetVMargin')
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()

    ##########################################################################################################################
    # Streamlit part 
    ##########################################################################################################################

    my_button = st.sidebar.radio("Profitability at Movie Making", ('Homepage','Budget', 'Genre','Best_Time','Actors_And_Directors','Oscar_Winning','Runtime','Baseline for Success')) 

    if my_button == 'Homepage':
        st.title('Welcome Page!')
        st.title('These visualizations helps')
    elif my_button == 'Budget': 
        ques_1()
    elif my_button == 'Genre': 
        q2_1()
        q2_2()
        q2_3()
        q2_4()
    elif my_button == 'Best_Time': 
        q3_1()
        q3_2()
        q3_3()
        
    elif my_button == 'Actors_And_Directors': 
        q4_1()
        q4_2()
    elif my_button == 'Oscar_Winning': 
        q5_1()
    elif my_button == 'Runtime': 
        q6_1()
    elif my_button == 'Baseline for Success': 
        q7_1()
        #We examine the data in a scatter plot again to see if we can determine trends. Our data is much more spread out when comparing profit margin and budget. The trend line in this plot is negative which cautions against spending too much money as we may potentially hurt our profit margin. Looking at the filtered data, we see median profit margin of 81.9%. Hence, we recommend movie studios to strive for a *profit margin of about 81.9% to stay afloat*
        q2_2()
        #for second chart: We recommend Movie Studios to focus their efforts on the top 6 most profitable movie genres: Adventure, Action, Comedy, Drama, Sci-Fi and Animation.
        q3_3()
        #best time
        q4_1()
        q4_2()
        #We recommend that studios focus their cast and crew search to individuals who consistently score at least 1.0 on the VAR score. We can, with a high level of confidence, conclude that these individuals will elevate the overall production.
        q5_1()
        #Ideal range to spend for oscar between 30 to 65 million dollar
