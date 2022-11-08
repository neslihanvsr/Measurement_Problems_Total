# Measurement Problems
# - Rating Products
# - Sorting Products
# - Sorting Reviews
# - AB Testing
#
# Social Proof
# The Wisdom of Crowds
# Marketplace
#
#
## Rating Products
#
#   AVERAGE
# - Time-Based Weighted Average
# - User-Based Weighted Average
# - Weighted Rating

## Sorting Products
#
# - Sorting by Rating
# - Sorting by Comment Count or Purchase Count
# - Sorting by Rating, Comment and Purchase
# - Sorting by Bayesian Average Rating Score (Sorting Products with 5 Star Rated)
# - Hybrid Sorting BAR Score
#
#
## Sorting Reviews
#
# - Up-Down Difference Score
# - Average Rating
# - Wilson Lower Bound Score (Wilson Lower Bound Score)



###  Rating Products  ###

import pandas as pd
import datetime as dt
import math
import scipy.stats as st
from sklearn.preprocessing import MinMaxScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%5f' % x)

df = pd.read_csv(r'C:\Users\Desktop\course_reviews.csv')
df.head()
df.shape

df['Rating'].value_counts() #rating distribution
df.groupby('Questions Asked').agg({'Questions Asked': 'count',
                                   'Rating': 'mean'})

df.head()

# AVERAGE #

df['Rating'].mean()


## Time-Based Weighted Average ##

df.head()
df.info()
df['Timestamp'] = pd.to_datetime(df['Timestamp'])

current_date = pd.to_datetime('2021-02-10 0:0:0') #express the comments made in days, the comment dates are certain; timestamp

df[df['days'] <= 30] .count() #comments made in the last 30 days

df.loc[df['days'] <= 30, 'Rating'].mean() #average of the comments made in the last 30 days

# we can now reflect the effect of time to the weight calculation by giving different weights to the results at these different time intervals.
df.loc[df['days'] <= 30, 'Rating'].mean() * 28/100 + \
df.loc[(df['days'] > 30) & (df['days'] <= 90), 'Rating'].mean() * 26/100 + \
df.loc[(df['days'] > 90) & (df['days'] <= 180), 'Rating'].mean() * 24/100 + \
df.loc[(df['days'] > 180), 'Rating'].mean() * 22/100

# write those calculation in a function
def time_based_weighted_average(dataframe, w1=28, w2=26, w3=24, w4=22):
    return  dataframe.loc[df['days'] <= 30, 'Rating'].mean() * w1/100 + \
            dataframe.loc[(df['days'] > 30) & (dataframe['days'] <= 90), 'Rating'].mean() * w2/100 + \
            dataframe.loc[(df['days'] > 90) & (dataframe['days'] <= 180), 'Rating'].mean() * w3/100 + \
            dataframe.loc[(df['days'] > 180), 'Rating'].mean() * w4/100


time_based_weighted_average(df)
time_based_weighted_average(df, 27, 30, 23, 20) #can change weights



## User-Based Weighted Average ##

#The ratings of users who watched the course at different rates should be evaluated with different weights:

df.head()

df.loc[df['Progress'] <= 10, 'Rating'].mean() * 22/100 + \
    df.loc[(df['Progress'] > 10) & (df['Progress'] <= 45), 'Rating'].mean() * 24/100 + \
    df.loc[(df['Progress'] > 45) & (df['Progress'] <= 75), 'Rating'].mean() * 26/100 + \
    df.loc[(df['Progress'] > 75), 'Rating'].mean() * 28/100


def user_based_weighted_average(dataframe, w1=22, w2=24, w3=26, w4=28):
    return  dataframe.loc[df['Progress'] <= 10, 'Rating'].mean() * w1/100 + \
            dataframe.loc[(df['Progress'] > 10) & (dataframe['Progress'] <= 45), 'Rating'].mean() * w2/100 + \
            dataframe.loc[(df['Progress'] > 45) & (dataframe['Progress'] <= 75), 'Rating'].mean() * w3/100 + \
            dataframe.loc[(df['Progress'] > 75), 'Rating'].mean() * w4/100

user_based_weighted_average(df)
user_based_weighted_average(df, 21, 23, 27, 29)



## Weighted Rating ##

#will combine Time-Based Weighted Average and User-Based Weighted Average calculations in a single function:

def course_weighted_rating(dataframe, time_w=50, user_w=50):
    return time_based_weighted_average(dataframe) * time_w/100 + user_based_weighted_average(dataframe) * user_w/100

course_weighted_rating(df)
course_weighted_rating(df, time_w=45, user_w=55) ##can change the default values



###  Sorting Products  ###

import pandas as pd
import math
from sklearn.preprocessing import MinMaxScaler
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 1000)
pd.set_option('display.expand_frame_repr', True)
pd.set_option('display.float_format', lambda x: '%5f' % x)

df = pd.read_csv(r'C:\Users\Desktop\course_reviews.csv')
df.head()
df.shape

## Sorting by Rating ##

df['rating'].mean()
df.sort_values('rating', ascending=False).head(5)

## Sorting by Comment Count or Purchase Count ##

df.sort_values('purchase_count', ascending=False).head(20) #cannot give for the most accurate ranking, comments should also be taken into account; social value
df.sort_values('commment_count', ascending=False).head(20)

## Sorting by Rating, Comment and Purchase ##

#convert the purchasing variables to values 1-5:
df['purchase_count_scaled'] = MinMaxScaler(feature_range=(1, 5)). \
    fit(df[['purchase_count']]). \
    transform(df[['purchase_count']])
df.head()

df.describe().T

#convert the comment variables to values 1-5:
df['comment_count_scaled'] = MinMaxScaler(feature_range=(1, 5)). \
    fit(df[['commment_count']]). \
    transform(df[['commment_count']])
df.head()

df.describe().T


(df['comment_count_scaled'] * 32 / 100 +
df['purchase_count_scaled'] * 26 / 100 +
df['rating'] * 42 / 100)

#now that the scales are of the same type, merge in a function:
def weighted_sorting_score(dataframe, w1=32, w2=26, w3=42):
    return (dataframe['comment_count_scaled'] * w1 / 100 +
            dataframe['purchase_count_scaled'] * w2 / 100 +
            dataframe['rating'] * w3 / 100)
df['weighted_sorting_score'] = weighted_sorting_score(df)

df.sort_values('weighted_sorting_score', ascending=False).head(5)


## Sorting by Bayesian Average Rating Score (Sorting Products with 5 Star Rated) ##

#calculattion the average over the score distribution:
def bayesian_average_rating(n, confidence=0.95):
    if sum(n) == 0:
        return 0
    K = len(n)
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    N = sum(n)
    first_part = 0.0
    second_part = 0.0
    for k, _k in enumerate(n):
        first_part += (k + 1) * (n[k] + 1) / (N + N)
        second_part += (k +1) * (k+1) * (n[k] +1) / (N + K)
    score = first_part - z * math.sqrt((second_part - first_part * first_part) / (N + K + 1))
    return score

df.head()

df['bar_score'] = df.apply(lambda x: bayesian_average_rating(x[['1_point',
                                                                '2_point',
                                                                '3_point',
                                                                '4_point',
                                                                '5_point']]), axis=1)

df.sort_values('bar_score', ascending=False).head(20)

df[df['course_name'].index.isin([5, 1])].sort_values('bar_score', ascending=False)


## Hybrid Sorting ##

#BAR Score and other factors

def hybrid_sorting_score(dataframe, bar_w=60, wss_w=40):
    bar_score = dataframe.apply(lambda x: bayesian_average_rating(x[['1_point',
                                                                '2_point',
                                                                '3_point',
                                                                '4_point',
                                                                '5_point']]), axis=1)
    wss_score = weighted_sorting_score(dataframe)
    return bar_score * bar_w / 100 + wss_score*wss_w / 100

df['hybrid_sorting_score'] = hybrid_sorting_score(df)
df.sort_values('hybrid_sorting_score', ascending=False).head(5)
df[df['course_name'].str.contains('Veri Bilimi')].sort_values('hybrid_sorting_score', ascending=False).head(5)




###  Sorting Reviews  ###

#user quality score should also be taken into account:

import pandas as pd
import math
import scipy.stats as st

pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

## Up-Down Difference Score ##
#
# Up-Down Diff Score = (up ratings) - (down ratings)

# Review 1: 600 up 400 down total 1000
# Review 2: 5500 up 4500 down total 10000

def score_up_down_diff(up, down):
    return up - down

# Review 1 Score:
score_up_down_diff(600, 400) # up 60%

# Review 2 Score:
score_up_down_diff(5500, 4500)  # up 55%

#the total number of comments should also be taken into account!



## Average Rating ##
#
# Score = Average rating = (up ratings) / (all rating)

def score_average_rating(up, down):
    if up + down == 0:
        return 0
    return up / (up + down)

score_average_rating(600, 400)
score_average_rating(5500, 4500)

# Review 1: 2 up 0 down total 2
# Review 2: 100 up 1 down total 101

score_average_rating(2, 0)
score_average_rating(100, 1)

#this method also calculates the ratio information correctly, but the frequency information is missing!



## Wilson Lower Bound Score (Wilson Lower Bound Score) ##
#
#provides sorting facility for any item with dual interaction.
#both frequency and rate information are taken into account.

# 600-400
# 0.6
# 0.5 0.7  #up ratio
# 0.5

def wilson_lower_bound(up, down, confidence=0.95):
    """
     Parameters
    ----------
    up: int
        up count
    down: int
        down count
    confidence: float
        confidence

    Returns
    -------
    wilson score: float

    """
    n = up + down
    if n == 0:
        return 0
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    phat = 1.0 * up / n
    return (phat + z * z /(2 * n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)

wilson_lower_bound(600, 400)
wilson_lower_bound(5500, 4500)

wilson_lower_bound(2, 0)
wilson_lower_bound(100, 1)

#calculated with 95% confidence and 5% margin of error.


# Wilson Lower Bound Score calculation:
#- The lower limit of the confidence interval to be calculated for the Bernoulli parameter p is accepted as the WLB score.
#- The score to be calculated is used for product ranking.

# Note:
     # If the scores are between 1-5, 1-3 are marked as negative, 4-5 as positive and can be made to conform to Bernoulli.
     # This brings with it some problems. For this reason, it is necessary to make a bayesian average rating.

