import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# read in data
films = pd.read_csv('movies_data.csv')

films.head()
films.columns
films.info()
films.describe()

"""
- there is a column 'Unnamed: 0' - not necessary
- there are 58788 titles
- year
- length
- budget - there are only 5215 rows of data
- rating
- votes
- r1-r10
- mpaa - there are only 4924 rows with data
- category dummy variable - one film can have more that one category
- check for duplicates
"""

# 1. What are the 3 most common ratings (1­10) for movies in the list?
# Round each rating up or down to the nearest whole number.

films['rating_whole'] = films['rating'].map(lambda x: np.round(x, 0))
films['rating_whole'].value_counts()

# 2. Are there more R­Rated movies or PG­13 movies in this list?
films['mpaa'].value_counts()

# 3. Are there more Drama, Comedy, or Romance films in this list?
for i in ['Comedy', 'Drama', 'Romance']:
    print(films[i].value_counts())

# 4. How many movies are described as both Action and Comedy (but no other genre)?
action_filter = films['Action'] == 1
comedy_filter = films['Comedy'] == 1

action_comedy_films = films[action_filter & comedy_filter].copy()
genres = ['Action', 'Animation', 'Comedy', 'Drama', 'Documentary', 'Romance', 'Short']
sum_filter = action_comedy_films[genres].sum(axis=1) == 2
len(action_comedy_films[sum_filter])

# 5. What is the average, median, 25th percentile and 75th percentile of ratings?
films['rating'].describe()

# 6. Which of the following has the strongest correlation coefficient (r)?
# Rating vs. Votes, Length vs. Rating, or Year vs. Rating. What is that correlation?
np.corrcoef(films['rating'], films['votes'])
np.corrcoef(films['length'], films['rating'])
np.corrcoef(films['year'], films['rating'])

# 7. If you plot Length vs. Rating, and you look for the most obvious outlier, what is the name of that movie?
plt.scatter(films['length'], films['rating'])
plt.show()

outlier_filter = films['length'] > 4000
films[outlier_filter]


# Pick the genre for the next movie

def genre_mean_rating(df_data, genres_list):
    """
    Calculate the mean rating for a all movies from a specific genre
    :param df_data: data frame with movies data
    :param genres_list: list of all genres in a data frame
    :return: a data frame with movie genre and its mean rating
    """
    mean_ratings = pd.DataFrame(columns=['genre', 'mean rating'])
    for i in genres_list:
        genre_filter = df_data[i] == 1
        genre_ratings = df_data.loc[genre_filter, 'rating']
        mean_rating = round(np.float(np.mean(genre_ratings)), 1)
        mean_ratings.loc[len(mean_ratings)] = [i, mean_rating]
    return mean_ratings.sort_values(['mean rating'], ascending=False)

ratings = genre_mean_rating(films, genres)

def genre_popularity(df_data, genres_list):
    """
    Check which genre got the most votes from fans
    :param df_data: data frame with movies data
    :param genres_list: list of all genres in a data frame
    :return: a data frame with movie genre and the sum of the votes
    """
    popularity_vote = pd.DataFrame(columns=['genre', 'number of votes'])
    for i in genres_list:
        genre_filter = df_data[i] == 1
        genre_votes = df_data.loc[genre_filter, 'votes']
        votes_sum = np.sum(genre_votes)
        popularity_vote.loc[len(popularity_vote)] = [i, votes_sum]
    return popularity_vote.sort_values(['number of votes'], ascending=False)

popularity = genre_popularity(films, genres)
genre_summary = pd.merge(popularity, ratings, on='genre')

def plot_genre_votes_ratings(df_data):
    """
    Creates a plot comparing popularity of the movie with its mean rating
    :param df_data: ata frame with movies data
    :return: saves a plot
    """
    fig, ax1 = plt.subplots(figsize=(10,8))
    x = np.arange(len(df_data['genre']))
    plt.xticks(x, df_data['genre'], rotation=45)
    ax1.bar(x, df_data['number of votes']/1000000, alpha=0.5, color='green')
    ax1.set_ylabel('genre popularity (number of votes in mln)', color='green', fontsize=15)

    ax2 = ax1.twinx()
    ax2.plot(x, df_data['mean rating'], 'r*', markersize=15)
    ax2.set_ylim(0,10)
    ax2.set_ylabel('genre mean rating', color='red', fontsize=15)
    plt.title('Popularity vs. Rating score among movie genres', fontsize=25)
    ax2.title.set_position([0.5, 1.05])

    plt.savefig('genre_popularity.png')

plot_genre_votes_ratings(genre_summary)


def genre_budget(df_data, genres_list):
    """
    Calculates sum of dollars spent per movie genre (in mln $)
    :param df_data: data frame with movies data
    :param genres_list:
    :return:
    """
    budgets = pd.DataFrame(columns=['genre', 'total_budget'])
    for i in genres_list:
        filter = df_data[i] == 1
        budget = df_data.loc[filter, 'budget']
        budget_sum = np.sum(budget)/1000000
        budgets.loc[len(budgets)] = [i, budget_sum]
    return budgets.sort_values(['total_budget'], ascending=False)

budget = genre_budget(films, genres)

# compare
genre_profit = pd.merge(popularity, budget, on='genre')
genre_profit['cost per vote'] = genre_profit['total_budget'] / genre_profit['number of votes'] * 1000000
genre_profit.sort_values(['cost per vote'], ascending=False)


# Find something interesting
# show those movies where one rating category had distribution above 60 (any of the r1-r10)

def concentrated_ratings(df_data, labels):
    """
    Check which movies has the rating concentrated within one label
    :param df_data: data frame with movies data
    :param labels: distribution labels (number of stars)
    :return: data frame with new column as a marker for the movies with concentrated rating
    """
    for row in df_data.itertuples():
        for i in labels:
            if df_data.loc[row.Index, i] > 60:
                df_data.loc[row.Index, 'conc_ind'] = 1
    return df_data

films2 = films.copy()
distribution_labels = ['r1', 'r2', 'r3', 'r4', 'r5', 'r6', 'r7', 'r8', 'r9', 'r10']
concentrated_ratings(films2, distribution_labels)

con_filter = films2['conc_ind'] == 1
films2 = films2[con_filter]

# what is the mean number of votes per movie (for the general data and for concentrated group)
vote_mean = np.mean(films['votes'])
con_vote_mean = np.mean(films2['votes'])

# plot different mean distribution of votes for those two groups

def mean_group_distribution(df_data_general, df_data_conc, labels):
    """
    Calculate mean rating distribution for each movie category
    :param df_data_general: data frame with all movie data
    :param df_data_conc: data frame with movie groupes as concentrated rating
    :param labels: column labels
    :return: a data frame with summary of mean ratings for two groups of movies
    """
    dist_comparison = pd.DataFrame(columns=labels)
    for i in distribution_labels:
        mean_gen_dist = np.mean(df_data_general.loc[:, i])
        mean_conc_dist = np.mean(df_data_conc.loc[:, i])
        dist_comparison.loc[len(dist_comparison)] = [mean_gen_dist, mean_conc_dist]
    return dist_comparison

dist_comp = mean_group_distribution(films, films2, ['general distribution', 'conc distribution'])


def plot_mean_distribution(df_comp):
    """
    Creates a plot comparing mean rating distribution for two groups of movies (general and concentrated)
    :param df_comp: comparison of mean rating distribution for two categories
    :return: saves a plot
     """
    x = np.arange(1,11,1)
    ax = plt.subplot(111)
    w = 0.4
    ax.bar(x, df_comp.iloc[:, 0], width=0.3, color='blue', alpha=0.8, align='center')
    ax.bar(x+w, df_comp.iloc[:, 1], width=0.3, color='orange', alpha=0.8, align='center')
    plt.title('Distribution of mean rating for different movie categories')
    ax.set_xlabel('Rating (number of stars)')
    ax.set_ylabel('Mean distribution of ratings')
    ax.set_xticks(x)
    plt.legend(['general', 'concentrated'])
    plt.savefig('rating_distribution.png')

plot_mean_distribution(dist_comp)
