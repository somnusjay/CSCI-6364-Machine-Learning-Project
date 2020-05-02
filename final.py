#  49 states here
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.linear_model import LinearRegression


def check(i, j):
    x = covid_data.values[:, i].reshape((-1, 1))
    y = covid_data.values[:, j].reshape((-1, 1))
    lr = LinearRegression()
    model = lr.fit(x, y)
    y_pred = lr.predict(x)
    score = model.score(x, y)
    print('{}-{} score: {:0.3f}'.format(covid_data.columns[i], covid_data.columns[j], score))
    print('{}-{} RMSE: {:0.3f}'.format(covid_data.columns[i], covid_data.columns[j],
                                       np.sqrt(metrics.mean_squared_error(y, y_pred))))
    k = model.intercept_
    b = model.coef_

    plt.figure()
    plt.scatter(x, y)
    plt.plot(x, y_pred, color='red')
    plt.xlim(x.min() * 0.9, x.max() * 1.1)
    plt.ylim(y.min() * 0.9, y.max() * 1.1)
    plt.title('{}-{}'.format(covid_data.columns[i], covid_data.columns[j]))
    plt.xlabel('{}'.format(covid_data.columns[i]))
    plt.ylabel('{}'.format(covid_data.columns[j]))
    plt.show()


path = "D:/Graduate/CSCI 6364 Machine Learning/project/"
covid_data = pd.read_csv(path + "us_states_covid19_daily.csv")
covid_data = covid_data[['date', 'state', 'positive', 'negative', 'death']]
covid_data.sort_values(['date', 'positive'], axis=0, ascending=False, inplace=True)
covid_data = covid_data.drop(columns=['date'])
covid_data = covid_data.head(52)
covid_data = covid_data.drop([8, 25, 42])  # drop DC, MN, PR
covid_data.sort_values(['state'], axis=0, ascending=True, inplace=True)
covid_data.rename(columns={'state': 'state_abbreviation'}, inplace=True)
covid_data = covid_data.reset_index(drop=True)
# print(covid_data)

positive_rate = []
for row in covid_data.values:
    positive_rate.append(row[1] / (row[1] + row[2]))

total_test = []
for row in covid_data.values:
    total_test.append(row[1] + row[2])

death_rate = []
for row in covid_data.values:
    death_rate.append(row[3] / row[1])

covid_data['total_test'] = total_test
covid_data['positive_rate'] = positive_rate
covid_data['death_rate'] = death_rate

party_data = pd.read_csv(path + "primary_results.csv")
party_data.sort_values(['state_abbreviation'], axis=0, ascending=True, inplace=True)
party_data.drop(columns=['county', 'fips', 'candidate', 'fraction_votes'], inplace=True)
state_abbreviation = pd.unique(party_data['state_abbreviation'])
# print(party_data)

democrat = party_data[party_data['party'] == 'Democrat']
democrat = democrat.groupby('state_abbreviation').agg({'state': 'first', 'votes': 'sum'})
democrat.columns = ['state', 'democrat']
democrat = democrat.reset_index()
# print(democrat)
republican = party_data[party_data['party'] == 'Republican']
republican = republican.groupby('state_abbreviation').agg({'state': 'first', 'votes': 'sum'})
republican.columns = ['state', 'republican']
republican = republican.reset_index()
# print(republican)

covid_data['democrat'] = 0
covid_data['republican'] = 0
covid_data['state'] = ''
# print(covid_data)

index = 0
for row in democrat.values:
    while covid_data.values[index][0] != row[0]:
        index += 1
    covid_data.at[index, 'state'] = row[1]
index = 0
for row in democrat.values:
    while covid_data.values[index][0] != row[0]:
        index += 1
    covid_data.at[index, 'democrat'] = row[2]
index = 0
for row in republican.values:
    while covid_data.values[index][0] != row[0]:
        index += 1
    covid_data.at[index, 'republican'] = row[2]
# print(covid_data)

democrat_rate = []
for row in covid_data.values:
    democrat_rate.append(row[7] / (row[7] + row[8]))
covid_data['democrat_rate'] = democrat_rate
# print(covid_data)

census_data = pd.read_csv(path + "acs2017_county_data.csv")
census_data = census_data.groupby('State').agg({'TotalPop': 'sum', 'Income': 'mean', 'White': 'mean', 'Black': 'mean'})
census_data = census_data.reset_index()
census_data.columns = ['state', 'population', 'income', 'white', 'black']
census_data = census_data.drop([8, 23, 39])
census_data = census_data.reset_index(drop=True)
# print(census_data)

covid_data['population'] = 0
covid_data['income'] = 0.0
covid_data['white'] = 0.0
covid_data['black'] = 0.0

covid_data.sort_values(['state'], axis=0, ascending=True, inplace=True)
covid_data = covid_data.reset_index(drop=True)
index = 0
for row in census_data.values:
    while covid_data.values[index][9] != row[0]:
        index += 1
    covid_data.at[index, 'population'] = row[1]
    covid_data.at[index, 'income'] = row[2]
    covid_data.at[index, 'white'] = row[3]
    covid_data.at[index, 'black'] = row[4]
# print(covid_data)

test_rate = []
for row in covid_data.values:
    test_rate.append(row[4] / row[11])
    if row[4] / row[11] > 1:
        print(row)
covid_data['test_rate'] = test_rate
# print(covid_data)

hospital_data = pd.read_csv(path + "HospInfo.csv")
hospital_data = hospital_data.groupby('State').agg({'Hospital Name': 'count'})
hospital_data = hospital_data.reset_index()
hospital_data = hospital_data.drop([3, 8, 12, 25, 27, 42, 50])
hospital_data = hospital_data.reset_index(drop=True)
# print(hospital_data)

covid_data.sort_values(['state_abbreviation'], axis=0, ascending=True, inplace=True)
covid_data = covid_data.reset_index(drop=True)
covid_data['hospital'] = 0
index = 0
for row in hospital_data.values:
    while covid_data.values[index][0] != row[0]:
        index += 1
    covid_data.at[index, 'hospital'] = row[1]
# print(covid_data)


index = 0
for col in covid_data.columns:
    print(index, col)
    index += 1

# check(4, 16)
