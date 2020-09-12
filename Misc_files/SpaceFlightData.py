import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import seaborn as sns
from datetime import datetime
import statsmodels.api as sm

# %matplotlib inline

# https://www.kaggle.com/agirlcoding/all-space-missions-from-1957
df = pd.read_csv('/Users/richinman/Desktop/Udacity/Project_1/SpaceFlights/datasets_828921_1417131_Space_Corrected.csv')
df.head()
df.columns


# Rename a few columns to remove spaces
df = df.rename(columns={"Company Name": "Company_Name", "Status Rocket": "Rocket_Status", "Status Mission": "Mission_Status", " Rocket": "Rocket_Price"})
df.columns
 

# Create column for Success/Failure. Recode partial failures as failures
def Success_Fail(series):
	if series == 'Prelaunch Failure':
		return "Failure"
	elif series == 'Partial Failure':
		return "Failure"
	elif series == 'Failure':
		return "Failure"
	elif series == 'Success':
		return "Success"

df['SucessFail'] = df['Mission_Status'].apply(Success_Fail)
df.SucessFail.unique()


# Check a few things
df.Company_Name.unique()
df.Mission_Status.unique()
df.Datum.unique()
df.head()


# get our date info
df.Datum.describe
df.Datum[0:2]

# function to return Date Portion of string
def get_date_chars(string):
    return str(string)[4:16]

df["date_string"] = df['Datum'].apply(get_date_chars)
df.date_string.describe

# Strip out the commas from date field
df["date_string"] = df.date_string.replace(',','', regex=True)
df.date_string.describe

def get_date_time(string):
    return datetime.strptime(string, "%b %d %Y")

df["date_time"] = df['date_string'].apply(get_date_time)
df.date_time.describe

# Get Fields for Year and Month  
df['YEAR'] = pd.DatetimeIndex(df['date_time']).year
df['MONTH'] = pd.DatetimeIndex(df['date_time']).month
df['WEEK'] = df['date_time'].dt.isocalendar().week


# get Field for weekday
def get_date_chars(string):
    return str(string)[0:3]

df["WEEKDAY"] = df['Datum'].apply(get_date_chars)


# Clean up Rocket_Price
df.Rocket_Price.unique()
# strip commas
df["Rocket_Price"] = df.Rocket_Price.replace(',','', regex=True)
# Convert Rocket Price to numeric
df['Rocket_Price'] = pd.to_numeric(df['Rocket_Price'])


# Get our Datafields we want to play with
space_df = df[['SucessFail', 'Company_Name', 'Rocket_Price', 'date_time', 'YEAR', 'MONTH', 'WEEK', 'WEEKDAY']]

# create dummy columns for success/fail
sf = pd.get_dummies(space_df['SucessFail'])
# combine resulting dataframe with orginal
frames = [space_df, sf]
space_df = pd.concat(frames, axis=1)


##################################
######### questions:
## Which company has highest success rate? 
	# bar plot of %success by company
	# is it sig? 

# Create df with each row = company, columns ['Company_Name' 'success_rate', 'n']
# Get Success Rate by company
success_rate = space_df.groupby(['Company_Name'])['Success'].mean()
# Get N
company_n = space_df.Company_Name.value_counts()

# Combine summary dat
frames = [success_rate, company_n]
company_df = pd.concat(frames, axis=1)
company_df = company_df.rename(columns={"Success": "success_rate", "Company_Name": "Launches"})
company_df['Company_Name'] = company_df.index

# plot success rate by company
company_df.success_rate.sort_values().plot(kind="bar", use_index=True)
plt.show()

# only consider Companies with more than 10 launches
company_df_g10 = company_df[company_df['Launches'] >= 10]

# plot success rate by company for > 10 launches
company_df_g10.success_rate.sort_values().plot(kind="bar", use_index=True)
plt.show()

# look at company data & launch numbers
company_df_g10.sort_values(by=['success_rate'], ascending=False)


#############################
######### questions:
## has success rate improved through time? 
	# line plot of %success through time
	# is it sig? beta regression with year as X
		# comment on sample size through time

# Create df with each row = year, columns ['YEAR' 'success_rate', 'n']
# Get Success Rate by year
success_rate = space_df.groupby(['YEAR'])['Success'].mean()
# Get N
year_launches = space_df.YEAR.value_counts()
# Get mean money spent by year
rocket_costs = space_df.groupby(['YEAR'])['Rocket_Price'].mean()

# Combine summary dat
frames = [success_rate, year_launches, rocket_costs]
YEAR_df = pd.concat(frames, axis=1)
YEAR_df = YEAR_df.rename(columns={"Success": "success_rate", "YEAR": "Launches"})
YEAR_df['YEAR'] = YEAR_df.index

# Plot sucess by year
YEAR_df.sort_values(by=['YEAR']).success_rate.plot(kind="line", use_index=True)
plt.show()

# Smooth data with 3 year running average
YEAR_df['success_mv_ave'] = YEAR_df.sort_values(by=['YEAR']).success_rate.rolling(window=5).mean()

# Plot smoothed sucess by year
YEAR_df.sort_values(by=['YEAR']).success_mv_ave.plot(kind="line", use_index=True)
plt.show()

# Plot Price by year
YEAR_df.sort_values(by=['YEAR']).Rocket_Price.plot(kind="line", use_index=True)
plt.show()


X = YEAR_df[['YEAR', 'Launches']]
y = YEAR_df['success_rate']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .30, random_state=42) 
lm_model = LinearRegression(normalize=True) 
lm_model.fit(X_train, y_train) 
y_test_preds = lm_model.predict(X_test)
y_train_preds = lm_model.predict(X_train) 
train_score = r2_score(y_train, y_train_preds)
test_score = r2_score(y_test, y_test_preds)

print(train_score)
print(test_score)

######### MODEL COEFF WEIGHTS
def coef_weights(coefficients, X_train):    
    coefs_df = pd.DataFrame()
    coefs_df['est_int'] = X_train.columns
    coefs_df['coefs'] = lm_model.coef_
    coefs_df['abs_coefs'] = np.abs(lm_model.coef_)
    coefs_df = coefs_df.sort_values('abs_coefs', ascending=False)
    return coefs_df

#Use the function
coef_df = coef_weights(lm_model.coef_, X_train)

#A quick look at the top results
coef_df.head(20)



plt.plot(y_train, y_train_preds, 'o', alpha=0.2);
plt.show()


#############################
######### questions:
## Has day of week of launch changed through time? 

# add dummy variables for each day of week
wkdy = pd.get_dummies(space_df['WEEKDAY'])
# combine resulting dataframe with orginal
frames = [space_df, wkdy]
space_df = pd.concat(frames, axis=1)

# Create df with each row = year, columns ['YEAR' 'most ', 'n']
# Get Success Rate by year
year_launches = space_df.YEAR.value_counts()
def create_pct_var(df, joindat, cat_cols):
	for col in  cat_cols:
			try:
                # for each cat add dummy var, drop original column
				x = df.groupby(['YEAR'])[col].mean()
				frames = [joindat, x]
				joindat = pd.concat(frames, axis=1)
			except:
				continue
	return joindat

cols_lst = space_df.WEEKDAY.unique()
days_of_week_pct = create_pct_var(space_df, year_launches, cols_lst)

# Plot days of week by year
days_of_week_pct[['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']].plot(kind="bar", use_index=True)
plt.show()


#############################
######### questions:
## Has launch week changed through time? 
# Get Success Rate by year
success_rate = space_df.groupby(['YEAR'])['Success'].mean()
# Get N
year_launches = space_df.YEAR.value_counts()
# Get mean week by year
mean_week = space_df.groupby(['YEAR'])['WEEK'].mean()

# Combine summary dat
frames = [success_rate, year_launches, mean_week]
YEAR_df = pd.concat(frames, axis=1)
YEAR_df = YEAR_df.rename(columns={"Success": "success_rate", "YEAR": "Launches"})
YEAR_df['YEAR'] = YEAR_df.index


# Plot WEEK by year
YEAR_df.sort_values(by=['YEAR']).WEEK.plot(kind="line", use_index=True)
plt.show()

X = YEAR_df[['YEAR', 'WEEK', 'Launches']]
y = YEAR_df['success_rate']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .30, random_state=42) 
lm_model = LinearRegression(normalize=True) 
lm_model.fit(X_train, y_train) 
y_test_preds = lm_model.predict(X_test)
y_train_preds = lm_model.predict(X_train) 
train_score = r2_score(y_train, y_train_preds)
test_score = r2_score(y_test, y_test_preds)

print(train_score)
print(test_score)
# week does not change much by year, and does not influence success rate. Need better stats. 


#############################
######### questions:
# Does sucess improve with experience?
# Create column with total number of launces by year for that particular company
space_df['Index1'] = 1
space_df['CumLaunches'] = 1

joindat = pd.DataFrame(columns = space_df.columns)

def create_sum_launches(df, joindat, companies):
	for cp in  companies:
			try:
				x = df[df['Company_Name'] == cp].sort_values(by=['date_time'])
				x['CumLaunches'] = x.Index1.cumsum()
				frames = [joindat, x]
				joindat = pd.concat(frames, axis=0)
			except:
				continue
	return joindat

companies = space_df.Company_Name.unique()
cumulative_launches_df = create_sum_launches(space_df, joindat, companies)

cumulative_launches_df.shape
space_df.shape
cumulative_launches_df.columns


# Plot success by cumulative launches
plt.plot(cumulative_launches_df.CumLaunches, cumulative_launches_df.Success, 'o', color='black');
plt.show()


X = cumulative_launches_df[['CumLaunches', 'YEAR']]
y = cumulative_launches_df['Success']
y=y.astype('int')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .30, random_state=42) 

# logistic Regression
clf = LogisticRegression(random_state=0).fit(X_train, y_train)
y_test_preds = clf.predict(X_test)
y_train_preds = clf.predict(X_train) 


lm_model = LinearRegression(normalize=True) 
lm_model.fit(X_train, y_train) 
y_test_preds = lm_model.predict(X_test)
y_train_preds = lm_model.predict(X_train) 


train_score = r2_score(y_train, y_train_preds)
test_score = r2_score(y_test, y_test_preds)
print(train_score)
print(test_score)

plt.plot(y_train_preds, y_train, 'o', color='black');
plt.show()



