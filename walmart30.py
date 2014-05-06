import pandas as pd
import itertools
from pandas.stats.moments import ewma
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import scipy as scipy
import datetime

from scipy.interpolate import UnivariateSpline
from dateutil.relativedelta import relativedelta

import statsmodels.api as sm
from sklearn import linear_model
from sklearn import preprocessing

date0 = datetime.date(2010,2,5)

#Labour Day: first monday in sept
#Super Bowl: first sunday in feb
#thanksgiving: fourth thursday in november
#memorial day: the last monday in may
#easter: sunday

#length of sales week before xmas day
shoppingdaysbeforexmas= [0, 1, 3, 4]
#days between football and valentines day
# [2,3,4,6]

#easter, schoolstart, valentines, hallowe'en, christmas, 
holiday = [['2010-04-04', '2011-04-24','2012-04-08','2013-03-31']]
tgiving = [['2010-11-26', '2011-11-25', '2012-11-23', '2013-11-29']]
firstdayschool = [['2010-09-03', '2011-09-02', '2012-09-07', '2013-09-06']]
#holiday.append(['2010-09-01','2011-09-01','2012-09-04','2013-09-03'])
# holiday.append(['2010-02-14','2011-02-14','2012-02-14','2013-02-14'])
# holiday.append(['2010-10-31','2011-10-31','2012-10-31','2013-10-31'])
# holiday.append(['2010-12-25','2011-12-25','2012-12-25','2013-12-25'])

#date of week is the last friday
holweek = [['2010-02-12', '2011-02-11', '2012-02-10', '2013-02-08']] #superbowl
holweek.append(['2010-09-10', '2011-09-09', '2012-09-07', '2013-09-06']) #laborday
holweek.append(['2010-11-26', '2011-11-25', '2012-11-23', '2013-11-29']) #_thanksgiving
holweek.append(['2010-12-31', '2011-12-30', '2012-12-28', '2013-12-27']) #_christmas
holweek.append(['2010-02-19', '2011-02-18', '2012-02-17', '2013-02-15']) #valentines
holweek.append(['2010-04-09', '2011-04-29','2012-04-13','2013-04-05']) #easter


#stores, features, train, submission = doload()
def doload():
	print 'loading stores'
	stores = pd.read_csv('stores.csv')
	print 'loading features'
	features = pd.read_csv('features.csv')
	print 'loading training data'
	train = pd.read_csv('train.csv')
	#print 'loading test data'
	
	submission = pd.read_csv('sampleSubmission.csv')
	
	return stores, features, train, submission
	
	
def get_tgiving(year):
	for i in range(len(holweek[2])):
		
		if year == datetime.datetime.strptime(holweek[2][i], "%Y-%m-%d").year:
			return holweek[2][i]

	return holweek[2][0]
	
#week = get_week(train.Date[0:2].values)
def get_week(train_date):
	''' splits the year_month column to year and month and returns them in int format'''
	week =[]
	for i in train_date:
		#print i
		y,m,d = i.split('-')
		#print np.int(y),np.int(m),np.int(d)
		datetime.date
		week.append(datetime.date(np.int(y),np.int(m),np.int(d)))
	week = [np.int(np.ceil((w-date0).days/7.0)) for w in week]	
	return week
	
def get_day(train_date):
	''' splits the year_month column to year and month and returns them in int format'''
	day =[]
	for i in train_date:
		#print i
		y,m,d = i.split('-')
		#print np.int(y),np.int(m),np.int(d)
		datetime.date
		day.append(datetime.date(np.int(y),np.int(m),np.int(d)))
	day = [(w-date0).days for w in day]	
	return day
	
	
def diff(train):
	diff2010 = []
	diff2011 = []
# 	for store in range(1,46):#range(1,46):
# 		for dept in range(1,100):#range(1,100):
# 			xmas2010_week0_sales = train[train.Date == '2010-12-31'][np.logical_and(train.Store==store, train.Dept==dept)].Weekly_Sales.values
# 			xmas2010_week1_sales = train[train.Date == '2010-12-24'][np.logical_and(train.Store==store, train.Dept==dept)].Weekly_Sales.values
# 			xmas2011_week0_sales = train[train.Date == '2011-12-30'][np.logical_and(train.Store==store, train.Dept==dept)].Weekly_Sales.values
# 			xmas2011_week1_sales = train[train.Date == '2011-12-23'][np.logical_and(train.Store==store, train.Dept==dept)].Weekly_Sales.values
# 	
# 			if len(xmas2010_week0_sales) == 1 and len(xmas2010_week1_sales) == 1 and len(xmas2011_week0_sales) == 1 and len(xmas2011_week1_sales) == 1:
# 				diff2010.append(xmas2010_week1_sales - xmas2010_week0_sales)
# 				diff2011.append(xmas2011_week1_sales - xmas2011_week0_sales)


	sales= train[np.logical_or(np.logical_or(train.Date == '2010-12-31', train.Date == '2010-12-24'),np.logical_or(train.Date == '2011-12-30', train.Date == '2011-12-23'))]

	#sales_tot = sales[['Date', 'Store', 'Dept']].groupby(['Store', 'Dept']).agg(['count'])
	table = pd.pivot_table(sales, 'Weekly_Sales', rows=['Store', 'Dept'], cols=['Date'])
	table2 = table[pd.notnull(table['2010-12-31'])][pd.notnull(table['2010-12-24'])][pd.notnull(table['2011-12-30'])][pd.notnull(table['2011-12-23'])]

	diff2010 = table2['2010-12-24'].values -table2['2010-12-31'].values
	diff2011 = table2['2011-12-23'].values -table2['2011-12-30'].values
	
	print np.mean(diff2010/diff2011)
	plt.plot(diff2010, diff2011, 'k.')
	
#prepped_features =prep_features(features)	
def prep_features(features):
	f = features
	f['Week_Num'] = get_week(f.Date.values)	
	
	for store in range(1,46):
		
		for feat in ['MarkDown1','MarkDown2','MarkDown3','MarkDown4','MarkDown5']:
			for i in range(0,52):
				#oldmedian = f[feat].median()
				fill = f[feat][f.Store==store][f[f.Store==store].Week_Num.isin([0+i,52+i,104+i,156+i])].mean()
				f[feat][f.Store==store][f[f.Store==store].Week_Num.isin([0+i,52+i,104+i,156+i])] = f[feat][f.Store==store][f[f.Store==store].Week_Num.isin([0+i,52+i,104+i,156+i])].fillna(fill) #+0.0*oldmedian

			f[feat][f.Store==store] = f[feat][f.Store==store].fillna(0.0)
# 			f[feat][f.Store==store] = f[feat][f.Store==store].fillna(method='pad')
# 			f[feat][f.Store==store] = f[feat][f.Store==store].fillna(method='backfill')
			
 		for var in ['CPI', 'Unemployment', 'Fuel_Price']:
 			#f[var][f.Store==store] = f[var][f.Store==store].fillna(method='pad')

			#l = len(f[var][f.Date >'2012-10-26'])
			values = f[var][np.logical_and(f.Store==store,np.logical_and(f.Date <='2013-04-19', f.Date > '2013-02-01'))].values
			week  = get_week(f['Date'][np.logical_and(f.Store==store, np.logical_and(f.Date <='2013-04-19', f.Date > '2013-02-01'))].values)
			
			k=1  # line parabola cubicspline
			extrapolator = UnivariateSpline( week, values, k=k )
			y = extrapolator( get_week(f['Date'][np.logical_and(f.Store==store,f.Date >='2013-05-03')]) )

			#print len(f[var][np.logical_and(f.Store==store,f.Date >='2013-05-03')]), len(y)
			f[var][np.logical_and(f.Store==store,f.Date >='2013-05-03')] = pd.Series(y, index=f[np.logical_and(f.Store==store,f.Date >='2013-05-03')].index)	
			
	return f		
	
#explore_markdown(features,1)	
def explore_markdown(features,store):	
	f = features[features.Store==store]
	
	
	f['Week_Num'] = get_week(f.Date.values)
	
	#f['MarkDown5'] = f['MarkDown5'].fillna(f['MarkDown5'].median())
	#f['MarkDown5'] = f['MarkDown5']*1.0
	
	
	for feat in ['MarkDown1','MarkDown2','MarkDown3','MarkDown4','MarkDown5']:
		for i in range(0,52):
			oldmedian = f[feat].median()
			#print f['MarkDown5'][f.Week_Num.isin([0+i,52+i,104+i,156+i])]
			#fill = f['MarkDown5'][f.Week_Num.isin([0+i,52+i,104+i,156+i])].mean()*0.4+0.5*oldmedian
			fill = f[feat][f.Week_Num.isin([0+i,52+i,104+i,156+i])].min()
			f[feat][f.Week_Num.isin([0+i,52+i,104+i,156+i])] = f[feat][f.Week_Num.isin([0+i,52+i,104+i,156+i])].fillna(fill) #+0.0*oldmedian
			#print f['MarkDown5'][f.Week_Num.isin([0+i,52+i,104+i,156+i])]
		
		f[feat] = f[feat].fillna(method='pad')
		f[feat] = f[feat].fillna(method='backfill')
#	plt.plot(f.Week_Num, f.MarkDown1)
# 	plt.plot(f.Week_Num, f.MarkDown2)
# 	plt.plot(f.Week_Num, f.MarkDown3)
# 	plt.plot(f.Week_Num, f.MarkDown4)
# 	plt.plot(f.Week_Num, f.MarkDown5)
	
	
	var =  'Unemployment'#'CPI' #
	plt.plot(f.Week_Num, f[var], 'k-')
	
	
	l = len(f[var][f.Date >'2012-10-26'])
	values = f[var][np.logical_and(f.Date <='2013-04-19', f.Date > '2013-02-01')].values
	week  = get_week(f['Date'][np.logical_and(f.Date <='2013-04-19', f.Date > '2013-02-01')].values)
	
	#for k in (1):  # line parabola cubicspline
	k = 1
	extrapolator = UnivariateSpline( week, values, k=k )
	y = extrapolator( get_week(f['Date'][f.Date >='2013-05-03']) )

	f[var][f.Date >='2013-05-03'] = pd.Series(y, index=f[f.Date >='2013-05-03'].index )
	plt.plot(f.Week_Num, f[var], 'r-')
	raw_input('press a key')
	plt.close()	
	
		
	
def plotting(p):	
	plt.plot(p[p.Date <= '2012-10-26'].Days_Since_Paid, p[p.Date <= '2012-10-26'].Weekly_Sales, 'k.')	
	plt.xlabel('Days Since Pay Day')
	plt.ylabel('Sales')
	plt.title('Store 28 Dept 92')

	
	
#pred2 = correct_around_xmas(pred2, time, idx, train)	
def correct_around_xmas(pred2, time, idx, train):	
	xmas2010 = get_week([holweek[3][0]])[0]
	xmas2010_sales = train[train.Date==holweek[3][0]].Weekly_Sales.values

	weekbeforexmas2010 = (datetime.datetime.strptime(holweek[3][0], "%Y-%m-%d")-datetime.timedelta(days=7)).date().isoformat()
	weekbeforexmas2010_sales = train[train.Date==weekbeforexmas2010].Weekly_Sales.values

	
	if len(xmas2010_sales) == 1 and len(weekbeforexmas2010_sales) ==1:
		xmas2010_sales = xmas2010_sales[0]
		weekbeforexmas2010_sales = weekbeforexmas2010_sales[0]

		#sales per day in the week leading up to christmas
		salesperday = 0.15*(weekbeforexmas2010_sales - xmas2010_sales)*(weekbeforexmas2010_sales > xmas2010_sales)
	
		for i in range(1,4):
			this_xmas = get_week([holweek[3][i]])[0]
			#print this_xmas
			#print this_xmas, time[0]
			if this_xmas in time:
				now = time.index(this_xmas)
				pred2[now] += salesperday*shoppingdaysbeforexmas[i]
			if this_xmas-1 in time:
				now = time.index(this_xmas-1)
				pred2[now] -= salesperday*shoppingdaysbeforexmas[i]
				
	return pred2			

	
	
#pred2 = correct_around_easter(pred2, time, idx)	
def correct_around_easter_old(pred2, time, idx):

	for hol in holiday:
		for i in range(len(hol)-1):
			prevdate = get_week([hol[i]])[0]
			currdate = get_week([hol[i+1]])[0]		
			if prevdate in idx:
				diff = 52 - (currdate-prevdate)

				#holiday is not the same week in the next year
				if diff != 0:
					start = np.max((currdate-3-time[0], 0))
					end = np.max((0, np.min((currdate+3-time[0], len(pred2))) ))
					
					#if there is a substantial peak in the data
					mn = np.min(pred2)
					avg = np.mean(pred2)
					#if (end > start) and np.max(pred2[start:end]) > avg+5.0*(avg-min):
					if (end > start) and np.max(pred2[start:end]) > 1.3*np.mean(pred2[start:end]):	
						start2 = np.max((0, start+diff))
						end2 = np.min((len(pred2), end+diff))
						if len(pred2[start:start+(end2-start2)]) == len(pred2[start2:end2] ):
							pred2[start:start+(end2-start2)] = pred2[start2:end2] 
							
	return pred2
	
	
	
def correct_around_easter(pred2, time, idx):

	for hol in holiday:
		for i in range(len(hol)-1):
			prevdate = get_week([hol[i]])[0]
			currdate = get_week([hol[i+1]])[0]		
			if prevdate in idx:
				diff = 52 - (currdate-prevdate)

				#holiday is not the same week in the next year
				if diff > 0:
					start = np.max((currdate-3-time[0], 0))
					end = np.max((0, np.min((currdate+3-time[0]+diff, len(pred2))) ))
				if diff < 0:
					start = np.max((currdate-3-time[0]+diff, 0))
					end = np.max((0, np.min((currdate+3-time[0], len(pred2))) ))

				if diff != 0 and (end > start):	
					#if there is a substantial peak in the data
					try:
						m = np.median([i for i in pred2[np.min((start-5,0)):np.max((end+5, len(pred2)))] if i >= 0])
						min = np.min([i for i in pred2[np.min((start-5,0)):np.max((end+5, len(pred2)))] if i >= 0])
					except:
						m = np.nan
						min = np.nan
#					m = np.mean(pred2[start:end])
#					min = np.min(pred2[start:end])
#					print m, min, m + 1.0*(m-min), np.max(pred2[np.max((start-5,0)):np.min((end+5, len(pred2)))])
					if  np.max(pred2[np.max((start-5,0)):np.min((end+5, len(pred2)))]) > m + 1.0*(m-min):
						#print 'did shift'
						start2 = np.max((0, start+diff))
						end2 = np.min((len(pred2), end+diff))
						#print start2, end2, len(pred2)
						if len(pred2[start:start+(end2-start2)]) == len(pred2[start2:end2] ):
							pred2[start:start+(end2-start2)] = pred2[start2:end2] 
							
	return pred2	
	
	
	
def correct_around_labour(pred2, time, idx):

	for hol in firstdayschool:
		for i in range(len(hol)-1):
			prevdate = get_week([hol[i]])[0]
			currdate = get_week([hol[i+1]])[0]		
			if prevdate in idx:
				diff = 52 - (currdate-prevdate)
				#holiday is not the same week in the next year
				if diff > 0:
					start = np.max((currdate-3-time[0], 0))
					end = np.max((0, np.min((currdate+3-time[0]+diff, len(pred2))) ))
				if diff < 0:
					start = np.max((currdate-3-time[0]+diff, 0))
					end = np.max((0, np.min((currdate+3-time[0], len(pred2))) ))

				if diff != 0 and (end > start):	
					#if there is a substantial peak in the data
					try:
						m = np.median([i for i in pred2[np.min((start-5,0)):np.max((end+5, len(pred2)))] if i >= 0])
						min = np.min([i for i in pred2[np.min((start-5,0)):np.max((end+5, len(pred2)))] if i >= 0])
					except:
						m = np.nan
						min = np.nan
#					m = np.mean(pred2[start:end])
#					min = np.min(pred2[start:end])
#					print m, min, m + 1.0*(m-min), np.max(pred2[np.max((start-5,0)):np.min((end+5, len(pred2)))])
					if  np.max(pred2[np.max((start-5,0)):np.min((end+5, len(pred2)))]) > m + 1.0*(m-min):
						#print 'did shift'
						start2 = np.max((0, start+diff))
						end2 = np.min((len(pred2), end+diff))
						#print start2, end2, len(pred2)
						if len(pred2[start:start+(end2-start2)]) == len(pred2[start2:end2] ):
							pred2[start:start+(end2-start2)] = pred2[start2:end2] 
							
	return pred2	
	

#plotsales(train, features, 1,93,[],[])
def plotsales(train, features, store, dept, vars, newvars):
	p = gen_p(train, features, store, dept, vars, newvars)
	
	plt.plot(p.Week_Num, p.Weekly_Sales, 'k.-')
	plt.plot(p.Week_Num, p.Sales_52weeksprior, 'r.-')
	

	
	#datetime.datetime.strptime('1981-06-04', "%Y-%m-%d").month
	paycheque_weeks = [w for w in p.Date if (datetime.datetime.strptime(w, "%Y-%m-%d")+datetime.timedelta(days=-7*1)).month == datetime.datetime.strptime(w, "%Y-%m-%d").month-1]
	paycheque_weeks_52weeksprior = [w for w in p.Date_52weeksprior if (datetime.datetime.strptime(w, "%Y-%m-%d")+datetime.timedelta(days=-7*1)).month == datetime.datetime.strptime(w, "%Y-%m-%d").month-1]

	
	plt.plot(p['Week_Num'][p.Date.isin(paycheque_weeks)], p['Weekly_Sales'][p.Date.isin(paycheque_weeks)], 'k*')
	plt.plot(p['Week_Num'][p.Date_52weeksprior.isin(paycheque_weeks_52weeksprior)], p['Sales_52weeksprior'][p.Date_52weeksprior.isin(paycheque_weeks_52weeksprior)], 'r*')
	raw_input('press a key')
	plt.close()	

	week_before_paycheque = [(datetime.datetime.strptime(w, "%Y-%m-%d")+datetime.timedelta(days=-7*1)).date().isoformat() for w in paycheque_weeks if (datetime.datetime.strptime(w, "%Y-%m-%d")+datetime.timedelta(days=-7)).month == datetime.datetime.strptime(w, "%Y-%m-%d").month-1]

	#days between paycheque and end of friday of paycheque week
	#if len(paycheque_weeks)  != len(week_before_paycheque):
	print len(paycheque_weeks), len(week_before_paycheque)
	paycheque_weeks = paycheque_weeks[1:]
		
	diff = p['Weekly_Sales'][p.Date.isin(paycheque_weeks)].values - p['Weekly_Sales'][p.Date.isin(week_before_paycheque)].values
	diff_days = np.array( [-(datetime.date(datetime.datetime.strptime(w, "%Y-%m-%d").year, datetime.datetime.strptime(w, "%Y-%m-%d").month, 1)+relativedelta(months=0)-datetime.datetime.strptime(w, "%Y-%m-%d").date() ).days   for w in p['Date'][p.Date.isin(paycheque_weeks)].values])
	
	median = np.zeros(7)
	for j in range(0,6):
		idx = [i for i in range(len(diff_days)) if diff_days[i]==j]
		median[j] = np.median(diff[idx])
	
	plt.plot(diff_days, diff, 'k.')
	plt.plot(range(0,7), median, 'r*')
	
	raw_input('press a key')
	plt.close()	
	
	


#pred, time = make_prediction2(range(91,143), train[np.logical_and(train.Store==9, train.Dept==57), features[np.logical_and(features.Store==9)])
def make_prediction2(predictweeks, train, features):
	
	assert predictweeks[-1] - predictweeks[0] <= 52, "prediction period longer than 1 year" 
	
	week = get_week(train.Date.values)
	idx = [i for i in range(len(week)) if week[i]+52 in predictweeks]
	
	if not len(idx) > 1:
		return np.zeros(len(predictweeks)), predictweeks
	
	pred = train.iloc[idx].Weekly_Sales.values
	time = np.array(get_week(train.iloc[idx].Date.values))+52
	
	pred2 = np.zeros(len(predictweeks))
	pred2[0] = pred[0]
	
	j = 1
 	for i in range(1,len(predictweeks)):
 		if j < len(time):
 			if predictweeks[i] < time[j]:
				pred2[i] = pred2[i-1]	
 				if i >= 2 and pred2[i-1] != pred2[i-2]:
 					pred2[i] = pred2[i-1]
 				else:
 					pred2[i] = 0.0
 			else:
 				pred2[i] = pred[j]
 				j = j+1
 		else:
 			break
	
	#adjusting for movement of easter relative to fridays
	if not train.Dept.iloc[0] == 67:
		pred2 = correct_around_easter(pred2, predictweeks, idx)

	if train.Dept.iloc[0] == 3 and train.Store.iloc[0] in [1,2,3,5,6,8,9,10,11,13,16,21,30,31,36,37,39,44]:
		pred2 = correct_around_labour(pred2, predictweeks, idx)							

	#adjust for number of shopping days before christmas
	pred2 = correct_around_xmas(pred2, predictweeks, idx, train)
				
	return pred2, predictweeks

	
def get_days_until_paid(w):
	m = datetime.datetime.strptime(w, "%Y-%m-%d").month+1
	y = datetime.datetime.strptime(w, "%Y-%m-%d").year
	if m == 13:
		y = y+1
		m = 1
	d = 1
	return datetime.date(y,m,d)
	
#p = gen_p(train, features, 41,99, vars, newvars)	
#p = gen_p(train, features, 1,99, vars, newvars)	
def gen_p(train, features, store, dept, vars, newvars):

	train = train[np.logical_and(train.Store==store, train.Dept==dept)]
	assert len(train) >= 1, 'no data in train'
		
	p = pd.merge(train, features, on=['Store', 'Date', 'IsHoliday'])

#	p['Weekly_Sales'] = p['Weekly_Sales'].fillna(0.0)

	p['Weekly_Sales'] = 1.0*p['Weekly_Sales']
	p['Weekly_Sales'][np.logical_and(p.Weekly_Sales < 1, p.Weekly_Sales >0)] = 0.0
	p['Weekly_Sales'][p.Weekly_Sales < 0] = np.nan*1.0
	
	
# 	if (1.0*(p.Weekly_Sales < 30)).mean() > 0.6  or len(p['Weekly_Sales'][np.logical_and(p.Date >='2011-01-28', p.Date <='2012-10-26')]) < 30:
# 		p['Weekly_Sales'] = p['Weekly_Sales'].fillna(0.0)
# 	else:
# 		if p.Weekly_Sales.median() < 250:
# 			p['Weekly_Sales'] = p['Weekly_Sales'].fillna(p.Weekly_Sales.median())		
# 		else:
# 			p['Weekly_Sales'] = p['Weekly_Sales'].fillna(method='pad')
# 			p['Weekly_Sales'] = p['Weekly_Sales'].fillna(method='backfill')
# 			
# 	p['Weekly_Sales'] = p['Weekly_Sales'].fillna(0.0)
	
		
	lastdate = p.Date.values[len(p)-1]
	p = p.append(features[np.logical_and(features.Store==store, features.Date > lastdate)] )
	
	p['Dept'] = dept
	
	#fill in missing values
	for var in vars:	
		if not 'MarkDown' in var:
			p[var] = p[var].fillna(method='pad')
		else:	
			p[var] = p[var].fillna(method='pad')
			#p[var] = p[var].fillna(p[var].median())
	
	p['Week_Num'] = pd.Series(get_week(p.Date.values), index=p.index)
	
	#p['Date_52weeksprior'] = pd.Series( [(datetime.datetime.strptime(i, "%Y-%m-%d")+datetime.timedelta(days=-7*52)).date().isoformat() for i in p.Date.values] )
	#p['Sales_52weeksprior'] = pd.Series( [p[p.Date ==(datetime.datetime.strptime(i, "%Y-%m-%d")+datetime.timedelta(days=-7*52)).date().isoformat()].Weekly_Sales.values for i in p.Date.values] )
	#values = [np.append(p[p.Date ==(datetime.datetime.strptime(i, "%Y-%m-%d")+datetime.timedelta(days=-7*52)).date().isoformat()].Week_Num.values, np.nan*np.ones(1))[0] for i in p.Date.values]
	#p['Week_Num_52weeksprior'] = pd.Series( values )
	#p['Week_Num_52weeksprior'] = pd.Series( p[p.Date ==(datetime.datetime.strptime(i, "%Y-%m-%d")+datetime.timedelta(days=-7*52)).date().isoformat()].Week_Num.values for i in p.Date.values)
	p['Week_Num_52weeksprior'] = 1.0*(p['Week_Num']).values - 52
	p[p.Week_Num_52weeksprior < 0]['Week_Num_52weeksprior'] = np.nan
	#p['Date_52weeksprior'] = pd.Series( [np.append(p[p.Week_Num ==p.Week_Num_52weeksprior+52]['Date'].values, np.nan*np.ones(1))[0] for i in p.Date.values] )
	#p['Date_52weeksprior'] = pd.Series( [np.append((datetime.datetime.strptime(i, "%Y-%m-%d")+datetime.timedelta(days=-7*52)).date().isoformat(), np.nan*np.ones(1))[0] for i in p.Date.values] )
	p['Date_52weeksprior'] = pd.Series( [(date0+datetime.timedelta(days=7*i)).isoformat() for i in p.Week_Num_52weeksprior.values], index=p.index )
	#print p[['Week_Num_52weeksprior','Date_52weeksprior']]

	
	p['Days_Since_Paid'] =  pd.Series( [(datetime.datetime.strptime(w, "%Y-%m-%d").date()-datetime.date(datetime.datetime.strptime(w, "%Y-%m-%d").year, datetime.datetime.strptime(w, "%Y-%m-%d").month, 1) ).days for w in p.Date.values], index=p.index )
	p['Days_Since_Paid_52weeksprior'] =  pd.Series( [(datetime.datetime.strptime(w, "%Y-%m-%d").date()-datetime.date(datetime.datetime.strptime(w, "%Y-%m-%d").year, datetime.datetime.strptime(w, "%Y-%m-%d").month, 1) ).days for w in p.Date_52weeksprior.values], index=p.index )
# 	p['Days_Since_Paid_gt7'] =  1*(p['Days_Since_Paid'] <= 6)
# 	p['Days_Since_Paid_gt7_52weeksprior'] = 1*(p['Days_Since_Paid_52weeksprior'] <= 6)
#   p['Days_Since_Paid_gt14'] =  1.0*np.logical_and(p['Days_Since_Paid'] > 6, p['Days_Since_Paid'] < 14)
#   p['Days_Since_Paid_gt14_52weeksprior'] =  1.0*np.logical_and(p['Days_Since_Paid_52weeksprior'] > 6, p['Days_Since_Paid_52weeksprior'] < 14) 
	p['Days_Since_Paid_gt7'] =  -1*(p['Days_Since_Paid'] > 6) *p['Days_Since_Paid']
	p['Days_Since_Paid_gt7_52weeksprior'] = -1*(p['Days_Since_Paid_52weeksprior'] > 6)* p['Days_Since_Paid_52weeksprior']
  	p['Days_Since_Paid_gt14'] =  -1.0*np.logical_or(p['Days_Since_Paid'] <= 6, p['Days_Since_Paid'] >= 14)* p['Days_Since_Paid']
  	p['Days_Since_Paid_gt14_52weeksprior'] =  -1.0*np.logical_or(p['Days_Since_Paid_52weeksprior'] <= 6, p['Days_Since_Paid_52weeksprior'] >= 14)*p['Days_Since_Paid_52weeksprior']

	
	p['Days_Until_Paid'] =  pd.Series( [(datetime.datetime.strptime(w, "%Y-%m-%d").date()- get_days_until_paid(w)).days for w in p.Date.values], index=p.index )
	p['Days_Until_Paid_52weeksprior'] =  pd.Series( [(datetime.datetime.strptime(w, "%Y-%m-%d").date()-datetime.date(datetime.datetime.strptime(w, "%Y-%m-%d").year, np.mod(datetime.datetime.strptime(w, "%Y-%m-%d").month,12)+1, 1) ).days for w in p.Date_52weeksprior.values], index=p.index )
	p['Days_Until_Paid_gt7'] =  pd.Series( -1*(np.logical_or(p['Days_Until_Paid'].values < -21, p['Days_Until_Paid'].values >=0) ) *p['Days_Until_Paid'].values, index=p.index )
	p['Days_Until_Paid_gt7_52weeksprior'] = pd.Series( -1*(np.logical_or(p['Days_Until_Paid_52weeksprior'].values < -21, p['Days_Until_Paid_52weeksprior'].values >=0) ) *p['Days_Until_Paid'].values, index=p.index )



	p['Days_Before_Xmas'] =  pd.Series( [(datetime.datetime.strptime(w, "%Y-%m-%d").date()-datetime.date(datetime.datetime.strptime(w, "%Y-%m-%d").year, 12, 25) ).days for w in p.Date.values], index=p.index )
	p['Days_Before_Xmas_52weeksprior'] =  pd.Series( [(datetime.datetime.strptime(w, "%Y-%m-%d").date()-datetime.date(datetime.datetime.strptime(w, "%Y-%m-%d").year, 12,25) ).days for w in p.Date_52weeksprior.values], index=p.index )
	p['Days_Before_Xmas_gt7'] =  pd.Series( -1*(np.logical_or(p['Days_Before_Xmas'].values < -21, p['Days_Before_Xmas'].values >0) ) *p['Days_Before_Xmas'].values, index=p.index )
	p['Days_Before_Xmas_gt7_52weeksprior'] =  pd.Series( -1*(np.logical_or(p['Days_Before_Xmas_52weeksprior'].values < -21, p['Days_Before_Xmas_52weeksprior'].values >0) ) * p['Days_Before_Xmas_52weeksprior'].values, index=p.index )
 	p['Days_After_Xmas_gt7'] =  pd.Series( -1*(np.logical_or(p['Days_Before_Xmas'].values <= 0, p['Days_Before_Xmas'].values >=7) ) *p['Days_Before_Xmas'].values, index=p.index )
 	p['Days_After_Xmas_gt7_52weeksprior'] =  pd.Series( -1*(np.logical_or(p['Days_Before_Xmas_52weeksprior'].values <= 0, p['Days_Before_Xmas_52weeksprior'].values >=7) ) * p['Days_Before_Xmas_52weeksprior'].values, index=p.index )


	p['Days_Before_Hall'] =  pd.Series( [(datetime.datetime.strptime(w, "%Y-%m-%d").date()-datetime.date(datetime.datetime.strptime(w, "%Y-%m-%d").year, 10, 31) ).days for w in p.Date.values], index=p.index )
	p['Days_Before_Hall_52weeksprior'] =  pd.Series( [(datetime.datetime.strptime(w, "%Y-%m-%d").date()-datetime.date(datetime.datetime.strptime(w, "%Y-%m-%d").year, 10,31) ).days for w in p.Date_52weeksprior.values], index=p.index )
	p['Days_Before_Hall_gt7'] =  pd.Series( -1*(np.logical_or(p['Days_Before_Hall'].values <= -7, p['Days_Before_Hall'].values >0) ) *p['Days_Before_Hall'].values, index=p.index )
	p['Days_Before_Hall_gt7_52weeksprior'] =  pd.Series( -1*(np.logical_or(p['Days_Before_Hall_52weeksprior'].values <= -7, p['Days_Before_Hall_52weeksprior'].values >0) ) * p['Days_Before_Hall_52weeksprior'].values, index=p.index )
	p['Days_After_Hall_gt7'] =  pd.Series( -1*(np.logical_or(p['Days_Before_Hall'].values <= 0, p['Days_Before_Hall'].values >7) ) *p['Days_Before_Hall'].values, index=p.index )
	p['Days_After_Hall_gt7_52weeksprior'] =  pd.Series( -1*(np.logical_or(p['Days_Before_Hall_52weeksprior'].values <= 0, p['Days_Before_Hall_52weeksprior'].values >7) ) * p['Days_Before_Hall_52weeksprior'].values, index=p.index )



	p['Days_Before_Val'] =  pd.Series( [(datetime.datetime.strptime(w, "%Y-%m-%d").date()-datetime.date(datetime.datetime.strptime(w, "%Y-%m-%d").year, 2, 14) ).days for w in p.Date.values], index=p.index )
	p['Days_Before_Val_52weeksprior'] =  pd.Series( [(datetime.datetime.strptime(w, "%Y-%m-%d").date()-datetime.date(datetime.datetime.strptime(w, "%Y-%m-%d").year, 2,14) ).days for w in p.Date_52weeksprior.values], index=p.index )
	p['Days_Before_Val_gt7'] =  -1.0*(np.logical_or(p['Days_Before_Val'] <= -7, p['Days_Before_Val'] >0) ) * p['Days_Before_Val']
	p['Days_Before_Val_gt7_52weeksprior'] =  -1.0*(np.logical_or(p['Days_Before_Val_52weeksprior'] <= -7, p['Days_Before_Val_52weeksprior'] >0) ) * p['Days_Before_Val_52weeksprior']
	p['Days_After_Val_gt7'] =  -1.0*(np.logical_or(p['Days_Before_Val'] <= 0, p['Days_Before_Val'] >7) ) * p['Days_Before_Val']
	p['Days_After_Val_gt7_52weeksprior'] =  -1.0*(np.logical_or(p['Days_Before_Val_52weeksprior'] <= 0, p['Days_Before_Val_52weeksprior'] >7) ) * p['Days_Before_Val_52weeksprior']

	p['Days_Before_Labour'] =  pd.Series( [(datetime.datetime.strptime(w, "%Y-%m-%d").date()-datetime.date(datetime.datetime.strptime(w, "%Y-%m-%d").year, 9, 1) ).days for w in p.Date.values], index=p.index )
	p['Days_Before_Labour_52weeksprior'] =  pd.Series( [(datetime.datetime.strptime(w, "%Y-%m-%d").date()-datetime.date(datetime.datetime.strptime(w, "%Y-%m-%d").year, 9,1) ).days for w in p.Date_52weeksprior.values], index=p.index )
	p['Days_Before_Labour_gt7'] =  -1.0*(np.logical_or(p['Days_Before_Labour'] <= -28, p['Days_Before_Labour'] >0) ) * p['Days_Before_Labour']
	p['Days_Before_Labour_gt7_52weeksprior'] =  -1.0*(np.logical_or(p['Days_Before_Labour_52weeksprior'] <= -28, p['Days_Before_Labour_52weeksprior'] >0) ) * p['Days_Before_Labour_52weeksprior']
	p['Days_After_Labour_gt7'] =  -1.0*(np.logical_or(p['Days_Before_Labour'] <= 0, p['Days_Before_Labour'] >7) ) * p['Days_Before_Labour']
	p['Days_After_Labour_gt7_52weeksprior'] =  -1.0*(np.logical_or(p['Days_Before_Labour_52weeksprior'] <= 0, p['Days_Before_Labour_52weeksprior'] >7) ) * p['Days_Before_Labour_52weeksprior']

	
	week0 = p['Week_Num'].iloc[0]
	week1 = week0 + 52
	weeklast = p['Week_Num'].iloc[len(p)-1]
	#print week0, week1, weeklast

	weeksall = []
	pred = []
	
	pred = len(p[p.Week_Num < week1])*[np.nan]
	weeksall = list(p[p.Week_Num < week1].Week_Num.values)
			
	for i in range(0,1+np.int(np.ceil((weeklast-week1)/52.0))):
		#weeks = list(set(p[np.logical_and(p.Week_Num >= week1+(i)*52, p.Week_Num < week1+(i+1)*52)]['Week_Num'].values))
		weeks = list(p[np.logical_and(p.Week_Num >= week1+(i)*52, p.Week_Num < week1+(i+1)*52)]['Week_Num'].values)
		if len(weeks) > 0:
			pred2, time = make_prediction2(weeks, p, features[features.Store==store])
			pred = pred + list(pred2)
			#weeksall = weeksall + weeks
			weeksall = weeksall + time
		
	df = pd.DataFrame(zip(weeksall,pred), columns=['Week_Num','Sales_52weeksprior'])
	p = pd.merge(p, df, on=['Week_Num'])

#	p['Sales_52weeksprior'] = p['Sales_52weeksprior'].fillna(method='pad')
#	print np.mean(pd.isnull(p['Sales_52weeksprior']))
#  	if (p.Weekly_Sales < 30).mean() > 0.6:
#  		p['Sales_52weeksprior'] = p['Sales_52weeksprior'].fillna(0)
#  	else:

	
#	p['Sales_52weeksprior'] = p['Sales_52weeksprior'].fillna(0.0)
	p['Sales_52weeksprior'] = p['Sales_52weeksprior'].fillna(method='pad')
# 	if (1.0*(p.Sales_52weeksprior < 30)).mean() > 0.6:
# 		p['Sales_52weeksprior'] = p['Sales_52weeksprior'].fillna(0.0)
# 	else:
# 		if p.Weekly_Sales.median() < 250:
# 			p['Sales_52weeksprior'] = p['Sales_52weeksprior'].fillna(p.Weekly_Sales.median())	
# 			#p['Sales_52weeksprior'] = p['Sales_52weeksprior'].fillna(0.0)
# 		else:
# 			p['Sales_52weeksprior'] = p['Sales_52weeksprior'].fillna(method='pad')

#	p['Weekly_Sales'] = p['Weekly_Sales'].fillna(0.0)
	p['Weekly_Sales'] = p['Weekly_Sales'].fillna(method='pad')
# 	if (1.0*(p.Weekly_Sales < 30)).mean() > 0.6  or len(p['Weekly_Sales'][np.logical_and(p.Date >='2011-01-28', p.Date <='2012-10-26')]) < 100:
# 		p['Weekly_Sales'] = p['Weekly_Sales'].fillna(0.0)
# 	else:
# 		if p.Weekly_Sales.median() < 250:
# 			p['Weekly_Sales'] = p['Weekly_Sales'].fillna(p.Weekly_Sales.median())		
# 		else:
# 			p['Weekly_Sales'] = p['Weekly_Sales'].fillna(method='pad')
# 			p['Weekly_Sales'] = p['Weekly_Sales'].fillna(method='backfill')

# 	if p.Weekly_Sales.median() < 250:
# 		p['Sales_52weeksprior'] = p['Sales_52weeksprior'].fillna(p.Sales_52weeksprior.median())		
# 	else:
# 		p['Sales_52weeksprior'] = p['Sales_52weeksprior'].fillna(method='pad')
# 		p['Sales_52weeksprior'] = p['Sales_52weeksprior'].fillna(method='backfill')
# 	p['Sales_52weeksprior'] = p['Sales_52weeksprior'].fillna(0.0)	
	
	for var in vars:
#		p[var+'_52weeksprior'] = pd.Series( [p[p.Date ==(datetime.datetime.strptime(i, "%Y-%m-%d")+datetime.timedelta(days=-7*52)).date().isoformat()][var].values for i in p.Date.values] )			
#		p[var+'_52weeksprior'] = pd.Series( [np.append(p[p.Week_Num ==p.Week_Num_52weeksprior+52][var].values, np.nan*np.ones(1))[0] for i in p.Date.values] )			
		p[var+'_52weeksprior'] = pd.Series( [np.append(p[p.Date ==(datetime.datetime.strptime(i, "%Y-%m-%d")+datetime.timedelta(days=-7*52)).date().isoformat()][var].values, np.nan*np.ones(1))[0] for i in p.Date.values] )			

		
	for var in vars:
		#p[var+'_52weeksprior'] = p[var+'_52weeksprior'].fillna(0.0)
		#p[var+'_52weeksprior'] = p[var+'_52weeksprior'].fillna(method='pad')	
		if (p.Weekly_Sales < 30).mean() > 0.6:
			p[var+'_52weeksprior'] = p[var+'_52weeksprior'].fillna(p[var+'_52weeksprior'].median())
		else:	
			p[var+'_52weeksprior'] = p[var+'_52weeksprior'].fillna(method='pad')
			p[var+'_52weeksprior'] = p[var+'_52weeksprior'].fillna(method='backfill')
# 		p[var+'_52weeksprior'] = p[var+'_52weeksprior'].fillna(0.0)
	
# 	w = p[p.IsHoliday==True].Week_Num.values 
# 	w1 = p[p.IsHoliday==True].Week_Num.values+1
# 	w2 = p[p.IsHoliday==True].Week_Num.values-1

#	df = p[p.Date < '2012-11-02'][p.IsHoliday==True]
#	df =df.append(df)
#	df = df.append(df)
	
# 	df = df.append(df)
# 	df = df.append(p[p.Date < '2012-11-02'][p.Week_Num.isin(w2)])
# 	df = df.append(p[p.Date < '2012-11-02'][p.Week_Num.isin(w1)])

#	p = p.append(df)


	for v in vars+ ['Days_Since_Paid', 'Days_Since_Paid_gt7']:
		p[v+'_52weeksprior'][p.Week_Num < 52] = pd.Series(p[v][p.Week_Num < 52].values)
	p['Sales_52weeksprior'][p.Week_Num < 52] = pd.Series(p['Weekly_Sales'][p.Week_Num < 52].values)	
	
	return p
		
		

#p, glmf_cv, glmf, bestvars = diff_features(train, prepped_features, 1,1,True)
def diff_features(train, features, store, dept,doplot):
	n = 39
	train_start_date = '2010-02-05' #'2010-11-07' # '2011-01-28' #'2010-07-30' #'2010-09-10'  '2010-12-10' #    #'2010-07-02' #    
	train_start_scoring_date = '2012-07-13'#'2012-05-25' #'2011-12-16' #'2012-07-13'#'2011-12-16' #'2012-07-13'#'2012-02-24'  # '2012-02-24'  #'2011-11-11' # '2012-02-24'  #  '2011-12-16' #'2012-03-02'  # '2012-07-13'# '2012-03-02'  #   
	#train_stop_date = '2011-10-28' #'2012-01-13' # 
	train_stop_date = '2012-07-13'#'2012-05-25' #'2012-01-13' #'2012-03-02' # '2012-07-13'#'2012-10-26' #'2012-05-25' #'2012-07-13'#  '2011-01-28'#'2012-07-27'# '2012-04-27'#'2012-04-27' #
	test_stop_date = '2012-10-26'
	
	
	
	
	vars = ['CPI','Fuel_Price', 'Unemployment','Temperature','MarkDown1','MarkDown2','MarkDown3','MarkDown4','MarkDown5'] #,'Temperature','Fuel_Price'] #,'Fuel_Price', 'Unemployment'] #,'Temperature','Fuel_Price', 'Unemployment'] #
	newvars = [var+'_52weeksprior' for var in vars]
	
	t = train[np.logical_and(train.Store==store, train.Dept==dept)]
	
#	predweeks = list(set(get_week(t[np.logical_and( t['Date'] > train_start_date, t['Date'] <= train_stop_date)].Date.values) ))
#	holpredweeks = list(set(get_week(t[np.logical_and(np.logical_and( t['Date'] >=train_start_date, t['Date'] <= train_stop_date),t.IsHoliday==True)].Date.values) ))
	predweeks = list(set(get_week(t[t['Date'] <= train_stop_date].Date.values) ))
	holpredweeks = list(set(get_week(t[np.logical_and(t['Date'] <= train_stop_date,t.IsHoliday==True)].Date.values) ))

	
	if len(t) < 1:
		#print t.columns
		t = pd.DataFrame(columns=t.columns, index=[0])
		#t.iloc[0] = pd.Series({'Store':store, 'Dept':dept, 'Date':features.Date.iloc[0], 'Weekly_Sales':0, 'IsHoliday':False})
		#t[0] = pd.Series([dict(Store=store, Dept=dept, Date=features.Date.iloc[0], Weekly_Sales=0, IsHoliday=False), ])
		t['Store'] = store
		t['Dept'] = dept
		t['Date'] = features.Date.iloc[0]
		t['Weekly_Sales'] = 0
		t['IsHoliday'] = False
		print t
	
	first_week = get_week([t['Date'].iloc[0]])[0]
	
	p = gen_p(t, features, store, dept, vars, newvars)

	# get rid of data points with missing sales from prev year
	p['Sales_52weeksprior'][p.Sales_52weeksprior < 0] = np.nan
	
	for var in vars+newvars:
		#p = p[np.logical_or(p[var+'_52weeksprior'] > 0, p[var+'_52weeksprior'] < 0)]
		p[var] = p[var].fillna(method='pad')
		p[var] = p[var].fillna(0.0)
		
	p['Sales_52weeksprior']	= p['Sales_52weeksprior'].fillna(0.0)	
		
	#p['Sales_52weeksprior'] = p['Sales_52weeksprior'].fillna(method='backfill')
	
		
	
	#for var in vars+newvars+['Sales_52weeksprior']:
	#	p = p[p[var] >= 0]

	p = p[p.Sales_52weeksprior >= 0]
	#p['Weekly_Sales'] = p['Weekly_Sales'].fillna(method='pad')
	p['Weekly_Sales'] = p['Weekly_Sales'].fillna(0.0)
	
	weeks = get_week(p.Date)
	maxdate= np.max( t[np.logical_and(t.Store==store, t.Dept==dept)].Date )
	#print 'max date ', maxdate
	len_train = len(p[p.Date <= maxdate])
	#print p[p.Date <= maxdate][['Date', 'Weekly_Sales']]
	
	#print len(p.iloc[len_train:])

	if doplot:
		plt.plot(weeks, p.Weekly_Sales.values, 'k.-')
		plt.plot(weeks, p.Sales_52weeksprior, 'r.-')
		raw_input('press a key')
		plt.close()
	
	bestvars =[]
	
# 	df = p[np.logical_and(p.Date <= train_stop_date, p.Week_Num >= 52)]
# 	for v in vars+ ['Days_Before_Val','Days_Before_Val_gt7','Days_Before_Xmas','Days_Before_Xmas_gt7','Days_Since_Paid', 'Days_Since_Paid_gt7', 'Date', 'Week_Num']:
# 		df[v+'_52weeksprior'] = df[v]
# 	df['Sales_52weeksprior'] = df['Weekly_Sales']
# 	
# 	p = p.append(df)
	
	
#	p['Sales_52weeksprior'][p.Date <= '2011-01-28'] = 0.0
	
	

#	if len_train > n+ 10 and len(p[p.Weekly_Sales >= 0][p.Week_Num >= 52].values[0:len_train-n]) > 2:
#	if len_train-52 > n+ 10 and len(p[np.logical_and(np.logical_and(np.logical_and(p.Date >= train_start_date, p.Date < train_stop_date), p.Weekly_Sales >= 0),p.Week_Num >= first_week+52)].values) > 2:
	if len(p[np.logical_and(np.logical_and(p.Date >= train_start_date, p.Date < train_stop_date), p.Weekly_Sales >= 0)].values) > 10 and len(p[np.logical_and(np.logical_and(p.Date > train_start_scoring_date, p.Date <= test_stop_date), p.Weekly_Sales >= 0)].values) > 8:

		
		#families = [sm.families.Gaussian(link=sm.families.links.identity)]
		#f = sm.families.Gamma(link=sm.families.links.inverse_power)
		families = [sm.families.Gaussian(link=sm.families.links.identity), sm.families.Poisson(link=sm.families.links.sqrt) ]
		#families = [sm.families.Gaussian(link=sm.families.links.identity)]
		 
# 		actual = (p[np.logical_and(p.Date > train_stop_date, p.Date <= test_stop_date)].Weekly_Sales.values).astype(np.float)
# 		badpredict = (p[np.logical_and(p.Date > train_stop_date, p.Date <= test_stop_date)].Sales_52weeksprior.values).astype(np.float)
		actual = (p[np.logical_and(p.Date >= train_start_scoring_date, p.Date <= test_stop_date)].Weekly_Sales.values).astype(np.float)
		badpredict = (p[np.logical_and(p.Date >= train_start_scoring_date, p.Date <= test_stop_date)].Sales_52weeksprior.values).astype(np.float)


		
		#shift_fwd_score = np.sum(np.abs(badpredict-actual))
# 		ishol = (p[np.logical_and(p.Date > train_stop_date, p.Date <= test_stop_date)].IsHoliday.values).astype(np.float)
		ishol = (p[np.logical_and(p.Date >= train_start_scoring_date, p.Date <= test_stop_date)].IsHoliday.values).astype(np.float)
		ishol = ishol + np.append(ishol[1:], np.zeros(1))
		shift_fwd_score = 4.0*np.sum(ishol*np.abs(actual-badpredict)) + 1.0*np.sum(np.abs(actual-badpredict))  				

		bestvars = []
		bestscore = shift_fwd_score
		best_badpredict = badpredict
		best_predict = badpredict
		better_predict_score = shift_fwd_score
		glmf_cv = None
		
		#varcombs =[['CPI'],['CPI', 'Unemployment'],['CPI', 'Unemployment', 'Fuel_Price'], ['CPI','Unemployment','MarkDown4','MarkDown5']]
		#varcombs = [['CPI']]
		#varcombs =[['CPI'],['CPI', 'Days_Since_Paid','Days_Since_Paid_gt7']]
		#varcombs = [['CPI'],['CPI','Days_Since_Paid','Days_Since_Paid_gt7']]
		#varcombs = [['CPI'],['Days_Since_Paid','Days_Since_Paid_gt7','Days_Since_Paid_gt14','Days_Until_Paid', 'CPI'],['CPI', 'Days_Before_Xmas', 'Days_Before_Xmas_gt7'],['CPI', 'Days_Before_Xmas', 'Days_Before_Xmas_gt7','Days_Before_Val', 'Days_Before_Val_gt7'],['Days_Since_Paid','Days_Since_Paid_gt7','Days_Since_Paid_gt14','Days_Until_Paid','Days_Before_Xmas', 'Days_Before_Xmas_gt7','CPI'], ['Days_Since_Paid','Days_Since_Paid_gt7','Days_Since_Paid_gt14','Days_Until_Paid','Days_Before_Xmas', 'Days_Before_Xmas_gt7','Days_Before_Val', 'Days_Before_Val_gt7','CPI']]
		#varcombs = [['CPI'],['CPI', 'Unemployment'],['CPI', 'Days_Before_Xmas', 'Days_Before_Xmas_gt7'],['Days_Since_Paid','Days_Since_Paid_gt7', 'CPI'],['Days_Since_Paid','Days_Since_Paid_gt7','Days_Before_Xmas', 'Days_Before_Xmas_gt7','CPI'],['Days_Since_Paid','Days_Since_Paid_gt7','Days_Before_Xmas', 'Days_Before_Xmas_gt7','CPI', 'Temperature'],['Days_Since_Paid','Days_Since_Paid_gt7','Days_Before_Xmas', 'Days_Before_Xmas_gt7','CPI', 'Temperature','MarkDown4', 'MarkDown5' ],['Days_Since_Paid','Days_Since_Paid_gt7','Days_Before_Xmas', 'Days_Before_Xmas_gt7','CPI', 'Unemployment', 'Fuel_Price','Temperature','MarkDown4', 'MarkDown5'],['Days_Since_Paid','Days_Since_Paid_gt7','Days_Before_Xmas', 'Days_Before_Xmas_gt7','CPI', 'Unemployment', 'Fuel_Price','Temperature','MarkDown1','MarkDown2', 'MarkDown3','MarkDown4', 'MarkDown5']]
		##varcombs = [['CPI'],['CPI', 'Days_Before_Xmas', 'Days_Before_Xmas_gt7'],['Days_Since_Paid','Days_Since_Paid_gt7','Days_Since_Paid_gt14', 'CPI'],['Days_Since_Paid','Days_Since_Paid_gt7','Days_Since_Paid_gt14','Days_Before_Xmas', 'Days_Before_Xmas_gt7','CPI'],['Days_Since_Paid','Days_Since_Paid_gt7','Days_Since_Paid_gt14','Days_Before_Xmas', 'Days_Before_Xmas_gt7','CPI', 'Temperature'],['Days_Since_Paid','Days_Since_Paid_gt7','Days_Since_Paid_gt14','Days_Before_Xmas', 'Days_Before_Xmas_gt7','CPI', 'Temperature','MarkDown4', 'MarkDown5' ],['Days_Since_Paid','Days_Since_Paid_gt7','Days_Since_Paid_gt14','Days_Before_Xmas', 'Days_Before_Xmas_gt7','CPI', 'Unemployment', 'Fuel_Price','Temperature','MarkDown4', 'MarkDown5'],['Days_Since_Paid','Days_Since_Paid_gt7','Days_Since_Paid_gt14','Days_Before_Xmas', 'Days_Before_Xmas_gt7','CPI', 'Unemployment', 'Fuel_Price','Temperature','MarkDown1','MarkDown2', 'MarkDown3','MarkDown4', 'MarkDown5']]
		#varcombs = [[],['Days_Since_Paid','Days_Since_Paid_gt7'], ['Days_Before_Xmas', 'Days_Before_Xmas_gt7','Days_After_Xmas_gt7'], ['Days_Since_Paid','Days_Since_Paid_gt7','Days_Before_Xmas', 'Days_Before_Xmas_gt7','Days_After_Xmas_gt7']]
		varcombs = [[], ['Days_Before_Xmas', 'Days_Before_Xmas_gt7'], ['Days_Before_Xmas', 'Days_Before_Xmas_gt7','Days_Before_Val', 'Days_After_Val_gt7']]
		varcombs = varcombs + [comb+['Days_Since_Paid','Days_Since_Paid_gt7', 'Days_Until_Paid', 'Days_Until_Paid_gt7'] for comb in varcombs]
		#varcombs = varcombs + [comb+['Days_Before_Val', 'Days_After_Val_gt7'] for comb in varcombs] #,'Days_Before_Hall', 'Days_Before_Hall_gt7'
		#varcombs = varcombs + [comb+['CPI'] for comb in varcombs]
		#varcombs = [['CPI'], ['CPI', 'Unemployment'],['CPI', 'Fuel_Price'],['CPI', 'Unemployment', 'Fuel_Price']]
		#varcombs = varcombs +[comb+['CPI'] for comb in varcombs] +[comb+['CPI', 'Unemployment'] for comb in varcombs] +[comb+['CPI', 'Fuel_Price'] for comb in varcombs] + [comb+['CPI', 'Fuel_Price', 'Unemployment'] for comb in varcombs]
		varcombs = varcombs[1:] + [comb+['CPI'] for comb in varcombs]
		
		#varcombs = [comb+['CPI'] for comb in varcombs] +[comb+['CPI', 'Unemployment'] for comb in varcombs] #+[comb+['CPI', 'Fuel_Price'] for comb in varcombs]  #+[comb+['CPI', 'Fuel_Price', 'Unemployment'] for comb in varcombs] 
		#varcombs = varcombs +[comb+['Temperature'] for comb in varcombs]
		
		#varcombs = varcombs[1:] +[comb+['MarkDown1'] for comb in varcombs]+ [comb+['MarkDown2'] for comb in varcombs]+ [comb+['MarkDown3'] for comb in varcombs]+[comb+['MarkDown4'] for comb in varcombs]+ [comb+['MarkDown5'] for comb in varcombs] 
		#varcombs = varcombs[1:] +[comb+['MarkDown1'] for comb in varcombs]+ [comb+['MarkDown2'] for comb in varcombs]+ [comb+['MarkDown3'] for comb in varcombs]+[comb+['MarkDown4'] for comb in varcombs]+ [comb+['MarkDown5'] for comb in varcombs]  +[comb+['MarkDown1', 'MarkDown2'] for comb in varcombs] + [comb+['MarkDown1', 'MarkDown3'] for comb in varcombs] +[comb+['MarkDown1', 'MarkDown4'] for comb in varcombs]+ [comb+['MarkDown1', 'MarkDown5'] for comb in varcombs] +[comb+['MarkDown2', 'MarkDown5'] for comb in varcombs] +[comb+['MarkDown3', 'MarkDown5'] for comb in varcombs]+[comb+['MarkDown4', 'MarkDown5'] for comb in varcombs] + [comb+['MarkDown3','MarkDown4', 'MarkDown5'] for comb in varcombs] + [comb+['MarkDown1','MarkDown2','MarkDown3','MarkDown4', 'MarkDown5'] for comb in varcombs]
		#varcombs = [['CPI'],['CPI', 'Unemployment'],['CPI', 'Unemployment','Fuel_Price'], ['CPI', 'MarkDown1','MarkDown2', 'MarkDown3','MarkDown4', 'MarkDown5'], ['CPI','Unemployment', 'MarkDown3','MarkDown4', 'MarkDown5'], ['CPI','Unemployment', 'MarkDown1','MarkDown2', 'MarkDown3','MarkDown4', 'MarkDown5']]

		varcombs = [['CPI',], ['Days_Since_Paid','Days_Since_Paid_gt7', 'Days_Until_Paid', 'Days_Until_Paid_gt7'],['Days_Before_Val', 'Days_After_Val_gt7'], ['MarkDown1'],['MarkDown2'], ['MarkDown3'],['MarkDown4'], ['MarkDown5'], ['Temperature'], ['Fuel_Price'], ['Unemployment']]

		for j in range(len(varcombs)):
			vars = bestvars+varcombs[j]
			newvars = [var+'_52weeksprior' for var in vars]

 			x = (p[['Sales_52weeksprior']+vars+newvars][np.logical_and(np.logical_and(p.Date >= train_start_date, p.Date <= train_stop_date), p.Weekly_Sales >= 0)].values).astype(np.float)
 			y = (p['Weekly_Sales'][np.logical_and(np.logical_and(p.Date >= train_start_date, p.Date <= train_stop_date), p.Weekly_Sales >= 0)].values).astype(np.float)

			
			
#			x_test = (p[np.logical_and(p.Date > train_stop_date, p.Date <= test_stop_date)][['Sales_52weeksprior']+vars+newvars].values).astype(np.float)
			x_test = (p[np.logical_and(p.Date >= train_start_scoring_date, p.Date <= test_stop_date)][['Sales_52weeksprior']+vars+newvars].values).astype(np.float)

			x_test2 = (p[['Sales_52weeksprior']+vars+newvars]).astype(np.float)	
			y_test2 = (p['Sales_52weeksprior']).astype(np.float)
			
			x_test3 = (p[['Sales_52weeksprior']+vars+newvars][p.Date > test_stop_date]).astype(np.float)	
			y_test3 = (p['Sales_52weeksprior'][p.Date > test_stop_date]).astype(np.float)			

			for i in range(len(families)):
				glm = sm.GLM(y, x, families[i])
				try:
					glmf = glm.fit()	
				except:
 					#predict = (p[np.logical_and(p.Date > train_stop_date, p.Date <= test_stop_date)].Sales_52weeksprior.values).astype(np.float)
					glmf = None
					predict = (p[np.logical_and(p.Date >= train_start_scoring_date, p.Date <= test_stop_date)].Sales_52weeksprior.values).astype(np.float)
					p2 = (p.Sales_52weeksprior.values).astype(np.float)
# 					m = np.median(predict)
# 					predict_min = np.min(pred)
# 					predict = pred*(pred <= m+5.0*(m-predict_min)) + (p[np.logical_and(p.Date > train_stop_date, p.Date <= test_stop_date)]['Sales_52weeksprior'].values).astype(np.float)*(pred > m+5.0*(m-predict_min))
				else:
					predict= glmf.predict(x_test)
					p2= glmf.predict(x_test2)
					
				
				#print len(predict)
				#better_predict_score_t =  np.sum(np.abs(predict-actual))
 				#ishol = (p[np.logical_and(p.Date > train_stop_date, p.Date <= test_stop_date)].IsHoliday.values).astype(np.float)
				ishol = (p[np.logical_and(p.Date >= train_start_scoring_date, p.Date <= test_stop_date)].IsHoliday.values).astype(np.float)
				ishol = ishol + np.append(ishol[1:], np.zeros(1))	

				better_predict_score_t = 4.0*np.sum(ishol*np.abs(actual-predict)) + 1.0*np.sum(np.abs(actual-predict))  				
				

				if doplot:
 					print 'shift fwd predict score ', shift_fwd_score
 					print 'better predict score ', better_predict_score_t	
					
				if (i == 0 and j==0) or (np.min(predict) >= 0 and np.max(p2) < 1.5*np.max(y) and ( (better_predict_score_t < bestscore and len(vars) <= len(bestvars) ) or  (better_predict_score_t < bestscore-8.0*np.sqrt(bestscore) and len(vars) > len(bestvars) )) ):
									
					x2 = (p[['Sales_52weeksprior']+vars+newvars][np.logical_and(np.logical_and(p.Date >= train_start_date, p.Date <= test_stop_date), p.Weekly_Sales >= 0)].values).astype(np.float)
					y2 = (p['Weekly_Sales'][np.logical_and(np.logical_and(p.Date >= train_start_date, p.Date <= test_stop_date), p.Weekly_Sales >= 0)].values).astype(np.float)
					
					x3 = (p[['Sales_52weeksprior']+vars+newvars][np.logical_and(np.logical_and(p.Date >= train_start_scoring_date, p.Date <= test_stop_date), p.Weekly_Sales >= 0)].values).astype(np.float)
					y3 = (p['Weekly_Sales'][np.logical_and(np.logical_and(p.Date >= train_start_scoring_date, p.Date <= test_stop_date), p.Weekly_Sales >= 0)].values).astype(np.float)


# 					x2 = (p[['Sales_52weeksprior']+vars+newvars][p.Weekly_Sales >= 0][p.Week_Num >= 52][p.Date < '2012-11-02'].values).astype(np.float)
# 					y2 = (p['Weekly_Sales'][p.Weekly_Sales >= 0][p.Week_Num >= 52][p.Date < '2012-11-02'].values).astype(np.float)

# 				 	x2 = (p[['Sales_52weeksprior']+vars+newvars][np.logical_and(p.Date <= test_stop_date, p.Weekly_Sales >= 0)].values).astype(np.float)
#  					y2 = (p['Weekly_Sales'][np.logical_and( p.Date <= test_stop_date, p.Weekly_Sales >= 0)].values).astype(np.float)



					
					glm_t = sm.GLM(y2, x2, families[i])
					try:
						glmf_t = glm_t.fit()
					except:
						pred_test = -1.0*np.ones(len(y2))
						pred3 = -1.0*np.ones(len(y3))
					else:
						pred_test = glmf_t.predict(x_test2)
						pred3 = glmf_t.predict(x3)
						p3 = glmf_t.predict(x_test3)
					
					ishol = (p[np.logical_and(p.Date >= train_start_scoring_date, p.Date <= test_stop_date)].IsHoliday.values).astype(np.float)
					better_predict_score_t3 = 4.0*np.sum(ishol*np.abs(y3-pred3)) + 1.0*np.sum(np.abs(y3-pred3))  				
					
					#print better_predict_score_t, better_predict_score_t3
					#print np.max(pred_test), np.max(pred_test), 2.0*np.max(y_test2), 1.0*np.sum(np.abs(y_test2-pred_test))/len(y_test2) , 1.5*np.sum(np.abs(badpredict-actual))/len(actual)
					if better_predict_score_t > better_predict_score_t3 and np.min(pred_test) > 0 and np.max(pred_test) < np.mean(y_test2)+1.5*(np.max(y_test2) -np.mean(y_test2)) and np.min(pred_test) > 0.9*np.min(y_test2)  and 1.0*np.mean(np.abs(y_test2-pred_test)) < 1.3*np.mean(np.abs(badpredict-actual)) and 1.0*np.mean(np.abs(y_test3-p3)) < 1.3*np.mean(np.abs(badpredict-actual)): #np.max(pred_test) > 10.0*p.Weekly_Sales.max() or 
						f = families[i]
						better_predict_score = better_predict_score_t
						bestscore = np.min((bestscore, better_predict_score))
						bestvars = vars
						best_predict = predict
						glmf_cv = glmf
						
			
			bad_predict = best_badpredict
			predict = best_predict
	
	else:
		actual = (p.Weekly_Sales.values).astype(np.float)
		predict = (p.Sales_52weeksprior.values).astype(np.float)
		badpredict = predict
		glmf = None
		glmf_cv = None
		shift_fwd_score = np.sum(np.abs(badpredict-actual))
		better_predict_score =  np.sum(np.abs(predict-actual))


	
	if doplot:
		plt.plot(actual, predict, 'ko')
		raw_input('press a key')
		plt.close()	
		
		plt.plot(range(len(badpredict)), actual,'k.-')
		plt.plot(range(len(badpredict)), badpredict,'r.-')
		plt.plot(range(len(badpredict)), predict,'g.-')
		
		raw_input('press a key')
		plt.close()	
		
		print 'shift fwd predict score ', shift_fwd_score
		print 'better predict score ', better_predict_score	


	if glmf_cv==None or not better_predict_score < 0.99*shift_fwd_score :
		glmf = None
		glmf_cv = None
	else:
		print bestscore, bestvars
		vars = bestvars
		newvars = [var+'_52weeksprior' for var in vars]
		
# 		x = (p[['Sales_52weeksprior']+vars+newvars][p.Weekly_Sales >= 0][p.Week_Num >= 52].values[0:len_train]).astype(np.float)
# 		y = (p['Weekly_Sales'][p.Weekly_Sales >= 0][p.Week_Num >= 52].values[0:len_train]).astype(np.float)
#  		x = (p[['Sales_52weeksprior']+vars+newvars][np.logical_and(np.logical_and(p.Week_Num >= 52, p.Date < '2012-11-02'), p.Weekly_Sales >= 0)].values).astype(np.float)
#  		y = (p['Weekly_Sales'][np.logical_and(np.logical_and(p.Week_Num >= 52, p.Date < '2012-11-02'), p.Weekly_Sales >= 0)].values).astype(np.float)
# 		x = (p[['Sales_52weeksprior']+vars+newvars][np.logical_and( p.Date < '2012-11-02', p.Weekly_Sales >= 0)].values).astype(np.float)
# 		y = (p['Weekly_Sales'][np.logical_and( p.Date < '2012-11-02', p.Weekly_Sales >= 0)].values).astype(np.float)
#		x = (p[['Sales_52weeksprior']+vars+newvars][np.logical_and(p.Date <= train_stop_date, p.Weekly_Sales >= 0)].values).astype(np.float)
#		y = (p['Weekly_Sales'][np.logical_and( p.Date <= train_stop_date, p.Weekly_Sales >= 0)].values).astype(np.float)
		x = (p[['Sales_52weeksprior']+vars+newvars][np.logical_and(np.logical_and(p.Date <= test_stop_date, p.Date >= train_start_date), p.Weekly_Sales >= 0)].values).astype(np.float)
		y = (p['Weekly_Sales'][np.logical_and(np.logical_and(p.Date <= test_stop_date,p.Date >= train_start_date), p.Weekly_Sales >= 0)].values).astype(np.float)


		
		if len(y) < 2:
			glmf= None
			glmf_cv = None
		else:	
			glm = sm.GLM(y, x, f)
			glmf = glm.fit()

	if doplot:
		plt.plot(p.Week_Num, p.Weekly_Sales, 'k.-')
		plt.plot(p.Week_Num, p.Sales_52weeksprior, 'r.-')
		#x_test = (p[['Sales_52weeksprior']+vars+newvars][p.Date >= '2012-11-02']).astype(np.float)
		x_test = (p[['Sales_52weeksprior']+vars+newvars]).astype(np.float)
		if glmf == None:
			#predict = (p['Sales_52weeksprior'][p.Date >= '2012-11-02']).astype(np.float)
			predict = (p['Sales_52weeksprior']).astype(np.float)
		else:
			predict= glmf.predict(x_test)
# 			m = np.median(pred)
# 			predict_min = np.min(pred)
# 			predict = pred*(pred <= m+5.0*(m-predict_min)) + (p['Sales_52weeksprior'][p.Date >= '2012-11-02']).astype(np.float)*(pred > m+5.0*(m-predict_min))
			
		#plt.plot(p[p.Date >= '2012-11-02'].Week_Num, predict, 'g.-')
		plt.plot(p.Week_Num, predict, 'g.-')
		
		if glmf != None:
			plt.plot(p.Week_Num, glmf_cv.predict(x_test), 'b.-')
		raw_input('press a key')
		plt.close()		
		
		
	return p, glmf_cv, glmf, bestvars
	

	




#pred = make_prediction3(range(0,24), train, features, 1, 51)	
def make_prediction3(predweeks,train,features,store, dept, mode):
	p, glmf_cv, glmf, bestvars = diff_features(train, features, store, dept,False)
		
	if mode == 'cv':
		glmf = glmf_cv
			
	vars = bestvars
	newvars = [var+'_52weeksprior' for var in vars]
	
	#print predweeks	
	p = p[['Sales_52weeksprior', 'Week_Num', 'Date']+vars+newvars].groupby('Date').first()
	
	x = (p[['Sales_52weeksprior']+vars+newvars][p.Week_Num.isin(predweeks)].values).astype(np.float)
	
		
	if glmf != None:
		pred = glmf.predict(x)
#		y = (p['Sales_52weeksprior'][p.Week_Num.isin(predweeks)].values).astype(np.float)
# 		if np.max(pred) > 2.0*np.max(y):
# 			pred = y
		#else:	
		#m = np.median(pred)
		#predict_min = np.min(pred)
		#print m, predict_min, m+3.0*(m-predict_min)
		#pred = pred*(pred <= m+5.0*(m-predict_min)) + y*(pred > m+5.0*(m-predict_min))
		#pred = pred*(pred <= m+3.0*(m-predict_min)) + np.max((y, pred), axis=0)*(pred > m+3.0*(m-predict_min))

	else:
		pred =  (p['Sales_52weeksprior'][p.Week_Num.isin(predweeks)].values).astype(np.float)
	

	return pred
	


	
	
#bad = cv(train, prepped_features)
def cv(train, features):
	n = 52
	scoreall = 0.0
	bad = pd.DataFrame(columns=('Store', 'Dept', 'Score'))
	i = 0

#	partialcv = [[1,1]]
#	partialcv = [[1,15],[15,37],[17,99],[19,39],[21,99],[25,96],[29,99]]	
#	partialcv = [[41,99],[6,45],[1,99],[1,51],[1,1],[11,45],[1,18], [1,3],[1,92], [36,94],[42,38]]
   	for s in range(1,46):#range(1,46):
   		for d in range(1,100):#range(1,100):
# 	if True:
# 		for [s,d] in partialcv:
			#print s,d, len(train)
			t = train[np.logical_and(train.Store == s, train.Dept == d)]
			
			if len(t) > 0:
				i += 1
				#start = get_week([t.Date.values[-1]])[0]-52
				#end = start + n +1
# 				start = get_week([t.Date.values[-1]])[0]-n
# 				end = get_week([t.Date.values[-1]])[0]
				
				y,m, day = t.Date.values[-1].split('-')
				y,m,day = np.int(y), np.int(m), np.int(day)
				
# 				startdate =  str( datetime.date(y,m,day) + datetime.timedelta(days=-7*52) )
# 				enddate =  str( datetime.date(y,m,day) + datetime.timedelta(days=-7*(52-n)) )
				startdate =  str( datetime.date(y,m,day) + datetime.timedelta(days=-7*n) )
				enddate =  str( datetime.date(y,m,day) + datetime.timedelta(days=-7*(0)) )


# 				y,m, day = t.Date.values[0].split('-')
# 				y,m,day = np.int(y), np.int(m), np.int(day)				
# 				startpreddate =  str( datetime.date(y,m,day) + datetime.timedelta(days=7*52) )

#				predweeks = get_week(t[np.logical_and( t['Date'] >=startdate, t['Date'] <= enddate)][t['Date'] >startpreddate].Date.values)
				predweeks = list(set(get_week(t[np.logical_and( t['Date'] >=startdate, t['Date'] <= enddate)].Date.values) ))
				holpredweeks = list(set(get_week(t[np.logical_and(np.logical_and( t['Date'] >=startdate, t['Date'] <= enddate),t.IsHoliday==True)].Date.values) ))
				#print holpredweeks
				
				#print predweeks
				if len(predweeks) == 0:
					score = 0
				else:			
					pred = make_prediction3(predweeks, t, features,s,d, 'cv')
					#pred = make_prediction3(predweeks, t, features,s,d, 'pred')
					#actual = t[np.logical_and( t['Date'] >=startdate, t['Date'] <= enddate)][t['Date'] >startpreddate].Weekly_Sales.values
					actual = t[np.logical_and( t['Date'] >=startdate, t['Date'] <= enddate)].Weekly_Sales.values
					
					if len(actual) == len(pred):
						#score = np.mean(np.abs(actual-pred))
						holidx = [q for q in range(len(predweeks)) if predweeks[q] in holpredweeks]
						if len(holidx) > 0:
							score = 1.0*(4.0*np.sum(np.abs(actual[holidx]-pred[holidx])) + 1.0*np.sum(np.abs(actual-pred)) )/(4*len(holidx)+len(pred))
						else:
							score = np.mean(np.abs(actual-pred))
					else:
						score = 0
					
				print s,d,score
				scoreall += score

				row = {'Store': s, 'Dept': d, 'Score': score}
				#row = pd.DataFrame([dict(Store=s, Dept=d, Score=score), ])
				bad = bad.append(row,ignore_index=True)
		
	print 'final score ', scoreall/i
	return bad
		
		


	
#parsesubmission(submission)	
def parsesubmission(submission):
	submission['Store'] = 0
	submission['Dept'] = 0
	submission['Date'] = ''
	
	for i in range(len(submission)):
		s,d,date = submission.Id.iloc[i].split('_')
		#print s, d, date
		submission['Store'].iloc[i] = s
		submission['Dept'].iloc[i] = d
		submission['Date'].iloc[i] = date
	
	
	
	
#makesubmission(train,prepped_features, submission)
#makesubmission(train,features, submission.iloc[0:80])
#submission should be parsed
def makesubmission(train, features, parsed_submission):

	#for i in range(0,parsed_submission.shape[0],39):
# 	for s in [4]:
# 		for d in [99]:	
	for s in range(1,46):
		for d in range(1,100):
			t = parsed_submission[np.logical_and(parsed_submission.Store == s, parsed_submission.Dept == d)]
			
			if len(t) > 0:
				print s,d
				weeks = get_week(t.Date.values)
		
				#pred, time = make_prediction(weeks, train[np.logical_and(train.Store == s, train.Dept == d)])
				pred = make_prediction3(weeks, train[np.logical_and(train.Store == s, train.Dept == d)], features,s,d, 'pred')
				#print len(pred), len(weeks)
				
				if len(pred) != len(parsed_submission.Weekly_Sales[np.logical_and(parsed_submission.Store == s, parsed_submission.Dept == d)]):
					print 'uh oh'
				parsed_submission.Weekly_Sales[np.logical_and(parsed_submission.Store == s, parsed_submission.Dept == d)]= pred.astype(np.int)

		
	parsed_submission[['Id','Weekly_Sales']].to_csv('submission30.csv',index=False)
	print 'submission file created'	
		
		