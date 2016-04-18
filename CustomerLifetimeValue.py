import graphlab as gl
import datetime
import itertools

# In marketing, customer lifetime value (CLV), lifetime customer value (LCV),
# or life-time value (LTV) is a prediction of the net profit attributed to the 
# entire future relationship with a customer.

def calculate(sf, date_col, id_col, profit_col, churn_period=datetime.timedelta(days = 30)):
    """Calculate the profit and lifetime (in periods) for each customer, returning tupple(churned, active)
    
    A customer is considered 'churned' if they did not make a transaction in the last period of the data. 
    To avoid mistakes, the sum of profit_col is called 'CLV' for 'churned' customers and 'Total Profit' for 'active' customers.
    """

    data = sf.groupby(id_col, {'Profit':gl.aggregate.SUM(profit_col),
                               'Start':  gl.aggregate.MIN(date_col), 'End':  gl.aggregate.MAX(date_col)})
    data['Lifetime'] = data.apply(lambda row: (row['End'] - row['Start']).days/churn_period.days + 1)
    
    # Split customers into churned vs active based on last transaction date.
    # Replace with a builtfunction if it exists 
    boundary = sf[date_col].max() - churn_period  
    mask = data['End'] < boundary
    inverse_mask = mask.apply(lambda x: 1 if x == 0 else 0)
	
    data = data.remove_columns(['End', 'Start'])	       # These dates are not useful in CLV

    churned = data[mask].rename({'Profit': 'CLV'}) # CLV applied to churned customers ONLY
    active =  data[inverse_mask]                         # so leave it as Total Profit for active users
	
    return (churned, active)
  
def period(sf, date_col, id_col, churn_period=datetime.timedelta(days = 30)):
    '''Return an SArray of the membership period for each transaction'''
    
    start_dates = sf.groupby(id_col, {'Start':gl.aggregate.MIN(date_col)})
     
    data = sf.join(start_dates, on='CustomerID')
    period = data.apply(lambda row: (row[date_col] - row['Start']).days/churn_period.days + 1)
    
    return period


# The engineered features will be specific to the data and model but is a generic start that can be build on
def features(sf, date_col, id_col, profit_col, churn_period=datetime.timedelta(days = 30)):
    '''Engineer new features for CLV model from raw transaction data, returning one observation per customer'''

    data = sf
    data['Period'] = period(sf, date_col, id_col, churn_period)
    
    data = data.groupby([id_col, 'Period'], {'Txs' :      gl.aggregate.COUNT(),
                                             'Profit':    gl.aggregate.SUM(profit_col),
                                             'Quantity':  gl.aggregate.SUM('Quantity'),
                                             'Purchases': gl.aggregate.CONCAT('StockCode'),
											 'End':	      gl.aggregate.MAX(date_col)
                                            })

    data = data.groupby('CustomerID', {'Lifetime':  gl.aggregate.MAX('Period'),
                                       'Txs':       gl.aggregate.SUM('Txs'),
                                       'Profit':    gl.aggregate.SUM('Profit'),
                                       'Quantity':  gl.aggregate.SUM('Quantity'),
                                       'Purchases': gl.aggregate.CONCAT('Purchases'),
                                       'Txs_by_period':       gl.aggregate.CONCAT('Period', 'Txs'),
                                       'Profit_by_period':    gl.aggregate.CONCAT('Period', 'Profit'),
                                       'Quantity_by_period':  gl.aggregate.CONCAT('Period', 'Quantity'),
                                       #'Purchases_by_period': gl.aggregate.CONCAT('Period', 'Purchases'), #Not usable with regression APIs
									   'End':	      gl.aggregate.MAX('End')
                                            })
    # Flatten out the list of list of purchases and ensure they are unique
    data['Purchases'] = data['Purchases'].apply(lambda lst: list(set(itertools.chain(*lst))))
    
     # Split customers into churned vs active based on last transaction date.
    # Replace with a builtfunction if it exists 
    boundary = sf[date_col].max() - churn_period  
    mask = data['End'] < boundary
    inverse_mask = mask.apply(lambda x: 1 if x == 0 else 0)
	
    data = data.remove_columns(['End'])

    churned = data[mask].rename({'Profit': 'CLV'}) # CLV applied to churned customers ONLY
    active =  data[inverse_mask]                         # so leave it as Total Profit for active users
	
    return (churned, active)