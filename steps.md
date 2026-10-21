1) overview of dataset
2) column names
3) missing values
  removed 43 columns with more than 60% missing values

4) groupping data types
5) target column cleaning
 6 unique strings which included current and grace which were irrelevant
 -> standarize the data to contain only cahrged or not (binary)
 -> there is sever class imbalance

 6) checked duplicates
 -> no duplicates

 7) filter some redudant columns from domain knowledge and find corr to reduce columns
  id/idenitfier
  #text field
  funded



   we had date time but uniques are also num and cat. no datetime so we have to clean the variables

  <!-- 8) also saved df to sql for better analysis  -->

 8) cleaning emp_time, time data types  
    # term and emp_length stored as text
 9) also found const col so removed them
 10) some values were near constant removed them too

 11) looked more redudant values (overlapping features)
 
 installment: removed because it is largely determined by loan_amnt, term, and int_rate. Keeping all four can add redundant information.
purpose: removed for simplification because it is categorical and can create several encoded features; it may contain signal, but it was not essential for the first-pass model.
initial_list_status: removed because it is more of a LendingClub operational/listing attribute than a core borrower-risk characteristic.

deliqncyu variable: overlap
inuqiery: same



## EDA
