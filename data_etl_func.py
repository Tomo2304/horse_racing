import pandas as pd
import numpy as np 
from datetime import datetime, timedelta
from sklearn.preprocessing import LabelEncoder
from os import path

############  Create functions ############

#### reading data, cleaning, formatting, and remove columns and records 
def data_to_read(history_df,mapping_df):
    tbl_mapping = pd.read_excel(mapping_df,sheet_name='hist_tbl')
    var_name = tbl_mapping['col_name'].tolist()
    
    df_data = pd.read_csv('race_history.csv', header=0, names=var_name,index_col=False)
    df_data['year_yyyy'] = '20' + df_data['year'].astype(str)   
    df_data['month_mm'] = df_data['month'].apply(lambda x : '0' + str(x) if int(x) < 10 else str(x))
    df_data['day_dd'] = df_data['day'].apply(lambda x : '0' + str(x) if int(x) < 10 else str(x))
    df_data['horse_name_original'] = df_data['horse_name']
    ## same horse name can exist in data. Create unique horse name
    df_data['horse_name'] = df_data['horse_name'] + df_data['producer_name'] + df_data['horse_father'] + df_data['horse_mother']
    return df_data

df = data_to_read('race_history.csv','mapping_tbl.xlsx')

#### remove unnecessary data and variables
def data_to_delete(df):
    ## races that were cancelled, only for new horses, some horses with handicapped, and obstacle races are excluded 
    ## They are have a lot less number of race histories.
    df = df[~df['track_cd'].isin([51,52,53,54,55,56,57,58,59])]
    df['delete_flg'] = (df['race_name'].str.contains('１勝ｸﾗｽ|未勝利|新馬|H|障害')).astype(int)
    df = df[df['異常コード'].isnull() | (df['異常コード'] == 0) & (df['delete_flg'] == 0)]
    ## some duplicated variables and variables that have no associations with the race results are dropped
    drop_vars = ['タイムS','レースID','単勝オッズ','人気','異常コード', 'delete_flg',\
                 'フルゲート頭数','1着本賞金','発送時刻','基準タイム秒','レースコメント','枠版','ブリンカー','入線着順','馬主名','日付S']
    df['time_diff'] = df['time_diff'].astype(float)
    df['total_horses'] = df['total_horses'].astype(float)
    df = df.drop(columns=drop_vars)
    return df

df = data_to_delete(df)

#### create track type feature
def track_type(df):
    def tr_type(df):
        if df['track_cd'] >= 10 and df['track_cd'] <= 22:
            return('grass')
        elif df['track_cd'] in [23,24,25,26,29]:
            return('dirt')
        elif df['track_cd'] in [27,28]:
            return('sand')
        else:
            return('Unknown')
    df['track_type'] = df.apply(tr_type,axis=1)
    return(df)

df = track_type(df)

#### label encoding categorical string variables
def label_encode(df):
    # get all object column name in list
    cat = list(df.select_dtypes(include='object').columns)
    # remove object but non categorical variable
    non_cat = {'year_yyyy','month_mm','day_dd','horse_name'}
    cat = [ele for ele in cat if ele not in non_cat]
    # replace null value to abnormal value to be recognised as null
    for i in cat:
        df[i] = df[i].fillna('Nan')
    # label encoding all the variables
    le = LabelEncoder()
    df[cat] = df[cat].apply(le.fit_transform)

    print("label encoded:",cat)
    return df, cat

df_l, cat = label_encode(df)

#### Peak age of horse for racing is around 4 years old. 
#### create age in months as this feature may have more impact on the prediction
def age_month(df):
    df['DOB'] = df['DOB'].astype(str)
    df['age_month'] = (df['year_yyyy'].astype('float64') - df['DOB'].str[:4].astype('float64')) * 12 + (df['month'].astype('float64') - df['DOB'].str[4:6].astype('float64'))
    df.drop(columns=['DOB'])
    return(df)

df = age_month(df_l)


#### WORK FROM HERE
# enter label encoding category list and add other cat variables
def int_cat_list(cat):
    other_cat = ['place','class_cd','race_name','field', 'weather',
        'field_cond', 'horse_name', 'sex','jockey','rank', 'track_cd',
        'leg_typ','producer_name','horse_father','horse_mother','month']
    # merge list with no duplicates
    int_cat_list = list(set(cat + other_cat))
    # delete vars not required
    drop_cat = {'sex','horse_name','producer_name','horse_father','horse_mother'}
    int_cat_list = [ele for ele in int_cat_list if ele not in drop_cat]
    return(int_cat_list)

#### list of columns to keep for previous race variables
def rename_histry_data(var_hist,df_name):
    #NK horse_name deleted, jockey added, race_name added
    df = df_name[['horse_name','place','sum_num','horse_num','rank','field','track_type',\
                 'dist','field_cond', 'weather','time','last_3F','date','class_cd','jockey',\
                  'corner_cnt', 'Ave_3F', 'track_cd','time_diff', 'weight_change','month','race_name',\
                  'RPCI','PCI3','leg_typ','PCI','last_3F_diff','weight_carry', 'horse_weight','age_month']]
    for col in df.columns.tolist():
        df = df.rename(columns={col : 'p' + '_' + col + '_' + var_hist})
    return df

#### create x number of historical race features 
def merge_histry_data(df_name,cat):
    df_all = pd.DataFrame([])
    # clean and prep date variable. The variable is used to rank the date for each horse 
    # rank is used to find most x number of recent races for each horse
    df_name['date'] = (df_name['year'] + df_name['month'].astype(str) + df_name['day']).astype(int) 
    df_name["order"]=df_name.groupby(["horse_name"])["date"].rank(ascending=False)
    df_name['date'] = pd.to_datetime((df_name['date']), format='%Y%m%d').dt.date
    df_name['month'] = df_name['month'].astype('int64')
    # find horse with highest number of races. deduct it by 2 as it looks at last 3 races
    l_count = df_name.horse_name.value_counts().max()-2
    for t in range(1,l_count):
        print(str(t) + " out of " + str(l_count))
        df_to_add = df_name[df_name['order'] == t]
        for i in [str(i) for i in range(0, 10)]:
            df_hist = df_name[df_name['order'] == int(i) + 1 + t]
            df_to_add = pd.merge(df_to_add, rename_histry_data(i,df_hist), how='left', \
                      left_on='horse_name', right_on='p_horse_name_' + i)

            #calculate l_days
            if i == '0':
                df_to_add['l_days_0'] = (pd.to_datetime(df_to_add['date']) - pd.to_datetime(df_to_add['p_date_0'])).dt.days
            else:
                df_to_add['l_days_' + str(int(i))] = (pd.to_datetime(df_to_add['p_date_' + str(int(i) - 1)]) - pd.to_datetime(df_to_add['p_date_' + str(int(i))])).dt.days

            ### comment out 
#            df_to_add = df_to_add[df_to_add['p_horse_name_' + i].notnull()].drop(columns='p_horse_name_' + i)
            df_to_add = df_to_add.drop(columns='p_horse_name_' + i)
            for var in cat:
                df_to_add['p_' + var + '_' + i] = df_to_add['p_' + var + '_' + i].astype('Int64')
        
        if df_all is None:
            df_all = df_to_add.copy()
        else:
            df_all = pd.concat([df_all, df_to_add])
    return df_all


#### count number of times horse and jockey won in each rank 
# rolling sum function 
def rolling_sum(df, r, last_d, count_col, primary_key, new_col_name):
    rolling = pd.DataFrame(df.groupby(primary_key).rolling(last_d,on='date')[count_col].sum().reset_index())
    new_col = new_col_name + str(r) + '_' + 'last' + last_d
    rolling = rolling.rename(columns = {count_col : new_col})
    # deduct the rolling sum by the result of this time of race
    rolling = rolling.merge(df, on = [primary_key,"date"], how="left")
    rolling[new_col] = rolling[new_col] - rolling[count_col]
    rolling = rolling[[primary_key,'date',new_col]]
    return(rolling)

# min days, max days, and interval decide number of variables for period e.g. 90 to 365 for every 90 days
# lowest rank decides the range of ranks included for counting e.g. 3 for 1 to 3
def horse_jocky_rank_count(df,min_days=90,max_days=365,days_interval=90,min_rank=3):
    # Prepare data frame and list 
    df['date'] = pd.to_datetime(df['date'])
    df['jockey_field'] = df['jockey'].astype(str) + '_' + df['field'].astype(str)
    df_short = df[['horse_name','jockey_field','date','rank']]
        # convert ranks larger than min_rank parameter to be excluded from get dummies 
    df_short['rank'] = np.where(df_short['rank'] <= min_rank,df_short['rank'],None)
    
    # Prepare horse rank df 
    h_short = df_short[['horse_name','date','rank']]
    h_short = pd.get_dummies(h_short,columns=["rank"])
    h_short = h_short.sort_values(['horse_name','date'], ascending=(True,True)).reset_index()

    # Prepare jockey rank df 
    j_short = df_short[['jockey_field','date','rank']]
    j_short = pd.get_dummies(j_short,columns=["rank"])
        # jockey needs to sum as one jocky can have more than once race and get 1st etc in a day
    rank_col = list(j_short.drop(columns=['jockey_field','date']).columns)
    j_short_sum = j_short.groupby(['jockey_field','date'])[rank_col].sum().reset_index()
    j_short_sum = j_short_sum.sort_values(['jockey_field','date'], ascending=(True,True)).reset_index()
    
    range_hist = list(range(min_days,max_days,days_interval))
    range_rank = list(range(1,min_rank+1,1))
    for r in range_rank:
        # iterate through each rank list
        count_col = "rank_" + str(r)

        for d in range_hist:
            # iterate through each days list and rolling sum on last_d days
            last_d = str(d) + 'D'
            h_roll = rolling_sum(df = h_short, r =r, last_d = last_d, count_col = count_col, primary_key = 'horse_name', new_col_name = 'horse_rank')
            df = df.merge(h_roll, on = ['horse_name','date'], how='left')
            
            j_roll = rolling_sum(df = j_short_sum, r=r, last_d = last_d, count_col = count_col, primary_key = 'jockey_field', new_col_name = 'jockey_field_rank')
            df = df.merge(j_roll, on = ['jockey_field','date'], how='left')
            dd = d
    df = df.drop(columns=['jockey_field'])
    return df,dd

# find days between this and last race in history_merge 
#def days_between_race(df):
#    df['l_days'] = (df['date'] - pd.to_datetime(df['p_date_0'])).dt.days
#    df['l_days_0'] = (pd.to_datetime(df['p_date_0']) - pd.to_datetime(df['p_date_1'])).dt.days
#    df['l_days_1'] = (pd.to_datetime(df['p_date_1']) - pd.to_datetime(df['p_date_2'])).dt.days
#    return df

# Check the test result for the model if model improves features added 
def winning_times_competitiors(df):
    ## variables for target horse winning history 
    print("WIP") 

def create_race_key(df):
    df['race_num'] = df['race_num'].apply(lambda x : '0' + str(x) if int(x) < 10 else str(x))
    df['month_dummy'] = df['month'].apply(lambda x : '0' + str(x) if x < 10 else str(x))
    df['race_key'] = df['year'].astype(str) + df['month_dummy'].astype(str) + df['day'].astype(str) + \
        df['place'].astype(str) + df['race_num'].astype(str)
    df = df.drop(columns='month_dummy')
    return df

def drop_some(df,maxd):
    # history to be removed. last 12 month rolling is used for delete the oldest 12 month records
    cut_off_date = df['date'].min() + timedelta(days=maxd)
    df = df[df['date'] > cut_off_date]

    # columns not needed
    col = ['order','race_num','last_3F_diff','prize','DOB','Ave_3F'
           ,'time_diff','time','last_3F','prize','RPCI','PCI3','leg_typ','PCI','age']
    df = df.drop(columns=col)
    
    drop_cols = df.columns[df.columns.str.contains('date_')]
    df = df.drop(columns=drop_cols)
    return df 

# save pandas data frame with dtype in csv file
def to_csv(df, file_name, file_path):
    # Prepend dtypes to the top of df
    dd = datetime.today().strftime('%Y_%m_%d_%H')
    file = path.join(file_path,file_name + "_" + dd + ".csv")
    ## datetime needs to be converted to string to read csv later
    df['date'] = df['date'].dt.strftime('%Y_%m_%d')
    df.loc[-1] = df.dtypes
    df.index = df.index + 1
    df.sort_index(inplace=True)
    # Then save it to a csv
    df.to_csv(file, index=False)