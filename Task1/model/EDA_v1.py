import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import sys
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from math import radians, cos, sin, asin, sqrt
from sklearn.metrics import mean_squared_error as mse


class EDA():
    def __init__(self, df_train, df_test=None):
        self.df = df_train
        self.bed_bath_size_model = None
        self.df_test = df_test
        self.built_year_mean = 0.0
        self.built_year_min = 0.0
        self.built_year_range = 0.0
        
    def str_clean_up(self):
        self.df['title'] = self.df['title'].str.lower()
        self.df['title'] = self.df['title'].str.replace(r'[^\w\s]+', '', regex = True)
        self.df['address'] = self.df['address'].str.lower()
        self.df['address'] = self.df['address'].str.replace(r'[^\w\s]+', '', regex = True)
        self.df['property_name'] = self.df['property_name'].str.lower()
        self.df['property_name'] = self.df['property_name'].str.replace(r'[^\w\s]+', '', regex = True)
        self.df['property_type'] = self.df['property_type'].str.lower()
        self.df['property_type'] = self.df['property_type'].str.replace(r'[^\w\s]+', '', regex = True)
        self.df['planning_area'] = self.df['planning_area'].str.lower()
        self.df['planning_area'] = self.df['planning_area'].str.replace(r'[^\w\s]+', '', regex = True)
        self.df['subzone'] = self.df['subzone'].str.lower()
        self.df['subzone'] = self.df['subzone'].str.replace(r'[^\w\s]+', '', regex = True)
        
        
        self.df_test['title'] = self.df_test['title'].str.lower()
        self.df_test['title'] = self.df_test['title'].str.replace(r'[^\w\s]+', '', regex = True)
        self.df_test['address'] = self.df_test['address'].str.lower()
        self.df_test['address'] = self.df_test['address'].str.replace(r'[^\w\s]+', '', regex = True)
        self.df_test['property_name'] = self.df_test['property_name'].str.lower()
        self.df_test['property_name'] = self.df_test['property_name'].str.replace(r'[^\w\s]+', '', regex = True)
        self.df_test['property_type'] = self.df_test['property_type'].str.lower()
        self.df_test['property_type'] = self.df_test['property_type'].str.replace(r'[^\w\s]+', '', regex = True)
        self.df_test['planning_area'] = self.df_test['planning_area'].str.lower()
        self.df_test['planning_area'] = self.df_test['planning_area'].str.replace(r'[^\w\s]+', '', regex = True)
        self.df_test['subzone'] = self.df_test['subzone'].str.lower()
        self.df_test['subzone'] = self.df_test['subzone'].str.replace(r'[^\w\s]+', '', regex = True)
        
        
    # handle abnormal data in train dataset
    def handle_train_abnormal(self):
        
        # abnormal floor area
        self.df = self.df.drop(self.df[self.df['size_sqft'] == 0 ].index)
        self.df = self.df.drop(self.df[self.df['size_sqft'] > 100000 ].index)
        
        # abnormal lng & lat
        self.df = self.df.drop(self.df[self.df['lng'] > 104.6 ].index)
        self.df = self.df.drop(self.df[self.df['lng'] < 103.38 ].index)
        self.df = self.df.drop(self.df[self.df['lat'] < 1.23 ].index)
        self.df = self.df.drop(self.df[self.df['lat'] > 1.5 ].index)
                
        # abnormal num of beds and baths
        bed_arr = self.df['num_beds'].to_numpy()
        bath_arr = self.df['num_baths'].to_numpy()
        diff_arr = abs(bed_arr - bath_arr)
        diff_index = np.where(diff_arr > 7)[0]
        self.df = self.df.drop(diff_index)
        # abnormal price
        self.df = self.df.drop(self.df[self.df['price'] == 0].index)
        self.df = self.df.drop(self.df[self.df['price'] > 100000000].index)
        
#         col = self.df['price']
#         iqr = col.quantile(0.75) - col.quantile(0.25)
#         u_th = col.quantile(0.75) + 1.5*iqr # upper bound
#         l_th = col.quantile(0.25) - 1.5*iqr # lower bound
#         self.df = self.df[(col <= u_th) & (col >= l_th)]
        
        self.df = self.df.reset_index(drop=True)
        
    def handle_property_type(self, property_type):
        if 'land' in property_type:
            return 'landed'
        elif 'bungalow' in property_type:
            return 'bungalow'
        else:
            return property_type

    def property_type_method(self):
        df_concat = pd.concat([self.df,self.df_test],axis=0, ignore_index=True)
        df_concat['property_type'] = df_concat['property_type'].map(self.handle_property_type)
        one_hot_property_type = pd.get_dummies(df_concat['property_type'], prefix='property_type', prefix_sep='_')
        df_concat = df_concat.join(one_hot_property_type)
        
        self.df = df_concat.loc[0:len(self.df)-1]
        
        self.df_test = df_concat.loc[len(self.df):]
        self.df_test = self.df_test.drop(columns=['price'])
        self.df_test = self.df_test.reset_index(drop=True)
        
    
    
    # process method for feature: furnishing         
    def furnishing_method(self):
        
        
        df_concat = pd.concat([self.df,self.df_test],axis=0, ignore_index=True)
        df_concat = df_concat.replace({'furnishing':{'na':'unspecified'}})
        one_hot_furnishing = pd.get_dummies(df_concat['furnishing'], prefix='furnish', prefix_sep='_')
        df_concat = df_concat.join(one_hot_furnishing)
        
        self.df = df_concat.loc[0:len(self.df)-1]
        
        self.df_test = df_concat.loc[len(self.df):]
        self.df_test = self.df_test.drop(columns=['price'])
        self.df_test = self.df_test.reset_index(drop=True)
            
    
    def handle_built_year1(self, df_nan, df_find):
        df = df_find.dropna(subset=['built_year'])
        df = df.reset_index(drop=True)        
        
        index = np.where(df_nan['built_year'].isnull())[0]
        for i in index:
            name = df_nan.loc[i, 'property_name']
            index_train = np.where(df['property_name'] == name)[0]
            year_sum = 0.0
            if len(index_train) == 0:
                continue
            for j in index_train:
                year_sum += df.loc[j, 'built_year']
            df_nan.loc[i, 'built_year'] = float(round(year_sum/len(index_train)))
            
            
    # process method 1 for feature: built_year    
    def built_year_method1(self, for_test=False):
        if for_test:
            self.handle_built_year1(self.df_test, self.df)
            self.df_test['built_year'] = self.df_test['built_year'].fillna(self.built_year_mean)
        else:
            self.handle_built_year1(self.df, self.df)
            self.df = self.df.dropna(subset=['built_year'])
            self.built_year_mean = round(self.df['built_year'].mean())
    
    
    def handle_built_year2(self, year):
        count = 1
        if year == -1:
            return 0
        
        while(self.built_year_min + count*self.built_year_range) < year:
            count += 1
        return count
            
    # process method for feature: built_year    
    def built_year_method2(self, for_test=False):
        if for_test:
            self.df_test['built_year'] = self.df_test['built_year'].fillna(-1)
            self.df_test['built_year'] = self.df_test['built_year'].map(self.handle_built_year2)
        else:
            self.built_year_min = self.df['built_year'].min()
            self.built_year_range = 10
            self.df['built_year'] = self.df['built_year'].fillna(-1)
            self.df['built_year'] = self.df['built_year'].map(self.handle_built_year2)
            
    def handle_tenure_v1(self, tenure):
        if tenure == 'NAN':
            return 0
        if 'freehold' in tenure:
            return 3

        lease_year = re.findall(r"\d+",tenure)
        if len(lease_year) == 0:
            return 0

        lease_year = int(lease_year[0])
        if lease_year < 900:
            return 1
        elif lease_year < 1900:
            return 2
        else:
            return 3

    # process method for feature: tenure
    def tenure_method(self, for_test=False):
        if for_test:
            self.df_test['tenure'] = self.df_test['tenure'].fillna('NAN')
            self.df_test['tenure'] = self.df_test['tenure'].map(self.handle_tenure_v1)
        else:
            self.df['tenure'] = self.df['tenure'].fillna('NAN')
            self.df['tenure'] = self.df['tenure'].map(self.handle_tenure_v1)

    def num_bed_bath_regressor(self):
        df_re = self.df[['num_beds', 'num_baths', 'size_sqft']]
        df_re = df_re.dropna()
        
        X = df_re[['num_beds', 'num_baths']].to_numpy()
        y = df_re['size_sqft'].to_numpy()
        y = y / 100
        
        model = LinearRegression()
        model.fit(X, y)
        
        self.bed_bath_size_model = model
        
        
    def handle_bed_bath(self, df):
        w1, w2 = self.bed_bath_size_model.coef_
        w0 = self.bed_bath_size_model.intercept_

        df['num_beds'] = df['num_beds'].fillna(-1)
        df['num_baths'] = df['num_baths'].fillna(-1)

        index1 = list(np.where(df['num_beds'] == -1)[0])
        index2 = list(np.where(df['num_baths'] == -1)[0])
        index = list(set(index1+index2))

        for i in index:
            size_sqft = df.loc[i,'size_sqft']
            num_beds = df.loc[i,'num_beds']
            num_baths = df.loc[i,'num_baths']

            # only num_beds feature is NaN
            if num_beds == -1 and num_baths != -1:
                df.loc[i,'num_beds'] = round((size_sqft/100 - w0 - w2*num_baths) / w1)
            # only num_baths feature is NaN
            elif num_beds != -1 and num_baths == -1:
                df.loc[i,'num_baths'] = round((size_sqft/100 - w0 - w1*num_beds) / w2)
            # both num_baths and num_beds feature are NaN
            else:
                size_range = df[(size_sqft-100 <= df['size_sqft']) & (df['size_sqft'] <= size_sqft+100)]
                ave_num_beds = round(size_range['num_beds'].sum() / len(size_range))
                ave_num_baths = round(size_range['num_baths'].sum() / len(size_range))
                df.loc[i,'num_beds'] = ave_num_beds
                df.loc[i,'num_baths'] = ave_num_baths
                    
        
    # process method for feature: num_beds & num_baths
    def num_bed_bath_method(self, for_test=False):
        if for_test:
            self.handle_bed_bath(self.df_test)
        else:
            self.num_bed_bath_regressor()
            self.handle_bed_bath(self.df)
    
    
    # using lat and lng to fill missing planning_area by finding min dis subzone in train datasets
    def fill_missing_planning_area(self):
        
        df_ref = self.df.copy()
        df_ref = df_ref.dropna(subset=['planning_area'])
        df_ref = df_ref.reset_index(drop=True)
        
        lat_list = df_ref['lat'].to_numpy()
        lng_list = df_ref['lng'].to_numpy()
        
        
        # filling for train set
        self.df['planning_area'] = self.df['planning_area'].fillna('NAN')   
        index_list = list(np.where(self.df['planning_area'] == 'NAN')[0])
        for i in index_list:
            lat_cur = self.df.loc[i,'lat']
            lng_cur = self.df.loc[i,'lng']
            
            _, nearest_subzone_index = self.geodistance_min(lat_cur, lng_cur, lat_list, lng_list, return_index=True)
            self.df.loc[i,'planning_area'] = df_ref.loc[nearest_subzone_index, 'planning_area']
            
        
        # filling for test set
        self.df_test['planning_area'] = self.df_test['planning_area'].fillna('NAN') 
        index_list = list(np.where(self.df_test['planning_area'] == 'NAN')[0])
        for i in index_list:
            lat_cur = self.df_test.loc[i,'lat']
            lng_cur = self.df_test.loc[i,'lng']
            
            _, nearest_subzone_index = self.geodistance_min(lat_cur, lng_cur, lat_list, lng_list, return_index=True)
            self.df_test.loc[i,'planning_area'] = df_ref.loc[nearest_subzone_index, 'planning_area']
    
    
                
    # using lat and lng to fill missing subzone by finding min dis subzone in train datasets
    def fill_missing_subzone(self):
        
        df_ref = self.df.copy()
        df_ref = df_ref.dropna(subset=['subzone'])
        df_ref = df_ref.reset_index(drop=True)
        
        lat_list = df_ref['lat'].to_numpy()
        lng_list = df_ref['lng'].to_numpy()
        
        
        # filling for train set
        self.df['subzone'] = self.df['subzone'].fillna('NAN')   
        index_list = list(np.where(self.df['subzone'] == 'NAN')[0])
        for i in index_list:
            lat_cur = self.df.loc[i,'lat']
            lng_cur = self.df.loc[i,'lng']
            
            _, nearest_subzone_index = self.geodistance_min(lat_cur, lng_cur, lat_list, lng_list, return_index=True)
            self.df.loc[i,'subzone'] = df_ref.loc[nearest_subzone_index, 'subzone']
            
        
        # filling for test set
        self.df_test['subzone'] = self.df_test['subzone'].fillna('NAN') 
        index_list = list(np.where(self.df_test['subzone'] == 'NAN')[0])
        for i in index_list:
            lat_cur = self.df_test.loc[i,'lat']
            lng_cur = self.df_test.loc[i,'lng']
            
            _, nearest_subzone_index = self.geodistance_min(lat_cur, lng_cur, lat_list, lng_list, return_index=True)
            self.df_test.loc[i,'subzone'] = df_ref.loc[nearest_subzone_index, 'subzone']
    
    # find min distance  between location list using lat & lng
    def geodistance_min(self, lat_cur, lng_cur, lat_list, lng_list, return_index=False):
        shortest_dis = sys.maxsize
        index = 0
        for i in range(len(lat_list)):
            lng1, lat1, lng2, lat2 = map(radians, [float(lng_cur), float(lat_cur), float(lng_list[i]), float(lat_list[i])])
            dlon = lng2 - lng1
            dlat = lat2 - lat1
            a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
            distance= 2*asin(sqrt(a))*6371*1000
            distance= round(distance/1000,3)
            if distance < shortest_dis:
                shortest_dis = distance
                index = i
        if return_index:
            return shortest_dis, index
        else:
            return shortest_dis
    
        
    # insert new col which shows the min dis between different commercial type
    def cal_min_dis_to_diff_commercial(self, df_commercial, for_test=False):
        
        commercial_type_list = df_commercial['type'].unique()
        
        if for_test:
            self.df_test = self.df_test.reset_index(drop=True)
            for type_cur in commercial_type_list:
                df_type = df_commercial[df_commercial['type']==type_cur]
                tar_lat_list = df_type['lat'].to_numpy()
                tar_lng_list = df_type['lng'].to_numpy()
                min_dis_array = np.zeros(len(self.df_test))
                for i in range(len(self.df_test)):
                    lat_cur = self.df_test.loc[i,'lat']
                    lng_cur = self.df_test.loc[i,'lng']
                    min_dis_array[i] = self.geodistance_min(lat_cur, lng_cur, tar_lat_list, tar_lng_list)

                # insert new col
                col_name = 'min_dis_commercial_'+type_cur
                self.df_test[col_name] = min_dis_array
        else:
            self.df = self.df.reset_index(drop=True)
            for type_cur in commercial_type_list:
                df_type = df_commercial[df_commercial['type']==type_cur]
                tar_lat_list = df_type['lat'].to_numpy()
                tar_lng_list = df_type['lng'].to_numpy()
                min_dis_array = np.zeros(len(self.df))
                for i in range(len(self.df)):
                    lat_cur = self.df.loc[i,'lat']
                    lng_cur = self.df.loc[i,'lng']
                    min_dis_array[i] = self.geodistance_min(lat_cur, lng_cur, tar_lat_list, tar_lng_list)

                # insert new col
                col_name = 'min_dis_commercial_'+type_cur
                self.df[col_name] = min_dis_array
    
    
    # insert new col which shows the min dis between different mrt line
    def cal_min_dis_to_diff_mrt(self, df_mrt, for_test=False):
        mrt_line_list = df_mrt['line'].unique()
        
        if for_test:
            self.df_test = self.df_test.reset_index(drop=True)
            for type_cur in mrt_line_list:
                df_type = df_mrt[df_mrt['line']==type_cur]
                tar_lat_list = df_type['lat'].to_numpy()
                tar_lng_list = df_type['lng'].to_numpy()
                min_dis_array = np.zeros(len(self.df_test))
                for i in range(len(self.df_test)):
                    lat_cur = self.df_test.loc[i,'lat']
                    lng_cur = self.df_test.loc[i,'lng']
                    min_dis_array[i] = self.geodistance_min(lat_cur, lng_cur, tar_lat_list, tar_lng_list)

                # insert new col
                col_name = 'min_dis_mrt_'+type_cur
                self.df_test[col_name] = min_dis_array
        else:
            self.df = self.df.reset_index(drop=True)
            for type_cur in mrt_line_list:
                df_type = df_mrt[df_mrt['line']==type_cur]
                tar_lat_list = df_type['lat'].to_numpy()
                tar_lng_list = df_type['lng'].to_numpy()
                min_dis_array = np.zeros(len(self.df))
                for i in range(len(self.df)):
                    lat_cur = self.df.loc[i,'lat']
                    lng_cur = self.df.loc[i,'lng']
                    min_dis_array[i] = self.geodistance_min(lat_cur, lng_cur, tar_lat_list, tar_lng_list)

                # insert new col
                col_name = 'min_dis_mrt_'+type_cur
                self.df[col_name] = min_dis_array
        
        
    # insert new col which shows the min dis between pri_school or sec_school or mall
    # pc means primary school; sc means second school; sm means shopping mall;
    def cal_min_dis_to_school_or_mall(self, com_df, cal_type='pc', for_test=False):
        
        tar_lat_list = com_df['lat'].to_numpy()
        tar_lng_list = com_df['lng'].to_numpy()
        
        if for_test:
            self.df_test = self.df_test.reset_index(drop=True)

            min_dis_array = np.zeros(len(self.df_test))
            for i in range(len(self.df_test)):
                lat_cur = self.df_test.loc[i,'lat']
                lng_cur = self.df_test.loc[i,'lng']
                min_dis_array[i] = self.geodistance_min(lat_cur, lng_cur, tar_lat_list, tar_lng_list)

            # insert new col
            col_name = 'min_dis_'+ cal_type
            self.df_test[col_name] = min_dis_array
        else:
            self.df = self.df.reset_index(drop=True)
            min_dis_array = np.zeros(len(self.df))
            for i in range(len(self.df)):
                lat_cur = self.df.loc[i,'lat']
                lng_cur = self.df.loc[i,'lng']
                min_dis_array[i] = self.geodistance_min(lat_cur, lng_cur, tar_lat_list, tar_lng_list)

            # insert new col
            col_name = 'min_dis_'+ cal_type
            self.df[col_name] = min_dis_array  
            
    
    def attach_subzone_auxiliary_info(self, df_subzone, for_test=False):
        self.fill_missing_subzone()
        df_subzone['name'] = df_subzone['name'].str.lower()
        df_subzone['name'] = df_subzone['name'].str.replace(r'[^\w\s]+', '', regex = True)
        if for_test:
            length = len(self.df_test)
            size_array = np.zeros(length)
            population_array = np.zeros(length)
            
            for i in range(length):
                subzone = self.df_test.loc[i, 'subzone']
                size_array[i] = df_subzone[df_subzone['name']==subzone]['area_size'].to_numpy()[0]
                population_array[i] = df_subzone[df_subzone['name']==subzone]['population'].to_numpy()[0]
            
            self.df_test['subzone_size'] = size_array
            self.df_test['subzone_population'] = population_array
            
        else:
            length = len(self.df)
            size_array = np.zeros(length)
            population_array = np.zeros(length)
            
            for i in range(length):
                subzone = self.df.loc[i, 'subzone']
                size_array[i] = df_subzone[df_subzone['name']==subzone]['area_size'].to_numpy()[0]
                population_array[i] = df_subzone[df_subzone['name']==subzone]['population'].to_numpy()[0]
            
            self.df['subzone_size'] = size_array
            self.df['subzone_population'] = population_array

            
            
    def cal_subzone_population_density(self, df_subzone, for_test=False):
        self.fill_missing_subzone()
        df_subzone['name'] = df_subzone['name'].str.lower()
        df_subzone['name'] = df_subzone['name'].str.replace(r'[^\w\s]+', '', regex = True)
        if for_test:
            length = len(self.df_test)
            size_array = np.zeros(length)
            population_array = np.zeros(length)
            
            for i in range(length):
                subzone = self.df_test.loc[i, 'subzone']
                size_array[i] = df_subzone[df_subzone['name']==subzone]['area_size'].to_numpy()[0]
                population_array[i] = df_subzone[df_subzone['name']==subzone]['population'].to_numpy()[0]
            
            self.df_test['pop_density'] = (population_array/size_array)
            
        else:
            length = len(self.df)
            size_array = np.zeros(length)
            population_array = np.zeros(length)
            
            for i in range(length):
                subzone = self.df.loc[i, 'subzone']
                size_array[i] = df_subzone[df_subzone['name']==subzone]['area_size'].to_numpy()[0]
                population_array[i] = df_subzone[df_subzone['name']==subzone]['population'].to_numpy()[0]
            
            self.df['pop_density'] = (population_array/size_array)
    
    
    # one-hot planning area
    def planning_area_method(self):
        self.fill_missing_planning_area()
        df_concat = pd.concat([self.df,self.df_test],axis=0, ignore_index=True)
        one_hot_planning_area = pd.get_dummies(df_concat['planning_area'], prefix='pl_area', prefix_sep='_')
        df_concat = df_concat.join(one_hot_planning_area)
        
        self.df = df_concat.loc[0:len(self.df)-1]
        
        self.df_test = df_concat.loc[len(self.df):]
        self.df_test = self.df_test.drop(columns=['price'])
        self.df_test = self.df_test.reset_index(drop=True)
         
    
    
    # comment how to use each feaure function
    def setup_help(self):
        # string value formalization
        self.str_clean_up()
        # remove abnormal data
        self.handle_train_abnormal()
        
        
        # ------ Original Features -------------
        # one-hot property type for both train & test
        self.property_type_method()
        
        # processing tenure feature for train data
        self.tenure_method()
        # processing tenure feature for test data
        self.tenure_method(for_test=True)
        
        # processing num of beds & baths feature for train data
        self.num_bed_bath_method()
        # processing num of beds & baths feature for test data
        self.num_bed_bath_method(for_test=True)
        
        # NOTE: put it into last step, after all other features are processed
        # processing built year feature for train data using method 1
        self.built_year_method1()
        # processing built year feature for test data using method 1
        self.built_year_method1(for_test=True)
        
        # processing built year feature for train data using method 2
        self.built_year_method2()
        # processing built year feature for test data using method 2
        self.built_year_method2(for_test=True)
        
        # one-hot furnishing for both train & test data
        self.furnishing_method()
        
        # one-hot planning area for both train & test data
        self.planning_area_method()
        
        
        # ------ Auxiliary Features -------------
        # calculate shorest distance to different commerical type for train data
        self.cal_min_dis_to_diff_commercial(df_commercial)
        # calculate shorest distance to different commerical type for test data
        self.cal_min_dis_to_diff_commercial(df_commercial, for_test=True)
        
        # calculate shorest distance to different MRT lines for train data
        self.cal_min_dis_to_diff_mrt(df_mrt)
        # calculate shorest distance to different MRT lines for test data
        self.cal_min_dis_to_diff_mrt(df_mrt, for_test=True)

        # calculate shortest distance to primary school for train data
        self.cal_min_dis_to_school_or_mall(df_pri_school, cal_type='pc')
        # calculate shortest distance to primary school for test data
        self.cal_min_dis_to_school_or_mall(df_pri_school, cal_type='pc', for_test=True)

        # calculate shortest distance to second school for train data
        self.cal_min_dis_to_school_or_mall(df_sec_school, cal_type='sc')
        # calculate shortest distance to second school for test data
        self.cal_min_dis_to_school_or_mall(df_sec_school, cal_type='sc', for_test=True)

        # calculate shortest distance to shopping mall for train data
        self.cal_min_dis_to_school_or_mall(df_mall, cal_type='sm')
        # calculate shortest distance to shopping mall for test data
        self.cal_min_dis_to_school_or_mall(df_mall, cal_type='sm', for_test=True)

        # attach with subzone size & population for train data
        self.attach_subzone_auxiliary_info(df_subzone)
        # attach with subzone size & population for test data
        self.attach_subzone_auxiliary_info(df_subzone, for_test=True)

        # calculate population density for train data
        self.cal_subzone_population_density(df_subzone)
        # calculate population density for test data
        self.cal_subzone_population_density(df_subzone, for_test=True)
    
     
        

        


        