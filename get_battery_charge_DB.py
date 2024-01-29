# -*- coding: cp949 -*-   # ���� ���ڵ� ������ ���� ù�Ӹ��� ������־� ���ڵ� ������ ������

import numpy as np
import pandas as pd
import tensorflow as tf # pip install �ʿ�
import matplotlib.pyplot as plt # pip install �ʿ�
import seaborn as sns # pip install �ʿ�
import matplotlib.font_manager as fm # pip install �ʿ�
from matplotlib import rc # pip install �ʿ�
from sklearn.preprocessing import MinMaxScaler
import keras #  pip install �ʿ�
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, LSTM, Bidirectional,LeakyReLU
from sklearn.model_selection import TimeSeriesSplit
import sys
import schedule # pip install �ʿ�
import time
import pymysql # pip install �ʿ�
import datetime # pip install �ʿ�



current_datetime = datetime.datetime.now()
### �Է¾� Ȯ�� �� ���� ����(��ƾ 1)

## ���� �ڵ� ������ ���� �ʿ��� ���ڵ��� Ŀ�ǵ����(sys.argv)���� ���������� ���޵Ǿ����� Ȯ��
## ���� �̸�(Auto_Script.py), ���۳�¥(���糯¥ ���� 7������ ��), ���ᳯ¥(���糯¥),����,�浵, īī��APIŰ, ���ûAPIŰ


# ���� ���� ���� 7���� ����� ���޵Ǿ����� Ȯ���ϴ� �κ�(��ƾ 1-1)
if len(sys.argv) >= 7:  
    start_date_input = sys.argv[1]
    start_date_input = start_date_input.replace('-', '') 
    end_date_input = sys.argv[2] 
    end_date_input = end_date_input.replace('-', '') 
    
# ���� Ŀ�ǵ������ ���ڰ� 7�� ���϶��, �� ���ڰ� �����Ǿ��ų� �߸��� �������� �Էµ� ��Ȳ���� �����Է����� ��ȯ (�����ƾ1)
else: 
   
   print("Ŀ�ǵ���� �Է� ���� ���� �Ǵ� ������ Ȯ���ϼ���.") 
    
   # ���� ��¥�� ���� ��¥ ���� �Է��Ͽ� �����ƾ1 ����(���۳�¥�� ���糯¥ ���� 6�� ����, ���ᳯ¥�� ���糯¥, �������� 1�� �� ��¥)
   try:  
        
        start_date_input = input("������ ������ ���� ��¥�� YYYYMMDD �������� �Է��ϼ���(ex. 2�� 22���� �������� �����Ϸ��� YYYY0215):") 
        start_date_input = start_date_input.replace('-', '') 
       
        end_date_input = input("������ ������ ���� ��¥�� YYYYMMDD �������� �Է��ϼ���(ex. 2�� 22���� �������� �����Ϸ��� YYYY0215):") 
        end_date_input = end_date_input.replace('-', '')  
   # �����ƾ 1 ������ ���, ��ũ��Ʈ ����     
   except ValueError as e: 
        print(f"��ȿ�� �Է°��� �Է����ּ���. {e}") 
        exit() 
        sys.exit() # ��ũ��Ʈ ���� ����
    
# ���۳�¥�� ���ᳯ¥ ��ó��(��ƾ 1-2)
try: 
    start_date = datetime.datetime.strptime(start_date_input, '%Y%m%d') # ���۳�¥�� ȣ��URL�� ����� �� �ִ� �������� ��ó��. �̸� ���� ���ڿ��� datetime��ü�� ��ȯ�ϴ� strptime �޼��� Ȱ�� 
    end_date = datetime.datetime.strptime(end_date_input, '%Y%m%d')  # ���ᳯ¥��  ȣ��URL�� ����� �� �ִ� �������� ��ó��. �̸� ���� ���ڿ��� datetime��ü�� ��ȯ�ϴ� strptime �޼��� Ȱ��


# �Է°� ������ ���� ��ũ��Ʈ ����(��ƾ 1-2-1) 
except ValueError as e:
    print(f"��¥  ���Ŀ� ������ �ֽ��ϴ�. ��¥�� ��� YYYYMMDD �������� �Է��ϼ���. {e}") 
    sys.exit()


# �Է°� ��ó�� ����, �˰��� ������ ���̴� ��¥ ���� ����(��ƾ1-3)
# try ��ϰ� �Բ� ���̴� else���� ���, try-except���� ���ܰ� �߻����� �ʾ��� �� �ٷ� �����
else:
  
    current_date = datetime.date.today()
    # end_date = current_date
    current_date_str = current_date.strftime('%Y%m%d')
    end_date_str = end_date.strftime('%Y%m%d') # datetime��ü�� YYYYMMDD������ ���ڿ��� �ٲ� ���۳�¥(���ó�¥)�� ����
    previous_six_date = end_date - datetime.timedelta(days=6) 
    # start_date = previous_six_date 




### �� �н� �غ�(��ƾ 2)
## ���� 7�������� ����������͸� LSTM ���� �Է°����� �籸���ϴ� �κ�(��ƾ 2-1)


# 7������ ����������͸� �ϳ��� ������ ������ ����
accumulated_df = pd.DataFrame(columns=['RH', 'GHI', 'CC', 'T'])
print(start_date)
print(end_date)


# �ݺ����� ���� ����������� ���Ͽ��� ����������͸� �ϳ��� ������ �߰�
while start_date <= end_date:
    start_date_str = start_date.strftime('%Y%m%d')
    observation_data_path = fr"D:\hyeonseo\Battery_Charging_Algorithm\Battery_Charge_V3\Battery_Charge_V3\API\output_Weather_{start_date_str}_file.csv"



    try:
        #����������� ���Ͽ��� ������ ������ ���ȭ
        df_Weather_API = pd.read_csv(observation_data_path, encoding='cp949')
        column_mapping = {'HM': 'RH', 'SI': 'GHI', 'CA': 'CC', 'TS': 'T'}
        df_Weather_API.rename(columns=column_mapping, inplace=True)

        #����������� ������ column�� �Ű��� Ȱ���� column�� ����
        selected_columns = list(column_mapping.values())
        selected_data = df_Weather_API[selected_columns]
        print(selected_data)

        # concat �޼��带 Ȱ���� 7������ �����͸� �ϳ��� ������ �߰�
        accumulated_df = pd.concat([accumulated_df, selected_data], ignore_index=True)


    except Exception as e:
        print(f"���������� �� �Է°����� ��ȯ�ϴ� �������� ���� �߻�: {e}")

    start_date += datetime.timedelta(days=1)
    

#���� ������ üũ(98,4) �����̾�� ����
# accumulated_df_array = accumulated_df.to_numpy() # ���� �Է°��� accumulated_df_array�� x_test�� ����. �����ϸ� �������
size = accumulated_df.shape
if size !=(98,4):
    print(size)
    print(f"7������ 98�ð��밡 Ȯ������ �ʾҽ��ϴ�. 07�ú��� 20���� �����Ͱ� ������ Ȯ������ �ʾҽ��ϴ�, 20�� ���Ŀ� �ٽ� �õ����ּ���")
            
            
else:
        print("���������� �� �Է°��� ������ ���·� ��ó�� �Ϸ�")



## �� �н��� �ʿ��� �����������, ���н������� �غ�(��ƾ 2-2)
#���� ���ڵ� ������ utf-8�� ����Ʈ ���̸�, ��Ȳ�� ���� cp949, euc-kr Ȱ���� ��
forecast_data_path = fr"D:\hyeonseo\Battery_Charging_Algorithm\Battery_Charge_V3\Battery_Charge_V3\API\output_forecast_{end_date_str}.csv" # ���������� ����
train_x_data_path =r"D:\hyeonseo\Battery_Charging_Algorithm\Battery_Charge_V3\Battery_Charge_V3\Battery_Charging_1_3\ver1_3\df_x_train.csv" # ���н� ����
train_y_data_path =r"D:\hyeonseo\Battery_Charging_Algorithm\Battery_Charge_V3\Battery_Charge_V3\Battery_Charging_1_3\ver1_3\df_y_train.csv"


try:
 df_forecast_API = pd.read_csv(forecast_data_path, encoding='utf-8') 
 print("���������� �б� ����!")
except Exception as e:
 print(f"���������Ϳ��� ���� �߻�: {e}") 

try:
 train_x_data_df = pd.read_csv(train_x_data_path, encoding='utf-8',index_col = 0) 
 X_train = train_x_data_df.drop('datetime', axis = 1)
 print("�н�������1 �б� ����!")
except Exception as e: 
 print(f"�н�������1���� ���� �߻�: {e}") 


try:
  train_y_data_df = pd.read_csv(train_y_data_path, encoding='utf-8',index_col = 0) 
  Y_train = train_y_data_df.drop('datetime', axis = 1)
  print("�н�������2 �б� ����!")
except Exception as e: 

  print(f"�н�������2 ���� ���� �߻�: {e}") 




##�˰��� ���࿡ �ʿ��� �Լ� �� ���� ����(��ƾ 2-3)




# ������ ����ȭ �Լ�. ���� ���� ���������� ���� �� �������� ������ �������̸� �� ���ɹ��ְ� ��Ʊ� ������ ������ ����(0~1�� ��)���� �ٲ���
# scaler = MinMaxScaler(feature_range = (0,1))


# # train set �����ϸ� �Լ�
# def normalize(X,y):
#   X_norm = X.copy()
#   # numpy �迭, 2�������� ��ȯ, �����ϸ�
#   for name in X:
#     temp = X[name].to_numpy().reshape(-1,1)
#     X_norm[name] = scaler.fit_transform(temp)


#   temp = y.to_numpy().reshape(-1,1)
#   y_norm = scaler.fit_transform(temp)

#   return X_norm, y_norm


# # X_train, y_train, x_test �����ϸ� ����
# X_train_norm, y_train_norm = normalize(X_train,Y_train)
# X_train_norm = X_train_norm.to_numpy()

# def normalize_test(data):
#     data_norm = data.copy()
#     for name in data:
#         temp = data[name].to_numpy().reshape(-1,1)
#         data_norm[name] = scaler.transform(temp)
        
#     return data_norm

# accumulated_df_array = normalize_test(accumulated_df)


# �����Ϸ� �ʱ�ȭ

scalers_X = {name: MinMaxScaler(feature_range=(0, 1)) for name in X_train.columns}
scaler_y = MinMaxScaler(feature_range=(0, 1))


def normalize(X_train,X_test,Y_train):
    # X ������ �����ϸ�
    X_train_norm = X_train.copy()
    X_test_norm = X_test.copy()
    Y_train_norm = Y_train.copy()
    for name in X_train.columns:
        X_train_norm[name] = scalers_X[name].fit_transform(X_train_norm[name].values.reshape(-1, 1))
        X_test_norm[name] = scalers_X[name].transform(X_test_norm[name].values.reshape(-1,1))
 
    Y_train_norm = scaler_y.fit_transform(Y_train_norm.values.reshape(-1,1))


    return X_train_norm,X_test_norm,Y_train_norm


# X_train, y_train, x_test �����ϸ� ����
X_train_norm,accumulated_df_array,Y_train_norm = normalize(X_train,accumulated_df,Y_train)
# X_train_norm = X_train_norm.to_numpy()




# LSTM input������ 3D Tensor�� �°� ��ȯ���ִ� �Լ�

input_window = 14*7
output_window = 14*1
stride = 14


def input_maker(xdata, ydata, input_window, output_window, stride):
  L_X = xdata.shape[0]
  L_Y = ydata.shape[0]

  # stride�� ���� ���� ���� ���
  num_samples = min((L_X - input_window) // stride, (L_Y - output_window) // stride) + 1

  # input�� output �ʱ�ȭ
  X = np.zeros([input_window, num_samples, xdata.shape[1]])  # xdata feature ���� ����ϴ� shape[1]
  Y = np.zeros([output_window, num_samples])

  for i in np.arange(num_samples):
      start_x = stride * i
      end_x = start_x + input_window
      X[:, i, :] = xdata[start_x:end_x]

      start_y = stride * i
      end_y = start_y + output_window
      Y[:, i] = ydata[start_y:end_y].flatten()

  # LSTM�� �°� ���� ����
  xdata = X.transpose((1, 0, 2))  # (sample ����, window ũ��, feature ����) �� �𵨿��� x������ 4��(����, ��õ�ϻ緮, ���, �µ�)
  ydata= Y.transpose((1, 0))    # (sample ����, window ũ��) y������ ������ 

  return xdata, ydata



x_train_lstm_input, y_train_lstm_input = input_maker(X_train_norm, Y_train_norm, 98, 14, 14)


# ��ƾ 2-1���� ���� accumulated_df_array ������ �����Ͽ� �� �Է°����� Ȱ��
def model_input_maker(data, window, stride):
  
    L = data.shape[0]

    # stride�� ���� ���� ���� ���
    num_samples = ((L - window) // stride) + 1

    # input�� output �ʱ�ȭ
    X = np.zeros([window, num_samples, data.shape[1]])  # xdata feature ���� ���
  
    

    for i in np.arange(num_samples):
          start_x = stride * i
          end_x = start_x + window
          X[:, i, :] = data[start_x:end_x]


      # LSTM�� �°� ���� ���� (1,98,4)
    result = X.transpose((1, 0, 2))  # (sample ����, window ũ��, feature ����)
  

    return result



# �Ʒõ� �𵨷� ������ �����ϴ� �Լ�
def predict_sunenergy_for_date(date_str, model, data, features): 
       
    print(type(data))

    if data.empty:
        raise ValueError(f"no data available for the date: {date_str}") # �ش� ��¥ ���� �����Ͱ� ���� ��� ValueError ���
    
    else:
        
        model_input = model_input_maker(accumulated_df_array, 98, 14) # �����ϸ� ó���� x_test(accumulated_df_array)�� ��
        predicted_values = model.predict(model_input) # ���͸��� ���� �����͸� ������ ���н� ���� ���� ������ ���� 

    return predicted_values.tolist() # ���� �������� ����Ʈ���·� ��ȯ




#��󿹺������Ϳ��� �ϴû��¸� �����ؿ��� �Լ�.  T������ �ϴû��� ���������Ͱ� �� 20�� �̻� �ֱ� ������ �ֺ��� ����
def get_mode_of_sky(data): 
       
    return int(data['SKY'].mode()[0][-2:]) # mode()[0]�� ���� �ֺ��� ��������, ���� ���ڰ� ���ԵǾ��ֱ⶧���� �����̽��� ���� ���ڰ��� ��ȯ
 

 
# ���������Ϳ��� ���������� �����ؿ��� �Լ�. 
def get_season_final(data):        
    # �ش� ��¥�� ���� 'season' �÷��� �� ��ȯ
     
    if not data.empty: 
        return data['season'].iloc[0] # �������� season �÷����� ù ��° ���� ������ ��. ������ ���� ��¥�� ���������̱� ������ �ε��̿� ������� ���� ����.
    else: 
        raise ValueError(f"no season data available for the date") 




# ������ �¾籤������(���� MWh)�� ���� ���͸� �⺻������ ���� �Լ�. 
def decide_base_charge(sunenergy):
    if 0 <= sunenergy < 500:
        return 100
    elif 500 <= sunenergy < 1500:
        return 90
    elif 1500 <= sunenergy < 2000:
        return 80
    elif 2000 <= sunenergy < 2500:
        return 70
    elif 2500 <= sunenergy < 3000:
        return 60
    elif 3000 <= sunenergy:
        return 50

 
# ���������� �ϴû��¸� �ݿ��ϴ� �߰������� ���� �Լ�. �⺻�������� �߰��������� ���� ���������� ��ȯ
def decide_final_charge(sunenergy, sky_value, season): 
    base_charge = decide_base_charge(sunenergy) 
    additional_charge = 0 
    if season in ['spring', 'summer', 'fall']: 
        if sky_value <= 2:
            additional_charge = 0 
        elif sky_value >= 3: 
            additional_charge = 5 
    elif season == 'winter': 
        if sky_value <= 3: 
            additional_charge = 5 
        elif sky_value >= 4: 
            additional_charge = 10 
    final_charge = base_charge + additional_charge 
    #������ ���� ���� ���� 
    if season == 'summer'or season == 'spring' or season == 'fall': # ��,����,������ ���
        final_charge = min(70, max(50, final_charge)) #�ְ� �������� 70���� ����

    else: # �ܿ��� ���
        final_charge = min(100, max(50, final_charge)) # �ּ� �������� 50, �ְ� �������� 100
    
    return final_charge




### �� ���� �� �н�(��ƾ 3)

def create_combined_model(input_shape): # LSTM �� create �Լ��� input data�� shape�� ���� 
    # �� ��Ű��ó
    model = keras.models.Sequential([
        keras.layers.Input(shape=(98, 4)), # input data shape ���� ���
        keras.layers.Bidirectional(keras.layers.LSTM(256, return_sequences=True, name='BiLSTM_0')), # ���� ��, �ش� layer�� ��� ����� ���� layer�� �ѱ�� return_sequences�ɼ� Ȱ��ȭ, ���ſ� �̷� ���� �ѳ���� �н��ϴ� ����� �� ����
        keras.layers.Dropout(0.5), # �ش� layer�� ���� layer �̵� �� ������ ������ ���� �Ϻ� ���ֿ� ���� ��Ȱ��ȭ ��ġ. 0.5�� ��Ȱ��ȭ ����(50%)
        keras.layers.LSTM(128, return_sequences=True, name='LSTM_1'),
        keras.layers.LSTM(64, return_sequences=True, name='LSTM_2'),
        keras.layers.LSTM(32, return_sequences=True, name='LSTM_3'),
        keras.layers.Flatten(), # LSTM ���̾�� (��ġũ��, ����������, Ư�� ��) 3D �ټ��� ������, FC���̾�(Dense)�� [��ġũ��, Ư�� ��]�� 2���� �迭�� ����. �̿� Flatten() �޼���� ��ġũ�⸦ ������ ������ �� Ư���� �ϳ��� �������� ��źȭ��.
        keras.layers.Dense(64, activation="softsign"), # FC layer(����������), ��� �Է´����� ����Ǿ� ������ Ȱ��ȭ �Լ��� ���� ��°��� ������. softsign �Լ��� tanh�� ������������, �׷����Ʈ �ҽǹ����� ���� ������ ����
        keras.layers.Dense(14,  activation="relu"),# ������ FC layer�� ���� ���� ���� ���� �ʿ��� ��°� ����. �ش� ���� ��� 1������ 14�ð��� ��°��̹Ƿ� 14.
        
    ])
    #�� �����ϸ�
    model.compile(optimizer='adam', loss='mean_squared_error')  # ����ȭ adam, �ս��Լ� MSE(��տ�������)
    return model

# ���� ����(Early Stopping) �ݹ� ����
es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)

# TimeSeriesSplit ����
tscv = TimeSeriesSplit(n_splits=5)

# ���� ���� ����
# train data���� val_dataset�� �̾Ƴ��� �� 5ȸ �� ������ ����.
for train_index, val_index in tscv.split(x_train_lstm_input):
    X_train_fold, X_val_fold = x_train_lstm_input[train_index], x_train_lstm_input[val_index]
    y_train_fold, y_val_fold = y_train_lstm_input[train_index], y_train_lstm_input[val_index]

# �� ����
RL_model_bi = create_combined_model((X_train_fold.shape[1], 4)) # �μ��� (98,4) �Է��ؼ� �� ����


# �� �Ʒ�
#verbose = 0���� �����ϸ� cmd â���� �� �н� ������Ȳ �� ��
history = RL_model_bi.fit(x_train_lstm_input, y_train_lstm_input, epochs=500, batch_size=64,
                    shuffle=False, validation_data=[X_val_fold, y_val_fold],
                    callbacks=[es], verbose=1)


#�𵨰���(�ս��Լ� MSE)
RL_model_bi.evaluate(X_val_fold, y_val_fold)


### �Ʒõ� �𵨷� ���� ������ �� ������ ����(��ƾ 4)


# ����API���ϰ�  predict_sunenergy_for_date �Լ��� ����ؼ� T������ �ð��� �¾籤������ ����.(��ƾ 4-1)
features_mapping = {'RH' : 'HM', 'GHI' : 'SI', 'CC': 'CA', 'T' : 'TS'}
predicted_values_later_date_str = predict_sunenergy_for_date(current_date_str, RL_model_bi, df_Weather_API, list(features_mapping.values()))

predicted_values_later_date_str_array = np.array(predicted_values_later_date_str) # array �������� ��ȯ�Ͽ� 2���� �迭�� ��ȯ�� �� �ֵ��� ��
predicted_values_later_date_str_array = predicted_values_later_date_str_array.reshape(-1,1) # ������ ����ȯ�� ���� 2�����迭�� ��ȯ
predicted_values_later_date_str_scaled = scaler_y.inverse_transform(predicted_values_later_date_str_array) # ������ �����ϸ��� ���� ���� ������ �纯ȯ
daily_predicted_value_later_date_str= np.sum(predicted_values_later_date_str_scaled) # �ð��� �¾籤�������� ���� �Ϻ� �¾籤������

print(predicted_values_later_date_str_scaled)
print(daily_predicted_value_later_date_str)
 
# "later_date_str"�� �ϴ� ���� ��������(��ƾ 4-2)
sky_value_for_later_date_str = get_mode_of_sky(df_forecast_API) 
print(sky_value_for_later_date_str)
#"later_date_str"�� ���� ����  ��������(��ƾ 4-3)
season_for_later_date_str= get_season_final(df_forecast_API) 
season_for_later_date_str
print(season_for_later_date_str)
 
# ���� ���͸� ������ ���(��ƾ 4-4)
# �¾籤������, �ϴû���, ���������� ������ ���������� ����ϱ�
combined_battery_charge = decide_final_charge(daily_predicted_value_later_date_str, sky_value_for_later_date_str, season_for_later_date_str)
print(combined_battery_charge)
 


 
### �˰��� �α����� ����(��ƾ 5)


result = { 
     
    'datetime' : [current_date_str], # ��¥
    'daily_predicted_value_later_date_str' : [daily_predicted_value_later_date_str], # �Ϻ� �¾籤������
    'sky_value_for_later_date_str' : [sky_value_for_later_date_str], # �ϴû��� ����������
    'season_for_later_date_str': [season_for_later_date_str], # ��������
    'tommorow_battery_charge': [combined_battery_charge], # ����������
     
    }

print(result)

tommorow_battery_charge_file = pd.DataFrame(result, index = [0]) # result�� ������������ȭ




try:
    
    tommorow_battery_charge_file.to_csv(fr"D:\hyeonseo\Battery_Charging_Algorithm\Battery_Charge_V3\Battery_Charge_V3\Results\{end_date_str}_battery_charging_file.csv", index=False) # csv���Ϸ� ����
    print(fr"D:\hyeonseo\Battery_Charging_Algorithm\Battery_Charge_V3\Battery_Charge_V3\Results\{end_date_str}_battery_charging_file.csv")
except:
    print(f"result ���� ���� �� ���� �߻�")# ���� �߻� �� �����޽��� ���
    print(fr"D:\hyeonseo\Battery_Charging_Algorithm\Battery_Charge_V3\Battery_Charge_V3\Results\{end_date_str}_battery_charging_file.csv")


###  ���͸� ���� ������ ��(CoC)�� DB�� ����



# #MYSQL����
try: 
    conn = pymysql.connect(host='175.121.197.205',port =3306, user='bkc_manager', password='bkcManager1234@', db='smart_lora', charset='utf8')
    print(f" DB�����Ϸ�")
except:
    print(f"DB���� �� ���� �߻�. ���� �̸�, ��й�ȣ Ȥ�� ����ڱ����� �����Ǿ� �ִ��� ��Ȯ���ϼ���")

#Ŀ������
cur = conn.cursor()


try:
    # device_id���� ������ ����Ʈ �ʱ�ȭ
    device_ids = []

    # �Է� ���� �ۼ� �� ����
    # a.facility_dno  = '35';�� ���� facility_dno�� �¾籤���ε� �ĺ���ȣ�� �ǹ���. ����Ǿ��� �� �ڵ������� �ٲ���� ��
    sql_equip = """
    select a.*
    from equipment a, equipment_type b
    where a.type_dno = b.dno
    and binary b.service like '%c%'
    and binary b.service like '%d%'
    and a.facility_dno  = '35';
    """
    cur.execute(sql_equip)
    
    # ��� ����
    result_device = cur.fetchall()
    print(result_device)
    

    # �� ���ڵ忡�� device_id �����Ͽ� ����Ʈ�� ����
    for record in result_device:
        device_id = record[10]  # device_id ����
        device_ids.append(device_id)
        print(device_ids)
        
    # �� device_id�� ���� �߰� ���� ����
    for id in device_ids:
        # ���⿡ �� device_id�� ���� ������ �ۼ��ϰ� ����
        #�Է� ���� �ۼ�
        try:    

            sql_setting = """update setting
            set solar_coc = %s, updated = %s
            where r_device_id = %s;"""

        # ���� ����
            cur.execute(sql_setting, (combined_battery_charge,current_datetime, id))
        # �Է°� ����
            conn.commit()
            print(f"DB��  {id}�� �Է°� ���� �Ϸ�. ����� �ð� {current_datetime}, ������Ʈ �� �� ��: {cur.rowcount}")
    


        except:
            print(f"setting������ ���� �߻�. ���� �� ��Ÿ ���� ��Ȯ���ϼ���.")

except:
            print(f"��� ID������ ���� �߻�. ���� �� ��Ÿ ���� ��Ȯ���ϼ���.")


# MYSQL ���� ����
conn.close()