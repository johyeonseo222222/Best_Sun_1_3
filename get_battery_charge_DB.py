# -*- coding: cp949 -*-   # 파일 인코딩 형식을 파일 첫머리에 명시해주어 인코딩 오류를 방지함

import numpy as np
import pandas as pd
import tensorflow as tf # pip install 필요
import matplotlib.pyplot as plt # pip install 필요
import seaborn as sns # pip install 필요
import matplotlib.font_manager as fm # pip install 필요
from matplotlib import rc # pip install 필요
from sklearn.preprocessing import MinMaxScaler
import keras #  pip install 필요
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, LSTM, Bidirectional,LeakyReLU
from sklearn.model_selection import TimeSeriesSplit
import sys
import schedule # pip install 필요
import time
import pymysql # pip install 필요
import datetime # pip install 필요



current_datetime = datetime.datetime.now()
### 입력어 확인 및 설정 절차(루틴 1)

## 파일 자동 실행을 위해 필요한 인자들이 커맨드라인(sys.argv)으로 정상적으로 전달되었는지 확인
## 파일 이름(Auto_Script.py), 시작날짜(현재날짜 기준 7개일자 전), 종료날짜(현재날짜),위도,경도, 카카오API키, 기상청API키


# 파일 실행 인자 7개가 제대로 전달되었는지 확인하는 부분(루틴 1-1)
if len(sys.argv) >= 7:  
    start_date_input = sys.argv[1]
    start_date_input = start_date_input.replace('-', '') 
    end_date_input = sys.argv[2] 
    end_date_input = end_date_input.replace('-', '') 
    
# 만약 커맨드라인의 인자가 7개 이하라면, 즉 인자가 누락되었거나 잘못된 형식으로 입력된 상황에는 수동입력으로 전환 (서브루틴1)
else: 
   
   print("커맨드라인 입력 인자 개수 또는 형식을 확인하세요.") 
    
   # 시작 날짜와 종료 날짜 먼저 입력하여 서브루틴1 시작(시작날짜는 현재날짜 기준 6일 이전, 종료날짜는 현재날짜, 예측일은 1일 후 날짜)
   try:  
        
        start_date_input = input("예측을 수행할 시작 날짜를 YYYYMMDD 형식으로 입력하세요(ex. 2월 22일의 충전량을 예측하려면 YYYY0215):") 
        start_date_input = start_date_input.replace('-', '') 
       
        end_date_input = input("예측을 수행할 종료 날짜를 YYYYMMDD 형식으로 입력하세요(ex. 2월 22일의 충전량을 예측하려면 YYYY0215):") 
        end_date_input = end_date_input.replace('-', '')  
   # 서브루틴 1 실패의 경우, 스크립트 종료     
   except ValueError as e: 
        print(f"유효한 입력값을 입력해주세요. {e}") 
        exit() 
        sys.exit() # 스크립트 실행 종료
    
# 시작날짜와 종료날짜 전처리(루틴 1-2)
try: 
    start_date = datetime.datetime.strptime(start_date_input, '%Y%m%d') # 시작날짜를 호출URL에 사용할 수 있는 형식으로 전처리. 이를 위해 문자열을 datetime객체로 변환하는 strptime 메서드 활용 
    end_date = datetime.datetime.strptime(end_date_input, '%Y%m%d')  # 종료날짜를  호출URL에 사용할 수 있는 형식으로 전처리. 이를 위해 문자열을 datetime객체로 변환하는 strptime 메서드 활용


# 입력값 오류에 따른 스크립트 종료(루틴 1-2-1) 
except ValueError as e:
    print(f"날짜  형식에 오류가 있습니다. 날짜의 경우 YYYYMMDD 형식으로 입력하세요. {e}") 
    sys.exit()


# 입력값 전처리 이후, 알고리즘 내에서 쓰이는 날짜 변수 정의(루틴1-3)
# try 블록과 함께 쓰이는 else문의 경우, try-except에서 예외가 발생하지 않았을 때 바로 실행됨
else:
  
    current_date = datetime.date.today()
    # end_date = current_date
    current_date_str = current_date.strftime('%Y%m%d')
    end_date_str = end_date.strftime('%Y%m%d') # datetime객체를 YYYYMMDD형식의 문자열로 바꿔 시작날짜(오늘날짜)에 저장
    previous_six_date = end_date - datetime.timedelta(days=6) 
    # start_date = previous_six_date 




### 모델 학습 준비(루틴 2)
## 이전 7개일자의 종관기상데이터를 LSTM 모델의 입력값으로 재구성하는 부분(루틴 2-1)


# 7개일자 종관기상데이터를 하나의 변수에 모으는 과정
accumulated_df = pd.DataFrame(columns=['RH', 'GHI', 'CC', 'T'])
print(start_date)
print(end_date)


# 반복문을 통해 종관기상데이터 파일에서 종관기상데이터를 하나의 변수에 추가
while start_date <= end_date:
    start_date_str = start_date.strftime('%Y%m%d')
    observation_data_path = fr"D:\hyeonseo\Battery_Charging_Algorithm\Battery_Charge_V3\Battery_Charge_V3\API\output_Weather_{start_date_str}_file.csv"



    try:
        #종관기상데이터 파일에서 추출할 데이터 명목화
        df_Weather_API = pd.read_csv(observation_data_path, encoding='cp949')
        column_mapping = {'HM': 'RH', 'SI': 'GHI', 'CA': 'CC', 'TS': 'T'}
        df_Weather_API.rename(columns=column_mapping, inplace=True)

        #종관기상데이터 파일의 column을 매개로 활용할 column만 추출
        selected_columns = list(column_mapping.values())
        selected_data = df_Weather_API[selected_columns]
        print(selected_data)

        # concat 메서드를 활용해 7개일자 데이터를 하나의 변수에 추가
        accumulated_df = pd.concat([accumulated_df, selected_data], ignore_index=True)


    except Exception as e:
        print(f"종관데이터 모델 입력값으로 변환하는 과정에서 오류 발생: {e}")

    start_date += datetime.timedelta(days=1)
    

#변수 사이즈 체크(98,4) 형태이어야 정상
# accumulated_df_array = accumulated_df.to_numpy() # 모델의 입력값인 accumulated_df_array는 x_test의 역할. 스케일링 해줘야함
size = accumulated_df.shape
if size !=(98,4):
    print(size)
    print(f"7개일자 98시간대가 확보되지 않았습니다. 07시부터 20시의 데이터가 완전히 확보되지 않았습니다, 20시 이후에 다시 시도해주세요")
            
            
else:
        print("종관데이터 모델 입력값에 적절한 형태로 전처리 완료")



## 모델 학습에 필요한 예보기상데이터, 모델학습데이터 준비(루틴 2-2)
#파일 인코딩 형식은 utf-8이 디폴트 값이며, 상황에 따라 cp949, euc-kr 활용할 것
forecast_data_path = fr"D:\hyeonseo\Battery_Charging_Algorithm\Battery_Charge_V3\Battery_Charge_V3\API\output_forecast_{end_date_str}.csv" # 예보데이터 파일
train_x_data_path =r"D:\hyeonseo\Battery_Charging_Algorithm\Battery_Charge_V3\Battery_Charge_V3\Battery_Charging_1_3\ver1_3\df_x_train.csv" # 모델학습 파일
train_y_data_path =r"D:\hyeonseo\Battery_Charging_Algorithm\Battery_Charge_V3\Battery_Charge_V3\Battery_Charging_1_3\ver1_3\df_y_train.csv"


try:
 df_forecast_API = pd.read_csv(forecast_data_path, encoding='utf-8') 
 print("예보데이터 읽기 성공!")
except Exception as e:
 print(f"예보데이터에서 오류 발생: {e}") 

try:
 train_x_data_df = pd.read_csv(train_x_data_path, encoding='utf-8',index_col = 0) 
 X_train = train_x_data_df.drop('datetime', axis = 1)
 print("학습데이터1 읽기 성공!")
except Exception as e: 
 print(f"학습데이터1에서 오류 발생: {e}") 


try:
  train_y_data_df = pd.read_csv(train_y_data_path, encoding='utf-8',index_col = 0) 
  Y_train = train_y_data_df.drop('datetime', axis = 1)
  print("학습데이터2 읽기 성공!")
except Exception as e: 

  print(f"학습데이터2 에서 오류 발생: {e}") 




##알고리즘 실행에 필요한 함수 및 변수 정의(루틴 2-3)




# 데이터 정규화 함수. 여러 개의 독립변수가 쓰일 때 변수들의 단위가 제각각이면 모델 성능발휘가 어렵기 때문에 동일한 단위(0~1의 값)으로 바꿔줌
# scaler = MinMaxScaler(feature_range = (0,1))


# # train set 스케일링 함수
# def normalize(X,y):
#   X_norm = X.copy()
#   # numpy 배열, 2차원형태 변환, 스케일링
#   for name in X:
#     temp = X[name].to_numpy().reshape(-1,1)
#     X_norm[name] = scaler.fit_transform(temp)


#   temp = y.to_numpy().reshape(-1,1)
#   y_norm = scaler.fit_transform(temp)

#   return X_norm, y_norm


# # X_train, y_train, x_test 스케일링 과정
# X_train_norm, y_train_norm = normalize(X_train,Y_train)
# X_train_norm = X_train_norm.to_numpy()

# def normalize_test(data):
#     data_norm = data.copy()
#     for name in data:
#         temp = data[name].to_numpy().reshape(-1,1)
#         data_norm[name] = scaler.transform(temp)
        
#     return data_norm

# accumulated_df_array = normalize_test(accumulated_df)


# 스케일러 초기화

scalers_X = {name: MinMaxScaler(feature_range=(0, 1)) for name in X_train.columns}
scaler_y = MinMaxScaler(feature_range=(0, 1))


def normalize(X_train,X_test,Y_train):
    # X 데이터 스케일링
    X_train_norm = X_train.copy()
    X_test_norm = X_test.copy()
    Y_train_norm = Y_train.copy()
    for name in X_train.columns:
        X_train_norm[name] = scalers_X[name].fit_transform(X_train_norm[name].values.reshape(-1, 1))
        X_test_norm[name] = scalers_X[name].transform(X_test_norm[name].values.reshape(-1,1))
 
    Y_train_norm = scaler_y.fit_transform(Y_train_norm.values.reshape(-1,1))


    return X_train_norm,X_test_norm,Y_train_norm


# X_train, y_train, x_test 스케일링 과정
X_train_norm,accumulated_df_array,Y_train_norm = normalize(X_train,accumulated_df,Y_train)
# X_train_norm = X_train_norm.to_numpy()




# LSTM input형식인 3D Tensor에 맞게 변환해주는 함수

input_window = 14*7
output_window = 14*1
stride = 14


def input_maker(xdata, ydata, input_window, output_window, stride):
  L_X = xdata.shape[0]
  L_Y = ydata.shape[0]

  # stride에 따른 샘플 개수 계산
  num_samples = min((L_X - input_window) // stride, (L_Y - output_window) // stride) + 1

  # input과 output 초기화
  X = np.zeros([input_window, num_samples, xdata.shape[1]])  # xdata feature 수를 고려하는 shape[1]
  Y = np.zeros([output_window, num_samples])

  for i in np.arange(num_samples):
      start_x = stride * i
      end_x = start_x + input_window
      X[:, i, :] = xdata[start_x:end_x]

      start_y = stride * i
      end_y = start_y + output_window
      Y[:, i] = ydata[start_y:end_y].flatten()

  # LSTM에 맞게 차원 변경
  xdata = X.transpose((1, 0, 2))  # (sample 개수, window 크기, feature 개수) 이 모델에서 x변수는 4개(습도, 전천일사량, 전운량, 온도)
  ydata= Y.transpose((1, 0))    # (sample 개수, window 크기) y변수는 발전량 

  return xdata, ydata



x_train_lstm_input, y_train_lstm_input = input_maker(X_train_norm, Y_train_norm, 98, 14, 14)


# 루틴 2-1에서 만든 accumulated_df_array 변수를 적용하여 모델 입력값으로 활용
def model_input_maker(data, window, stride):
  
    L = data.shape[0]

    # stride에 따른 샘플 개수 계산
    num_samples = ((L - window) // stride) + 1

    # input과 output 초기화
    X = np.zeros([window, num_samples, data.shape[1]])  # xdata feature 수를 고려
  
    

    for i in np.arange(num_samples):
          start_x = stride * i
          end_x = start_x + window
          X[:, i, :] = data[start_x:end_x]


      # LSTM에 맞게 차원 변경 (1,98,4)
    result = X.transpose((1, 0, 2))  # (sample 개수, window 크기, feature 개수)
  

    return result



# 훈련된 모델로 예측값 도출하는 함수
def predict_sunenergy_for_date(date_str, model, data, features): 
       
    print(type(data))

    if data.empty:
        raise ValueError(f"no data available for the date: {date_str}") # 해당 날짜 날씨 데이터가 없는 경우 ValueError 출력
    
    else:
        
        model_input = model_input_maker(accumulated_df_array, 98, 14) # 스케일링 처리된 x_test(accumulated_df_array)가 들어감
        predicted_values = model.predict(model_input) # 필터링한 날씨 데이터를 가지고 모델학습 시켜 예상 발전량 추출 

    return predicted_values.tolist() # 예상 발전량을 리스트형태로 반환




#기상예보데이터에서 하늘상태를 추출해오는 함수.  T시점의 하늘상태 예보데이터가 약 20개 이상 있기 때문에 최빈값을 구함
def get_mode_of_sky(data): 
       
    return int(data['SKY'].mode()[0][-2:]) # mode()[0]을 통해 최빈값을 가져오되, 값에 문자가 포함되어있기때문에 슬라이싱을 통해 숫자값만 반환
 

 
# 예보데이터에서 계절정보를 추출해오는 함수. 
def get_season_final(data):        
    # 해당 날짜에 대한 'season' 컬럼의 값 반환
     
    if not data.empty: 
        return data['season'].iloc[0] # 데이터의 season 컬럼에서 첫 번째 값을 가져올 것. 어차피 같은 날짜의 계절정보이기 때문에 인덱싱에 상관없이 값은 동일.
    else: 
        raise ValueError(f"no season data available for the date") 




# 예측된 태양광발전량(단위 MWh)에 따른 배터리 기본충전량 결정 함수. 
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

 
# 계절정보와 하늘상태를 반영하는 추가충전량 결정 함수. 기본충전량과 추가충전량을 더해 최중충전량 반환
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
    #충전량 범위 제한 로직 
    if season == 'summer'or season == 'spring' or season == 'fall': # 봄,여름,가을인 경우
        final_charge = min(70, max(50, final_charge)) #최고 충전량을 70으로 제한

    else: # 겨울의 경우
        final_charge = min(100, max(50, final_charge)) # 최소 충전량은 50, 최고 충전량은 100
    
    return final_charge




### 모델 생성 및 학습(루틴 3)

def create_combined_model(input_shape): # LSTM 모델 create 함수는 input data의 shape를 가짐 
    # 모델 아키텍처
    model = keras.models.Sequential([
        keras.layers.Input(shape=(98, 4)), # input data shape 먼저 명시
        keras.layers.Bidirectional(keras.layers.LSTM(256, return_sequences=True, name='BiLSTM_0')), # 유닛 수, 해당 layer의 모든 출력을 다음 layer로 넘기는 return_sequences옵션 활성화, 과거와 미래 값을 넘나드며 학습하는 양방향 모델 정의
        keras.layers.Dropout(0.5), # 해당 layer과 다음 layer 이동 시 과적합 방지를 위해 일부 유닛에 대해 비활성화 조치. 0.5는 비활성화 비율(50%)
        keras.layers.LSTM(128, return_sequences=True, name='LSTM_1'),
        keras.layers.LSTM(64, return_sequences=True, name='LSTM_2'),
        keras.layers.LSTM(32, return_sequences=True, name='LSTM_3'),
        keras.layers.Flatten(), # LSTM 레이어는 (배치크기, 시퀀스길이, 특성 수) 3D 텐서를 받지만, FC레이어(Dense)는 [배치크기, 특성 수]의 2차원 배열을 받음. 이에 Flatten() 메서드는 배치크기를 제외한 나머지 두 특성을 하나의 차원으로 평탄화함.
        keras.layers.Dense(64, activation="softsign"), # FC layer(완전결합층), 모든 입력뉴런이 연결되어 있으며 활성화 함수를 거쳐 출력값을 내보냄. softsign 함수는 tanh의 개량버전으로, 그래디언트 소실문제를 보다 강력히 방지
        keras.layers.Dense(14,  activation="relu"),# 마지막 FC layer의 유닛 수는 실제 모델이 필요한 출력값 차원. 해당 모델의 경우 1개일자 14시간대 출력값이므로 14.
        
    ])
    #모델 컴파일링
    model.compile(optimizer='adam', loss='mean_squared_error')  # 최적화 adam, 손실함수 MSE(평균오차제곱)
    return model

# 조기 종료(Early Stopping) 콜백 정의
es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)

# TimeSeriesSplit 정의
tscv = TimeSeriesSplit(n_splits=5)

# 교차 검증 수행
# train data에서 val_dataset을 뽑아내어 총 5회 모델 성능을 검증.
for train_index, val_index in tscv.split(x_train_lstm_input):
    X_train_fold, X_val_fold = x_train_lstm_input[train_index], x_train_lstm_input[val_index]
    y_train_fold, y_val_fold = y_train_lstm_input[train_index], y_train_lstm_input[val_index]

# 모델 생성
RL_model_bi = create_combined_model((X_train_fold.shape[1], 4)) # 인수인 (98,4) 입력해서 모델 생성


# 모델 훈련
#verbose = 0으로 설정하면 cmd 창에서 모델 학습 진행현황 안 뜸
history = RL_model_bi.fit(x_train_lstm_input, y_train_lstm_input, epochs=500, batch_size=64,
                    shuffle=False, validation_data=[X_val_fold, y_val_fold],
                    callbacks=[es], verbose=1)


#모델검증(손실함수 MSE)
RL_model_bi.evaluate(X_val_fold, y_val_fold)


### 훈련된 모델로 예측 발전량 및 충전량 도출(루틴 4)


# 종관API파일과  predict_sunenergy_for_date 함수를 사용해서 T시점의 시간별 태양광발전량 예측.(루틴 4-1)
features_mapping = {'RH' : 'HM', 'GHI' : 'SI', 'CC': 'CA', 'T' : 'TS'}
predicted_values_later_date_str = predict_sunenergy_for_date(current_date_str, RL_model_bi, df_Weather_API, list(features_mapping.values()))

predicted_values_later_date_str_array = np.array(predicted_values_later_date_str) # array 형식으로 변환하여 2차원 배열로 변환할 수 있도록 함
predicted_values_later_date_str_array = predicted_values_later_date_str_array.reshape(-1,1) # 스케일 역변환을 위해 2차원배열로 변환
predicted_values_later_date_str_scaled = scaler_y.inverse_transform(predicted_values_later_date_str_array) # 리버스 스케일링을 통해 실제 단위로 재변환
daily_predicted_value_later_date_str= np.sum(predicted_values_later_date_str_scaled) # 시간별 태양광발전량을 합한 일별 태양광발전량

print(predicted_values_later_date_str_scaled)
print(daily_predicted_value_later_date_str)
 
# "later_date_str"의 하늘 상태 가져오기(루틴 4-2)
sky_value_for_later_date_str = get_mode_of_sky(df_forecast_API) 
print(sky_value_for_later_date_str)
#"later_date_str"의 계절 정보  가져오기(루틴 4-3)
season_for_later_date_str= get_season_final(df_forecast_API) 
season_for_later_date_str
print(season_for_later_date_str)
 
# 최종 배터리 충전량 계산(루틴 4-4)
# 태양광발전량, 하늘상태, 계절정보를 종합해 최종충전량 계산하기
combined_battery_charge = decide_final_charge(daily_predicted_value_later_date_str, sky_value_for_later_date_str, season_for_later_date_str)
print(combined_battery_charge)
 


 
### 알고리즘 로그파일 제작(루틴 5)


result = { 
     
    'datetime' : [current_date_str], # 날짜
    'daily_predicted_value_later_date_str' : [daily_predicted_value_later_date_str], # 일별 태양광발전량
    'sky_value_for_later_date_str' : [sky_value_for_later_date_str], # 하늘상태 예보데이터
    'season_for_later_date_str': [season_for_later_date_str], # 계절정보
    'tommorow_battery_charge': [combined_battery_charge], # 최종충전량
     
    }

print(result)

tommorow_battery_charge_file = pd.DataFrame(result, index = [0]) # result를 데이터프레임화




try:
    
    tommorow_battery_charge_file.to_csv(fr"D:\hyeonseo\Battery_Charging_Algorithm\Battery_Charge_V3\Battery_Charge_V3\Results\{end_date_str}_battery_charging_file.csv", index=False) # csv파일로 저장
    print(fr"D:\hyeonseo\Battery_Charging_Algorithm\Battery_Charge_V3\Battery_Charge_V3\Results\{end_date_str}_battery_charging_file.csv")
except:
    print(f"result 파일 저장 중 에러 발생")# 오류 발생 시 에러메시지 출력
    print(fr"D:\hyeonseo\Battery_Charging_Algorithm\Battery_Charge_V3\Battery_Charge_V3\Results\{end_date_str}_battery_charging_file.csv")


###  배터리 최종 충전량 값(CoC)를 DB에 저장



# #MYSQL연결
try: 
    conn = pymysql.connect(host='175.121.197.205',port =3306, user='bkc_manager', password='bkcManager1234@', db='smart_lora', charset='utf8')
    print(f" DB연동완료")
except:
    print(f"DB연동 중 문제 발생. 유저 이름, 비밀번호 혹은 사용자권한이 설정되어 있는지 재확인하세요")

#커서생성
cur = conn.cursor()


try:
    # device_id들을 저장할 리스트 초기화
    device_ids = []

    # 입력 쿼리 작성 및 실행
    # a.facility_dno  = '35';와 같이 facility_dno는 태양광가로등 식별번호를 의미함. 변경되었을 때 자동적으로 바꿔줘야 함
    sql_equip = """
    select a.*
    from equipment a, equipment_type b
    where a.type_dno = b.dno
    and binary b.service like '%c%'
    and binary b.service like '%d%'
    and a.facility_dno  = '35';
    """
    cur.execute(sql_equip)
    
    # 결과 추출
    result_device = cur.fetchall()
    print(result_device)
    

    # 각 레코드에서 device_id 추출하여 리스트에 저장
    for record in result_device:
        device_id = record[10]  # device_id 추출
        device_ids.append(device_id)
        print(device_ids)
        
    # 각 device_id에 대해 추가 쿼리 실행
    for id in device_ids:
        # 여기에 각 device_id에 대한 쿼리를 작성하고 실행
        #입력 쿼리 작성
        try:    

            sql_setting = """update setting
            set solar_coc = %s, updated = %s
            where r_device_id = %s;"""

        # 쿼리 실행
            cur.execute(sql_setting, (combined_battery_charge,current_datetime, id))
        # 입력값 저장
            conn.commit()
            print(f"DB에  {id}의 입력값 저장 완료. 저장된 시간 {current_datetime}, 업데이트 된 행 수: {cur.rowcount}")
    


        except:
            print(f"setting쿼리에 오류 발생. 문법 및 기타 사항 재확인하세요.")

except:
            print(f"장비 ID쿼리에 오류 발생. 문법 및 기타 사항 재확인하세요.")


# MYSQL 연결 종료
conn.close()