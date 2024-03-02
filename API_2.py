# -*- coding: utf-8 -*-   # 파일 인코딩 형식을 파일 첫머리에 명시해주어 인코딩 오류를 방지함

import pandas as pd
from haversine import haversine # pip install 필요
import requests
import json
import datetime
import os
import sys

print(" API 파일 2 스크립트 시작")
    
if len(sys.argv) >=7:  
    start_date_input = sys.argv[1] # 시작날짜는 리스트의 두 번째 요소이므로 [1] 인덱싱을 통해 변수에 저장
    start_date_input = start_date_input.replace('-', '') # 입력된 날짜의 형식을 전처리 하는 과정. 하이픈이 있다면 이를 제거하라는 명령
    print(start_date_input) # 시작날짜가 올바른 형식으로 바뀌었는지 확인하기 위한 print문
    end_date_input = sys.argv[2] # 종료날짜는 리스트의 세 번째 요소이므로 [2] 인덱싱을 통해 변수에 저장
    end_date_input = end_date_input.replace('-', '') # 입력된 날짜의 형식을 전처리 하는 과정. 하이픈이 있다면 이를 제거하라는 명령
    print(end_date_input) # 종료날짜가  올바른 형식으로 바뀌었는지 확인하기 위한 print문




else: # 만약 커맨드라인의 인자가 7개 이하라면, 즉 인자가 누락되었거나 잘못된 형식으로 입력된 상황이라면 => 일일이 인자를 입력하여 진행해야 함. 서브루틴: 수동입력
    print("커맨드라인 인자의 형식을 다시 확인하세요. 시작 날짜와 종료 날짜를 YYYYMMDD 형식으로 입력하세요.위도와 경도는 소숫점까지 입력하세요. ") # 입력형식을 다시 확인해보라는 메시지를 출력하고
    
    
    start_date_input = input("예측을 수행할 시작 날짜를 YYYYMMDD 형식으로 입력하세요(ex. 2월 22일~2월 25일의 충전량을 예측하려면 YYYY0221):") # 시작날짜를 YYYMMDD로 입력하는 input 부분. 예측하고자 하는 날 하루 전 날짜를 입력(T-1시점)
    start_date_input = start_date_input.replace('-', '') # 입력값에 하이픈이 있다면 제거하라는 명령. 하이픈이 있는 상태로 입력되면 호출 URL에 사용할 수 없음
       
    end_date_input = input("예측을 수행할 종료 날짜를 YYYYMMDD 형식으로 입력하세요(ex. 2월 22일~2월 25일의 충전량을 예측하려면 YYYY0224):") # 종료날짜를 YYYMMDD로 입력하는 input 부분. 예측하고자 하는 날 하루 전 날짜를 입력(T-1시점)
    end_date_input = end_date_input.replace('-', '')  # 입력값에 하이픈이 있다면 제거하라는 명령. 하이픈이 있는 상태로 입력되면 호출 URL에 사용할 수 없음     
    
    
   

try: # 여기서의 try - except 블록은 위의 if len(sys.argv) >=7 조건이 만족된 경우 바로 진행되는 시퀀스임. else문 내부 코드와 비슷한 이유는, if와 else 두 경우 모두 전처리를 수행해야 하기 때문임.
    start_date = datetime.datetime.strptime(start_date_input, '%Y%m%d') # 시작날짜를 호출URL에 사용할 수 있는 형식으로 전처리. 이를 위해 문자열을 datetime객체로 변환하는 strptime 메서드 활용 
    end_date = datetime.datetime.strptime(end_date_input, '%Y%m%d')  # 종료날짜를  호출URL에 사용할 수 있는 형식으로 전처리. 이를 위해 문자열을 datetime객체로 변환하는 strptime 메서드 활용


except ValueError:# 커맨드라인에서 인자가 잘못 입력되었거나, 전처리가 올바르게 수행되지 않으면
    print("날짜 혹은 위경도 값 형식에 오류가 있습니다. 날짜의 경우 YYYYMMDD 형식으로 입력하세요. 위경도의 경우 소숫점까지 정확히 입력하세요 ") # 경고 메시지 출력하고
    sys.exit() # 스크립트 종료. 이러한 경우에는 다시 시작하여 커맨드라인을 제대로 입력해주어야 함
   

start_date = datetime.datetime.strptime(start_date_input, '%Y%m%d') # 시작날짜를 호출URL에 사용할 수 있는 형식으로 전처리. 이를 위해 문자열을 datetime객체로 변환하는 strptime 메서드 활용 
end_date = datetime.datetime.strptime(end_date_input, '%Y%m%d')  # 종료날짜를  호출URL에 사용할 수 있는 형식으로 전처리. 이를 위해 문자열을 datetime객체로 변환하는 strptime 메서드 활용


# 입력받은 날짜를 기반으로 현재 날짜 및 이전 날짜 계산

current_date = datetime.date.today()
# current_date = end_date # 현재날짜를 시작날짜로 지정. 이는 시작날짜부터 종료날짜까지 반복하기 위함.
current_date_str = current_date.strftime('%Y%m%d') # datetime객체를 YYYYMMDD형식의 문자열로 바꿔 시작날짜(오늘날짜)에 저장
end_date_str = end_date.strftime('%Y%m%d') # datetime객체를 YYYYMMDD형식의 문자열로 바꿔 시작날짜(오늘날짜)에 저장
previous_six_date = end_date - datetime.timedelta(days=6) 
# previous_six_date = start_date
previous_six_date_str = previous_six_date.strftime('%Y%m%d') # datetime객체를 YYYYMMDD형식의 문자열로 바꿈
previous9_date = end_date - datetime.timedelta(days=9) #호출 URL에서 tm1_value에 지정될 값. 예보데이터는 3~10일 이후의 날씨를 예보하기 때문에 9일 전 예보부터 확인
previous9_date_str = previous9_date.strftime('%Y%m%d') # datetime객체를 YYYYMMDD형식의 문자열로 바꿈
later_date = end_date + datetime.timedelta(days=1) # 호출 url에서 tm3_value에 저장될 값. T시점의 예보를 읽어내기 위함
later_date_str = later_date.strftime('%Y%m%d')# datetime객체를 YYYYMMDD형식의 문자열로 바꿈
later3_date = end_date + datetime.timedelta(days=3) # 호출 url에서 tm4_value에 저장될 값. T시점 전체 시간대의 예보를 읽어내기 위함
later3_date_str = later3_date.strftime('%Y%m%d')# datetime객체를 YYYYMMDD형식의 문자열로 바꿈 




def preprocess_weather_data(input_path, output_path): # 전처리하는 함수 정의, save_file_path를 통해 불러와서 전처리한 후 ouput_path로 내보낼 계획
    df_weather_API_pre = pd.read_csv(input_path, sep='\s+', comment='#', encoding='cp949') # input_path를 활용해 파일을 불러오되, 공백은 구분자로 인식하고 #부분을 주석처리된 것으로 간주함
    
   
    with open(input_path, "r", encoding="cp949") as file: # 읽기모드로 열기
        lines = file.readlines() # 파일 내 요소들을 리스트 내 요소로 반환. 인덱싱 등 요소에 접근하여 전처리하기 쉽게 바꿔주는 readlines()메서드 활용

    column_index = [i for i, line in enumerate(lines) if line.startswith("# YYMMDDHHMI")][0] # 위에서 만든 list 내 line들을 순회하면서 # YYMMDDHHMI로 시작하는 부분의 행과 인덱스를 반환하는 enumerate에서, 인덱스만 반환(i for i, line )
    column_names = lines[column_index].split() # 공백 기준으로 split해서 컬럼명으로 반환한 후 column_names에 저장
    column_names.remove('#') # 처음에 만들어진 컬럼 길이와 사용자 정의한 컬럼 길이가 맞지 않음. 이에 사용자 정의 측 컬럼 길이를 임의적으로 줄여서 길이 맞춰주기
    df_weather_API_pre.columns = column_names # 사용자 정의 메서드로 컬럼명 설정

    df_weather_API = df_weather_API_pre[['YYMMDDHHMI', 'STN', 'WS', 'HM', 'CA', 'SS', 'SI', 'TS']] # 사용할 컬럼만 추려서 종관데이터 파일 만들기
    df_weather_API['YYMMDDHHMI'] = df_weather_API['YYMMDDHHMI'].astype(str) # 날짜 및 시간정보를 가지는 YYMMDDHHM 열을 기준으로 필터링하기 위해, 먼저 문자열로 바꿔줌
    df_weather_API['datetime'] = pd.to_datetime(df_weather_API['YYMMDDHHMI']).dt.strftime('%Y-%m-%d %H:00') # datetime64ns 형식으로 형변환 해준 후, strftime메서드로 연-월-일- 시:00 형식으로 포맷팅해줌.
    df_weather_API['datetime'] = pd.to_datetime(df_weather_API['datetime']) # strftime 메서드를 활용하면 문자열로 다시 바뀌므로, datetime으로 재변환
    df_weather_API = df_weather_API[(df_weather_API['datetime'].dt.hour >= 7) & (df_weather_API['datetime'].dt.hour <= 20)] # dt.hour으로 연-월-일-시:00 에서 '시' 부분만 뽑아낸 후, 이를 기준으로 7시와 20시 사이만 필터링
    
    col = ['WS','HM','CA','SS','SI','TS']
    def no_outlier(df_weather_API, col):
        df = df_weather_API.copy()
    
        for k in col:
           Q1 = df[k].quantile(0.25)
           Q3 = df[k].quantile(0.75)
           IQR = Q3 - Q1
           rev_value = 1.5
           outliers = df[k][(df[k] > Q3 + (rev_value  * IQR)) | (df[k] < Q1 + (rev_value * IQR))]
           df[k] = df[k].replace(outliers,pd.NA)
           df[k] = df[k].interpolate(method='linear', limit_direction='both')
    
        return df_weather_API



    df_weather_API.to_csv(output_path, index=False) # 전처리 완료. output_path로 파일 내보내기
    return df_weather_API



# current_date = datetime.date.today()
# current_date = end_date # 현재날짜를 시작날짜로 지정. 이는 시작날짜부터 종료날짜까지 반복하기 위함.
# previous_six_date = current_date - datetime.timedelta(days=6) 
# previous_six_date = start_date

print(start_date)
print(end_date)

while start_date <= end_date:

    start_date_str = start_date.strftime('%Y%m%d')

    # 전처리 함수 호출

    input_path_weather = fr"C:\Users\user1\Desktop\Battery_Charging_Algorithm\API\Weather_{start_date_str}_file_pre.csv" # 전처리 이전의  종관기상데이터 경로
    output_path_weather = fr"C:\Users\user1\Desktop\Battery_Charging_Algorithm\API\Weather_{start_date_str}_file.csv"  # 전처리 완료한 종관기상데이터 경로


    try:
     df_weather_API= preprocess_weather_data(input_path_weather , output_path_weather) # 전처리 완료된 종관API데이터 파일을 df_weather_API에 저장
     print(f"종관api전처리 완료")# 전처리 완료 메시지 출력                                         
                                         
                                         
    except:
     print(f"종관api전처리함수 처리 중 문제 발생") # 전처리함수 처리 중 문제 발생. API_pre 파일을 확인해볼 것. 컬럼명 추출이 제대로 되어있지 않아 오류가 났을 가능성이 높음. 컬럼명 추출이 제대로 되지않은 이유는 with_open메서드 부분에서 sep, comment 옵션이 잘못되어 있을 가능성
     sys.exit() # 스크립트 종료. 이러한 경우에는 다시 시작하여 커맨드라인을 제대로 입력해주어야 함

    start_date += datetime.timedelta(days=1) 

#여기까지



#예보api 전처리과정



def preprocess_forecast_data(input_path, output_path): # 파일 전처리 하는 함수 정의. 불러올 때 파일경로, 내보낼 때 파일경로를 인자로 사용함
 
    df_forecast_API = pd.read_csv(input_path, sep='\s+', comment='#', encoding='cp949') # 불러올 떄 파일, 공백을 구분자로 인식, #를 주석처리하고, cp949 인코딩 방식 사용



# 컬럼명을 추출
    with open(input_path, "r", encoding="cp949") as file: # 먼저 파일을 읽기 전용모드로 열기
     lines = file.readlines() # 파일 내 요소들을 리스트 내 요소로 반환. 인덱싱 등 요소에 접근하여 전처리하기 쉽게 바꿔주는 readlines()메서드 활용


    # "reg_id"로 시작하는 라인을 찾아 컬럼명으로 사용
    column_index = [i for i, line in enumerate(lines) if line.startswith("# REG_ID")][0] # 위에서 만든 list 내 line들을 순회하면서 #reg_id로 시작하는 부분의 행과 인덱스를 반환하는 enumerate에서, 행만 반환(0)
    column_names = lines[column_index].split() # 공백 기준으로 split해서 컬럼명으로 반환한 후 column_names에 저장


    # '#' 컬럼 제거
    column_names.remove('#') # 처음에 만들어진 컬럼 길이와 사용자 정의한 컬럼 길이가 맞지 않음. 이에 사용자 정의 측 컬럼 길이를 임의적으로 줄여서 길이 맞춰주기
    
   

    # 데이터 프레임의 컬럼명을 설정
    df_forecast_API.columns = column_names # 사용자 정의 메서드로 컬럼명 설정

    print(df_forecast_API.columns)


    df_forecast_API['TM_EF'] = df_forecast_API['TM_EF'].astype(str) # TM_EF 컬럼을 문자열로 변환. 이는 TM_EF (예보발효시간)의 값으로 필터링을 하기 위함.

    df_forecast_API['datetime'] = pd.to_datetime(df_forecast_API['TM_EF']) #  TM_EF 컬럼을 datetime형식으로 변환
   


    def season_searching(month): # 계절정보 정의하는 함수.

      if 3<= month <=5:
        return 'spring'
      elif 6<= month <= 8:
        return 'summer'
      elif 9<= month <=11:
        return 'fall'
      else:
        return 'winter'

  
    df_forecast_API['season'] = df_forecast_API['datetime'].dt.month.apply(season_searching) # apply 메서드를 통해 season_searching 함수 적용

    df_forecast_API.to_csv(output_path, index = False) # 전처리 완료. 파일 내보내기.


input_path_forecast = fr"C:\Users\user1\Desktop\Battery_Charging_Algorithm\API\forecast_{end_date_str}_pre.csv" # 전처리 이전 예보API데이터 경로
output_path_forecast = fr"C:\Users\user1\Desktop\Battery_Charging_Algorithm\API\forecast_{end_date_str}.csv" #전처리이후 예보API데이터 경로


try:
 df_forecast_API = preprocess_forecast_data(input_path_forecast, output_path_forecast) # 전처리완료한 예보api파일을 df_forecast_API에 저장
 print(f"예보api전처리완료")
except:
 print(f"예보api전처리함수 처리 중 문제 발생")
 sys.exit() # 스크립트 종료. 이러한 경우에는 다시 시작하여 커맨드라인을 제대로 입력해주어야 함

print("API 파일 2 스크립트 종료")

# os.system("pause")

