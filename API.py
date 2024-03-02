# -*- coding: utf-8 -*-   # 파일 인코딩 형식을 파일 첫머리에 명시해주어 인코딩 오류를 방지함

import pandas as pd
from haversine import haversine # pip install 필요
import requests
import json
import datetime
import os
import sys



print("API 파일 1 스크립트 시작")



print(sys.argv) # 커맨드라인을 통해 py파일 실행을 위한 인자들이 제대로 전달되었는지 출력하여 확인해보기 위한 print문


### 입력어 확인 및 설정 절차(루틴 1)

## 파일 자동 실행을 위해 필요한 인자들이 커맨드라인(sys.argv)으로 정상적으로 전달되었는지 확인
## 파일 이름(Auto_Script.py), 시작날짜(현재날짜 기준 7개일자 전), 종료날짜(현재날짜),위도,경도, 카카오API키, 기상청API키

# 파일 실행 인자 7개가 제대로 전달되었는지 확인하는 부분(루틴 1-1)
if len(sys.argv) >=7:

  
    start_date_input = sys.argv[1] 
    start_date_input = start_date_input.replace('-', '') 
    end_date_input = sys.argv[2] 
    end_date_input = end_date_input.replace('-', '') 
    latitude_input = sys.argv[3] 
    longitude_input = sys.argv[4]
    first_api_input = sys.argv[5] 
    second_api_input = sys.argv[6] 



# 만약 커맨드라인의 인자가 7개 이하라면, 즉 인자가 누락되었거나 잘못된 형식으로 입력된 상황에는 수동입력으로 전환 (서브루틴1)
else:  
    print("커맨드라인 인자의 형식을 다시 확인하세요. 시작 날짜와 종료 날짜를 YYYYMMDD 형식으로 입력하세요.위도와 경도는 소숫점 그대로 입력하세요. 또는 api 키 값을 다시 확인하세요 ") 
    try:  
        
        start_date_input = input("예측을 수행할 시작 날짜를 YYYYMMDD 형식으로 입력하세요(ex. 2월 22일의 충전량을 예측하려면 YYYY0215):")
        start_date_input = start_date_input.replace('-', '') 
       
        end_date_input = input("예측을 수행할 종료 날짜를 YYYYMMDD 형식으로 입력하세요(ex. 2월 22일의 충전량을 예측하려면 YYYY0221):") 
        end_date_input = end_date_input.replace('-', '')  


        latitude_input = float(input("위도를 입력하세요: "))
        longitude_input = float(input("경도를 입력하세요: ")) 
        if not (-90 <= latitude_input <= 90) or not (-180 <= longitude_input <= 180): 
            print("잘못된 위도 또는 경도 값입니다.") 
         
        first_api_input = input("KAKAO REST API KEY 값을 입력하세요. : ")
        first_api_input =  first_api_input.replace(" ", "").replace(".", "")
        second_api_input = input("기상청 API KEY 값을 입력하세요. : ")
        second_api_input =  second_api_input.replace(" ", "").replace(".", "")
       

    # 서브루틴 1 실패의 경우, 스크립트 종료     
    except ValueError as e: 
        print(f"유효한 입력값을 입력해주세요. {e}") 
        exit() 
        sys.exit() 
# 시작날짜와 종료날짜 전처리(루틴 1-2)    
try: 
    start_date = datetime.datetime.strptime(start_date_input, '%Y%m%d') 
    end_date = datetime.datetime.strptime(end_date_input, '%Y%m%d')  
    latitude_input = float(latitude_input) 
    longitude_input = float(longitude_input) 
    first_api_input =  first_api_input.replace(" ", "").replace(".", "") 
    second_api_input =  second_api_input.replace(" ", "").replace(".", "")


# 입력값 오류에 따른 스크립트 종료(루틴 1-2-1) 
except ValueError as e:
    print(f"날짜 혹은 위경도 값 형식에 오류가 있습니다. 날짜의 경우 YYYYMMDD 형식으로 입력하세요. 위경도의 경우 소숫점까지 정확히 입력하세요. 또는 api 키 값을 다시 확인하세요 {e} ") 
    sys.exit() 
    

# 입력값 전처리 이후, 알고리즘 내에서 쓰이는 날짜 변수 정의(루틴1-3)
# try 블록과 함께 쓰이는 else문의 경우, try-except에서 예외가 발생하지 않았을 때 바로 실행됨
else:
    current_date = datetime.date.today()
    # end_date = current_date 
    current_date_str = current_date.strftime('%Y%m%d') # datetime객체를 YYYYMMDD형식의 문자열로 바꿔 시작날짜(오늘날짜)에 저장
    end_date_str = end_date.strftime('%Y%m%d') # datetime객체를 YYYYMMDD형식의 문자열로 바꿔 시작날짜(오늘날짜)에 저장
    previous_six_date = end_date - datetime.timedelta(days=6) 
    # start_date = previous_six_date 
    previous_six_date_str = previous_six_date.strftime('%Y%m%d') # datetime객체를 YYYYMMDD형식의 문자열로 바꿈
    previous9_date = end_date - datetime.timedelta(days=9) #호출 URL에서 tm1_value에 지정될 값. 예보데이터는 3~10일 이후의 날씨를 예보하기 때문에 9일 전 예보부터 확인
    previous9_date_str = previous9_date.strftime('%Y%m%d') # datetime객체를 YYYYMMDD형식의 문자열로 바꿈
    later_date = end_date + datetime.timedelta(days=1) # 호출 url에서 tm3_value에 저장될 값. T시점의 예보를 읽어내기 위함
    later_date_str = later_date.strftime('%Y%m%d')# datetime객체를 YYYYMMDD형식의 문자열로 바꿈
    later3_date = end_date + datetime.timedelta(days=3) # 호출 url에서 tm4_value에 저장될 값. T시점 전체 시간대의 예보를 읽어내기 위함
    later3_date_str = later3_date.strftime('%Y%m%d')# datetime객체를 YYYYMMDD형식의 문자열로 바꿈 
  



print(start_date, end_date, latitude_input, longitude_input, first_api_input, second_api_input ) 

print("커맨드라인 인자 확인 완료") # 전처리가 완전히 수행되면 확인 완료 메시지 출력

base_url = f"https://dapi.kakao.com/v2/local/geo/coord2regioncode.json?x={longitude_input}&y={latitude_input}&input_coord=WGS84" 

print(base_url) #디버깅용


headers = {'Authorization' : 'KakaoAK '+first_api_input} 

api_req = requests.get(base_url,headers=headers) 
print(api_req) #디버깅용. 200 


input_data = json.loads(api_req.text) 



try: # API가 성공적으로 불러와진 경우
 input_lat = input_data['documents'][0]["y"] # 위도값 추출
 input_lon = input_data['documents'][0]["x"] # 경도값 추출
 input_region_1depth_name = input_data['documents'][0]['region_1depth_name'] # 도 단위 행정구역 추출
 input_region_2depth_name = input_data['documents'][0]['region_2depth_name'] # 시군구 단위 행정구역 추출
except:
  print(f"API에서 좌표정보 또는 행정구역 추출 실패") # 오류가 발생하는 등 API 데이터가 제대로 불러와지지 않았다면 오류메시지 출력
  sys.exit() # 스크립트 종료. 이러한 경우에는 다시 시작하여 커맨드라인을 제대로 입력해주어야 함

print(input_region_1depth_name) # 도 단위 행정구역 확인. 디버깅
print(input_region_2depth_name) # 시군구 단위 행정구역 확인. 디버깅


extracted_location = (input_lat,input_lon) # 추출한 위경도값을 튜플형태로 변수에 저장. 이 변수를 토대로 주어진 위치와 가장 가까운 종관 관측소 ID를 구해야 함.

API_LOCATION = pd.read_csv(r"C:\Users\user1\Desktop\Battery_Charging_Algorithm\API_LOCATION_V2.csv" )


# 입력값 기준 최단거리의 종관기상 관측소 얻는 함수 
def find_nearest_observatory(extracted_location, API_LOCATION): # API를 통해 추출한 위경도 값과 전국 종관 관측소의 위경도 값을 인자로 활용하는 find_nearest_observatory 함수 선언. 좌표 간 거리를 계산하고 최단거리에 있는 종관관측소 ID를 구함.



    min_distance = float('inf') # 최단거리를 저장할 변수 선언. 초기값은 inf로 저장하는 이유는 반복문이 다 실행되는 동안 nearest_location 값이 지정되면 안 되기 때문임. ex. 값이 총 5개인데 3번째 값에서 최단거리가 구해지면 나머지 2개 값은 비교가 불가능하니까, 임의로 inf값을 최소거리로 지정해서 모든 값에 대한 비교가 이뤄질 수 있도록 하는것임.
    nearest_location = None # 가장 가까운 종관 관측소 저장할 변수 선언. 초기값은 None

    # API_LOCATION에서 관측소 정보 추출 및 계산
    for index, row in API_LOCATION.iterrows(): # 관측소 위경도 파일에서 인덱스를 순회하는 반복조건
        obs_latitude = row['LAT']  # API_LOCATION에서 관측소의 위도
        obs_longitude = row['LON']  # API_LOCATION에서 관측소의 경도
        endpoint = (obs_latitude, obs_longitude)  # 관측소 좌표 튜플 생성

        # Haversine 메서드를 사용하여 거리 계산
        try:
         distance = haversine(extracted_location, endpoint, unit = 'km') # haversine 메서드를 사용하여 입력좌표값과 종관관측소 좌표 간 거리 구하기
        except:
         print(f"최단거리 계산 중 에러 발생") # haversine 메서드가 제대로 실행되지 않았을 때의 오류일 가능성 높음
         # sys.exit() # 스크립트 종료. 이러한 경우에는 다시 시작하여 커맨드라인을 제대로 입력해주어야 함/ 이거 활성화시키면 find_nearest_observatory 함수가 실행되지 않음. 왤까?


        if distance < min_distance: # 반복을 수행하여 최단거리를 구하는 과정. 만약 최단거리가 등장하면
            min_distance = distance # min_distance에  최단거리 값을 저장
            nearest_location = row['STN_2']  # 최단거리를 갖는 endpoint가 있는 행에서 STN_2 컬럼을 참조해 관측소 ID를 nearest_location에 저장

    return min_distance, nearest_location # 최단거리


# 가장 가까운 종관기상관측소 ID 찾기
nearest_location = find_nearest_observatory(extracted_location, API_LOCATION) #함수를 실행하여 최단거리의 종관관측소 ID 찾기

print(f" 입력값 기준 최단거리의종관관측소 : {nearest_location}")

def download_file(file_url, save_path): # 종관 데이터 API 파일 다운로드를 위한 download_file함수. 기상청 API허브에서 제공하는 함수임.
    try:
        # 디렉토리 생성
        if not os.path.exists(os.path.dirname(save_path)): # os.path.dirname메서드로 디렉토리 부분을 추출한 후 os.path.exists()메서드로 디렉토리 유무를 확인
            os.makedirs(os.path.dirname(save_path)) # os.makedirs 메서드로 디렉토리 생성

        # 파일 다운로드
        with open(save_path, 'wb') as f:
            response = requests.get(file_url)
            if response.status_code == 200:
                f.write(response.content)
            else:
                print(f"다운로드 실패. HTTP 상태 코드: {response.status_code}")
    except Exception as e:
        print(f"오류 발생: {e}")
        sys.exit() # 스크립트 종료. 이러한 경우에는 다시 시작하여 커맨드라인을 제대로 입력해주어야 함






# URL과 저장 경로 동적으로 생성하기
base_url_weather = "https://apihub.kma.go.kr/api/typ01/url/kma_sfctm3.php" # url의 불변 포맷
# auth_key = "ghBQh0IXRimQUIdCFzYp4w" # url의 가변 포맷- 인증키 부분


# 반복문 삽입. tm1_value, tm2_value가 각각 7개씩 총 14개 나와야 함. url_weather이 총 7개 나오고, 7번 호출이 되어야함


# current_date = datetime.date.today()
# current_date = end_date # 현재날짜를 시작날짜로 지정. 이는 시작날짜부터 종료날짜까지 반복하기 위함.
# previous_six_date = current_date - datetime.timedelta(days=6) 
# previous_six_date = start_date


while start_date <= end_date:
    start_date_str = start_date.strftime('%Y%m%d')
    # RL의 tm1과 tm2 값을 설정
    tm1_value = start_date_str + "0600" # url의 가변 포맷 - 관측 시간 지정
    tm2_value = start_date_str + "2100" # url의 가변 포맷 - 관측 시간 지정

    # 완성된 URL
    url_weather = f"{base_url_weather}?tm1={tm1_value}&tm2={tm2_value}&stn={nearest_location[1]}&help=0&authKey={second_api_input}" # url_weather 조합


    print("종관URL 완성")
    print(url_weather)



    # 저장 경로

    weather_save_file_path = fr"C:\Users\user1\Desktop\Battery_Charging_Algorithm\API\Weather_{start_date_str}_file_pre.csv" # 종관, 예보 날씨 데이터 파일들은 모두 현재날짜)_csv의 이름으로 저장됨. 
   

    try:
       # 파일 다운로드 함수 호출
        download_file(url_weather, weather_save_file_path)
    except Exception as e:
        print(f"파일 다운로드 중 오류 발생: {e}")
        sys.exit() # 스크립트 종료. 이러한 경우에는 다시 시작하여 커맨드라인을 제대로 입력해주어야 함



    response = requests.get(url_weather)

    # 응답 상태 코드 확인
    print(response.status_code)

    # 응답 내용 출력
    print(response.text)



    print(f"종관API 호출 완료")

    start_date += datetime.timedelta(days=1) 





# 예보데이터관측소 얻는 함수


API_LOCATION_FORECAST = pd.read_csv(r"C:\Users\user1\Desktop\Battery_Charging_Algorithm\API_FORECAST_REGION_V2.csv") # 예보데이터 관측소 정보를 담고 있는 파일

list_east = ['강릉시', '삼척시', '태백시', '속초시', '양양군', '동해시', '고성군'] #  강원영동 예보관측소.  input_region_2depth_name에 리스트 내 포함되는 문자열이 있으면 '강원영동' 관측소를 반환함.
list_west = ['철원군', '화천군', '양구군', '인제군', '춘천시', '홍천군', '횡성군', '원주시', '평창군', '영월군', '정선군'] # 강원영동 예보관측소.  input_region_2depth_name에 리스트 내 포함되는 문자열이 있으면 '강원영서' 관측소를 반환함.

def find_nearest_obesrvatory_forecast_first(depth1_name, depth2_name): # 위경도 값을 통해 행정구역 반환하는 함수. 반환된 행정구역값을 기상예보데이터 관측소 파일과 비교해 일치하는 값을 지닌 관측소 reg_value를 반환
    if depth2_name in list_east:# depth2_name은 시군구단위 행정구역, depth2_name 변수와 list_east 내 문자열과 일치하는 것이 있다면 
        return '강원영동' # 강원영동을 반환함. 이 경우 '강원영동' 기상예보관측소 reg_value가 반환됨
    elif depth2_name in list_west: # depth2_name 변수와 list_west 내 문자열과 일치하는 것이 있다면
        return '강원영서' # 강원영서 반환
    elif '서울' in depth1_name or '경기' in depth1_name or '인천' in depth1_name: # depth1_name에 서울이나 경기나 인천이 포함되어있다면
        return '서울.인천.경기' # 서울.인천.경기 반환
    
    elif '제주특별자치도' in depth1_name: # 제주특별자치도의 경우
        return '제주도' # 제주도 기상예보관측소 reg_value 반환

    else: # 그 외의 경우
        return depth1_name # depth1_name 반환

extracted_location_forecast = find_nearest_obesrvatory_forecast_first(input_region_1depth_name, input_region_2depth_name) #함수 실행 결과를 extracted_location_forecast에 저장.  아래 오는 find_nearest_obesrvatory_forecast_second 함수에서 입력값으로 활용

def find_nearest_obesrvatory_forecast_second(extracted_location_forecast, location): # 위경도 값을 통해 반환된 행정구역값과 기상예보관측소 정보 데이터를 비교해 해당 기상예보관측소 reg_value를 반환하는 함수
    matched_location = location[location['REG_NAME'] == extracted_location_forecast] # 기상예보데이터에서 REG_NAME이 위에서 구한extracted_location_forecast 이름과 같으면 해당 관측소의 reg_name 반환
    if not matched_location.empty: #제대로 불러오기 성공했다면
        return matched_location.iloc[0]['REG_ID'] #REG_ID 반환
    else:
        print(f"API_LOCATION_FORECAST에서 {extracted_location_forecast} 값을 찾는데 실패") # 불러오기 실패한 경우 
        return None # none값 반환
        sys.exit() # 스크립트 종료. 이러한 경우에는 다시 시작하여 커맨드라인을 제대로 입력해주어야 함

nearest_location_forecast = find_nearest_obesrvatory_forecast_second(extracted_location_forecast, API_LOCATION_FORECAST) # 행정구역 토대로 입력값 기준 최단거리의 기상예보관측소 추출
print(nearest_location_forecast)





def download_file(file_url, save_path): #API 받아와서 파일 다운로드 하는 함수
    with open(save_path, 'wb') as f: #API 받아와서 파일 다운로드 하는 함수
        response = requests.get(file_url) #API 받아와서 파일 다운로드 하는 함수
        f.write(response.content) #API 받아와서 파일 다운로드 하는 함수

# URL과 저장 경로 동적으로 생성하기

base_url_forecast = "https://apihub.kma.go.kr/api/typ01/url/fct_afs_wl.php" # url의 불변 포맷
# auth_key = "ghBQh0IXRimQUIdCFzYp4w" # url의 가변 포맷, 인증키 부분

reg_value = nearest_location_forecast # url의 가변 포맷 - 관측소 id 부분

# URL의 tm1과 tm2 값을 설정
tm1_value_forecast = previous9_date_str + "0600" # url의 가변 포맷 - 관측 시간 지정
tm2_value_forecast = end_date_str + "1800" # url의 가변 포맷 - 관측 시간 지정
tm3_value_forecast = later_date_str + "0000" # url의 가변 포맷 - 관측 시간 지정
tm4_value_forecast = later3_date_str + "0000" # url의 가변 포맷 - 관측 시간 지정




# 완성된 URL

url_forecast = f"{base_url_forecast}?reg={reg_value}&tmfc1={tm1_value_forecast}&tmfc2={tm2_value_forecast}&tmef1={tm3_value_forecast}&tmef2={tm4_value_forecast}&mode=0&disp=0&help=1&authKey={second_api_input}"  # 호출 URL 조합 




print("URL 확인완료")
print(url_forecast)



# 저장 경로
forecast_save_file_path = fr"C:\Users\user1\Desktop\Battery_Charging_Algorithm\API\forecast_{end_date_str}_pre.csv" # 종관, 예보 날씨 데이터 파일들은 모두 현재날짜)_csv의 이름으로 저장됨. 



try:
    # 파일 다운로드 함수 호출
    download_file(url_forecast, forecast_save_file_path)  # 조합된 url과 지정된 파일경로를 활용해 download_file 함수 돌려서 예보데이터 api 파일 저장

except Exception as e:
        print(f"파일 다운로드 중 오류 발생: {e}")
        sys.exit() # 스크립트 종료. 이러한 경우에는 다시 시작하여 커맨드라인을 제대로 입력해주어야 함 


response = requests.get(url_forecast) # 완성된  URL로 request.get 메서드를 실행해 api파일을 가져옴

# 응답 상태 코드 확인
print(response.status_code) # 200이 정상. 

# 응답 내용 출력
print(response.text)


print(f"예보API 호출 완료")

print("API파일 1 스크립트 종료")


# os.system("pause")


