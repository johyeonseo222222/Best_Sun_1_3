# -*- coding: cp949 -*-   # 파일 인코딩 형식을 파일 첫머리에 명시해주어 인코딩 오류를 방지함

import os 
import sys
import datetime
import schedule
import time
import pymysql



current_date = datetime.date.today() # datetime.date.today()메서드로 오늘 날짜 반환
previous_six_date = current_date - datetime.timedelta(days=6)

# 지정된 날짜를 업데이트하는 함수
def update_date(input_date): # input_date를 인자로 활용하는 update_date함수 정의. 사용자의 날짜입력값을 올바른 형태의 날짜형식으로 변환하여 API 호출 URL구성 시 활용하도록 '전처리'해주는 함수
    if input_date: # 날짜값이 입력되면
        date_object = datetime.datetime.strptime(input_date, '%Y%m%d') # strptime 메서드를 활용해 문자열을 datetime 객체로 변환하고 YYYYMMDD형식으로 맞춰줌
        return date_object #전처리 완료된 날짜값을 반환
    else: # 만약 입력값이 존재하지 않으면
        # input_date가 None인 경우 현재 날짜를 반환
        current_date = datetime.date.today() # datetime.date.today()메서드로 오늘 날짜 반환
        previous_six_date = current_date - datetime.timedelta(days=6)

        return current_date,previous_six_date # strftime 메서드를 활용해 연월일형식의 문자열로 전환

# py 파일을 실행하는 함수
def execute_py(file_path, start_date, end_date, latitude_input, longitude_input, first_api_input, second_api_input): # py파일 실행함수, 입력값을 요구하는 여러 개의 파일을 자동으로 실행하도록 하는 함수. 파일경로와 날짜,위경도 등 입력값들을 인자로 받음.
    try:
        os.system(f"python {file_path} {start_date} {end_date} {latitude_input} {longitude_input} {first_api_input} {second_api_input}") # 파일 실행 스크립트 포맷.  운영체제에서 문자열로 구성된 명령을 실행하는 내장함수 os.system() 메서드 내에 py파일 실행하는 스크립트 포맷을 f-string으로 전달.
    except Exception as e:
        print(f"작업 실행 중 오류 발생: {e}")



# 실행할 py 파일의 목록
py_files = ["API.py","API_2.py","get_battery_charge_DB.py"] # 본 스크립트 파일(Auto_Script)과 같은 디렉토리 내 파일들은, 파일이름만 전달해도 접근 가능함.

# 커맨드 라인 인자로 시작 날짜와 종료 날짜 받기 (인자가 없으면 None)

start_date_input = sys.argv[1] if len(sys.argv) > 1 else None # execute_py 함수를 통해 실행되는 파이썬 스크립트 내 인자(날짜,위경도,위치)가 리스트의 형태로 저장되는데, 이 리스트를 의미함. execute함수 구성에 따라 2번째요소는 시작날짜 값
end_date_input = sys.argv[2] if len(sys.argv) >=2  else None # execute_py 함수를 통해 실행되는 파이썬 스크립트 내 인자(날짜,위경도,위치)가 리스트의 형태로 저장되는데, 이 리스트를 의미함. execute함수 구성에 따라 3번째요소는 종료날짜 값
latitude_input = sys.argv[3] if len(sys.argv) >=3  else None # execute_py 함수를 통해 실행되는 파이썬 스크립트 내 인자(날짜,위경도,위치)가 리스트의 형태로 저장되는데, 이 리스트를 의미함. execute함수 구성에 따라 4번째요소는 위도 값
longitude_input = sys.argv[4] if len(sys.argv) >=4  else None# execute_py 함수를 통해 실행되는 파이썬 스크립트 내 인자(날짜,위경도,위치)가 리스트의 형태로 저장되는데, 이 리스트를 의미함. execute함수 구성에 따라 5번째요소는 경도 값
first_api_input = sys.argv[5] if len(sys.argv) >=5  else None# 첫 번째 api인 카카오 api 키 값
second_api_input = sys.argv[6] if len(sys.argv) >=6  else None # 두 번째 api인 기상청 api 키 값



# 지정된 날짜 업데이트
start_date = update_date(start_date_input) # update_date 함수를 사용하여 command line으로 전달된 시작날짜 값의 형식 전처리
end_date = update_date(end_date_input) # update_date 함수를 사용하여 command line으로 전달된 종료날짜 값의 형식 전처리


print(sys.argv)
print(start_date)
print(end_date)
print(latitude_input)
print(longitude_input)
print(first_api_input)
print(second_api_input)

# 시작 날짜부터 종료 날짜까지 각 py 파일을 실행
# previous_six_date = start_date # 6일 전 날짜를 시작날짜에 맞추고(반복 시행 위함)
# current_date = end_date # 현재 날짜를 종료 날짜에 맞춤
print(start_date)
print(end_date)

try:
    while start_date <= end_date: #현재날짜가 종료날짜에 도달할 떄까지
        for file in py_files:# 실행할 py파일들에 대해서
            # execute_py(file, current_date.strftime("%Y%m%d"), end_date.strftime("%Y%m%d"), float(latitude_input), float(longitude_input),  first_api_input.replace(" ", "").replace(".", ""),second_api_input.replace(" ", "").replace(".", "") )  # execute함수 실행하여 각 py파일에 대하여 정해 둔 파이썬 스크립트 실행. 문자열 형태로 변환하여 전달
            execute_py(file, start_date.strftime("%Y%m%d"), end_date.strftime("%Y%m%d"), latitude_input, longitude_input,  first_api_input,second_api_input )
        start_date += datetime.timedelta(days=1)  # 한 개의 파일 실행이 완료되면 timedelta 활용하여 다음 날짜로 이동
except:

    paths = [r"C:\Users\\user1\Desktop\Battery_Charging_Algorithm\Algorithm_1_3\API.py",r"C:\Users\user1\Desktop\Battery_Charging_Algorithm\Algorithm_1_3\API_2.py",r"C:\Users\user1\Desktop\Battery_Charging_Algorithm\Algorithm_1_3\get_battery_charge_DB.py" ]

    while start_date <= end_date: #현재날짜가 종료날짜에 도달할 떄까지
          for path in paths:# 실행할 py파일들에 대해서
                # execute_py(file, current_date.strftime("%Y%m%d"), end_date.strftime("%Y%m%d"), float(latitude_input), float(longitude_input),  first_api_input.replace(" ", "").replace(".", ""),second_api_input.replace(" ", "").replace(".", "") )  # execute함수 실행하여 각 py파일에 대하여 정해 둔 파이썬 스크립트 실행. 문자열 형태로 변환하여 전달
                execute_py(path, start_date.strftime("%Y%m%d"), end_date.strftime("%Y%m%d"), latitude_input, longitude_input,  first_api_input,second_api_input )
                start_date += datetime.timedelta(days=1)  # 한 개의 파일 실행이 완료되면 timedelta 활용하여 다음 날짜로 이동

 
print(sys.argv)
print('chainsmokers')