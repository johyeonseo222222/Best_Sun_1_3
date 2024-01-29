# -*- coding: cp949 -*-   # 파일 인코딩 형식을 파일 첫머리에 명시해주어 인코딩 오류를 방지함

import os 
import sys
import datetime as dt
import schedule
import time
import pymysql
import datetime



# #MYSQL연결
try: 
    conn = pymysql.connect(host='175.121.197.205',port =3306, user='bkc_manager', password='bkcManager1234@', db='smart_lora', charset='utf8')
    print(f" DB연동완료")
except:
    print(f"DB연동 중 문제 발생. 유저 이름, 비밀번호 혹은 사용자권한이 설정되어 있는지 재확인하세요")

#커서생성
cur = conn.cursor()
#입력 쿼리 작성
try:
    
    sql_inputmaker = f"""select distinct b.* from equipment a, facility b, equipment_type c
where a.dno > 0
and a.type_dno = c.dno
and binary c.service like '%c%'
and binary c.service like '%d%'
and a.facility_dno = b.dno
;"""
    cur.execute(sql_inputmaker)
    
    # 결과 추출
    result_input = cur.fetchall()  # 모든 결괏값 반환


    # 특정 컬럼 값 변수화
    latitude_input = result_input[0][7]  # 위도 값을 변수에 저장
    longitude_input = result_input[0][8] # 경도 값을 변수에 저장

    print(latitude_input)
    print(longitude_input)
    

except:
    print(f"쿼리에 오류 발생. 문법 및 기타 사항 재확인하세요.")
    
# 쿼리 실행
cur.execute(sql_inputmaker)
# 입력값 저장
conn.commit()
print(f" 입력값 저장 완료")
# MYSQL 연결 종료
conn.close()


current_date = dt.date.today() # datetime.date.today()메서드로 오늘 날짜 반환
current_date_input = current_date.strftime('%Y%m%d') # start_date가 요구하는 형식은 'yyyymmdd' 형식이므로, strftime메서드를 활용해 문자열로 바꿈
current_datetime = datetime.datetime.now()    


# dt.datetime.now()메서드로 오늘 날짜 반환. 2023-12-07 형식으로 내보냄

next_date = current_date + dt.timedelta(days=1) # current_date로부터 하루 후 날짜 계산
next_date_input = next_date.strftime('%Y%m%d') # start_date가 요구하는 형식은 'yyyymmdd' 형식이므로, strftime메서드를 활용해 문자열로 바꿈
previous_six_date = current_date - dt.timedelta(days=6)# current_date로부터 하루 전 날짜 계산
previous_six_date_input = previous_six_date.strftime('%Y%m%d') # start_date가 요구하는 형식은 'yyyymmdd' 형식이므로, strftime메서드를 활용해 문자열로 바꿈

# py 파일을 실행하는 함수
def execute_py(file_path, start_date, end_date, latitude_input, longitude_input, first_api_input, second_api_input): # py파일 실행함수, 입력값을 요구하는 여러 개의 파일을 자동으로 실행하도록 하는 함수. 파일경로와 날짜,위경도 등 입력값들을 인자로 받음.
    try:
        os.system(f"python {file_path} {start_date} {end_date} {latitude_input} {longitude_input} {first_api_input} {second_api_input}") # 파일 실행 스크립트 포맷.  운영체제에서 문자열로 구성된 명령을 실행하는 내장함수 os.system() 메서드 내에 py파일 실행하는 스크립트 포맷을 f-string으로 전달.
    except Exception as e:
        print(f"작업 실행 중 오류 발생: {e}")
        
# 실행할 py 파일의 목록
py_files = ["API.py","API_2.py","get_battery_charge_DB.py"] # 본 스크립트 파일(Auto_Script)과 같은 디렉토리 내 파일들은, 파일이름만 전달해도 접근 가능함.
        

def scheduled_job(file_path):

    
    execute_py(file_path=file_path, 
               start_date=previous_six_date_input, 
               end_date=current_date_input, 
               latitude_input=latitude_input, 
               longitude_input=longitude_input, 
               first_api_input='01e9193c9c3f6b3631db3612599f6246', 
               second_api_input='ghBQh0IXRimQUIdCFzYp4w')

file_path = r"D:\hyeonseo\Battery_Charging_Algorithm\Battery_Charge_V3\Battery_Charge_V3\Battery_Charging_1_3\ver1_3\Auto_Script.py"

schedule.every().day.at("20:30").do(scheduled_job, file_path=file_path)

while True:
    # 파일 경로가 유효하지 않으면 루프 중단
    print(f"예약작업실행조건 체크 중..{current_datetime}")
    if not os.path.exists(file_path): 
        print("파일 경로가 유효하지 않습니다. 스케줄러를 중단합니다.")
        break

    schedule.run_pending()
    time.sleep(1) # 1초마다 작업실행조건이 충족되었는지 검사                 