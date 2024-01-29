# -*- coding: cp949 -*-   # ���� ���ڵ� ������ ���� ù�Ӹ��� ������־� ���ڵ� ������ ������

import os 
import sys
import datetime as dt
import schedule
import time
import pymysql
import datetime



# #MYSQL����
try: 
    conn = pymysql.connect(host='175.121.197.205',port =3306, user='bkc_manager', password='bkcManager1234@', db='smart_lora', charset='utf8')
    print(f" DB�����Ϸ�")
except:
    print(f"DB���� �� ���� �߻�. ���� �̸�, ��й�ȣ Ȥ�� ����ڱ����� �����Ǿ� �ִ��� ��Ȯ���ϼ���")

#Ŀ������
cur = conn.cursor()
#�Է� ���� �ۼ�
try:
    
    sql_inputmaker = f"""select distinct b.* from equipment a, facility b, equipment_type c
where a.dno > 0
and a.type_dno = c.dno
and binary c.service like '%c%'
and binary c.service like '%d%'
and a.facility_dno = b.dno
;"""
    cur.execute(sql_inputmaker)
    
    # ��� ����
    result_input = cur.fetchall()  # ��� �ᱣ�� ��ȯ


    # Ư�� �÷� �� ����ȭ
    latitude_input = result_input[0][7]  # ���� ���� ������ ����
    longitude_input = result_input[0][8] # �浵 ���� ������ ����

    print(latitude_input)
    print(longitude_input)
    

except:
    print(f"������ ���� �߻�. ���� �� ��Ÿ ���� ��Ȯ���ϼ���.")
    
# ���� ����
cur.execute(sql_inputmaker)
# �Է°� ����
conn.commit()
print(f" �Է°� ���� �Ϸ�")
# MYSQL ���� ����
conn.close()


current_date = dt.date.today() # datetime.date.today()�޼���� ���� ��¥ ��ȯ
current_date_input = current_date.strftime('%Y%m%d') # start_date�� �䱸�ϴ� ������ 'yyyymmdd' �����̹Ƿ�, strftime�޼��带 Ȱ���� ���ڿ��� �ٲ�
current_datetime = datetime.datetime.now()    


# dt.datetime.now()�޼���� ���� ��¥ ��ȯ. 2023-12-07 �������� ������

next_date = current_date + dt.timedelta(days=1) # current_date�κ��� �Ϸ� �� ��¥ ���
next_date_input = next_date.strftime('%Y%m%d') # start_date�� �䱸�ϴ� ������ 'yyyymmdd' �����̹Ƿ�, strftime�޼��带 Ȱ���� ���ڿ��� �ٲ�
previous_six_date = current_date - dt.timedelta(days=6)# current_date�κ��� �Ϸ� �� ��¥ ���
previous_six_date_input = previous_six_date.strftime('%Y%m%d') # start_date�� �䱸�ϴ� ������ 'yyyymmdd' �����̹Ƿ�, strftime�޼��带 Ȱ���� ���ڿ��� �ٲ�

# py ������ �����ϴ� �Լ�
def execute_py(file_path, start_date, end_date, latitude_input, longitude_input, first_api_input, second_api_input): # py���� �����Լ�, �Է°��� �䱸�ϴ� ���� ���� ������ �ڵ����� �����ϵ��� �ϴ� �Լ�. ���ϰ�ο� ��¥,���浵 �� �Է°����� ���ڷ� ����.
    try:
        os.system(f"python {file_path} {start_date} {end_date} {latitude_input} {longitude_input} {first_api_input} {second_api_input}") # ���� ���� ��ũ��Ʈ ����.  �ü������ ���ڿ��� ������ ����� �����ϴ� �����Լ� os.system() �޼��� ���� py���� �����ϴ� ��ũ��Ʈ ������ f-string���� ����.
    except Exception as e:
        print(f"�۾� ���� �� ���� �߻�: {e}")
        
# ������ py ������ ���
py_files = ["API.py","API_2.py","get_battery_charge_DB.py"] # �� ��ũ��Ʈ ����(Auto_Script)�� ���� ���丮 �� ���ϵ���, �����̸��� �����ص� ���� ������.
        

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
    # ���� ��ΰ� ��ȿ���� ������ ���� �ߴ�
    print(f"�����۾��������� üũ ��..{current_datetime}")
    if not os.path.exists(file_path): 
        print("���� ��ΰ� ��ȿ���� �ʽ��ϴ�. �����ٷ��� �ߴ��մϴ�.")
        break

    schedule.run_pending()
    time.sleep(1) # 1�ʸ��� �۾����������� �����Ǿ����� �˻�                 