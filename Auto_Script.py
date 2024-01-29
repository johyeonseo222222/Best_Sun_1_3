# -*- coding: cp949 -*-   # ���� ���ڵ� ������ ���� ù�Ӹ��� ������־� ���ڵ� ������ ������

import os 
import sys
import datetime
import schedule
import time
import pymysql



current_date = datetime.date.today() # datetime.date.today()�޼���� ���� ��¥ ��ȯ
previous_six_date = current_date - datetime.timedelta(days=6)

# ������ ��¥�� ������Ʈ�ϴ� �Լ�
def update_date(input_date): # input_date�� ���ڷ� Ȱ���ϴ� update_date�Լ� ����. ������� ��¥�Է°��� �ùٸ� ������ ��¥�������� ��ȯ�Ͽ� API ȣ�� URL���� �� Ȱ���ϵ��� '��ó��'���ִ� �Լ�
    if input_date: # ��¥���� �ԷµǸ�
        date_object = datetime.datetime.strptime(input_date, '%Y%m%d') # strptime �޼��带 Ȱ���� ���ڿ��� datetime ��ü�� ��ȯ�ϰ� YYYYMMDD�������� ������
        return date_object #��ó�� �Ϸ�� ��¥���� ��ȯ
    else: # ���� �Է°��� �������� ������
        # input_date�� None�� ��� ���� ��¥�� ��ȯ
        current_date = datetime.date.today() # datetime.date.today()�޼���� ���� ��¥ ��ȯ
        previous_six_date = current_date - datetime.timedelta(days=6)

        return current_date,previous_six_date # strftime �޼��带 Ȱ���� ������������ ���ڿ��� ��ȯ

# py ������ �����ϴ� �Լ�
def execute_py(file_path, start_date, end_date, latitude_input, longitude_input, first_api_input, second_api_input): # py���� �����Լ�, �Է°��� �䱸�ϴ� ���� ���� ������ �ڵ����� �����ϵ��� �ϴ� �Լ�. ���ϰ�ο� ��¥,���浵 �� �Է°����� ���ڷ� ����.
    try:
        os.system(f"python {file_path} {start_date} {end_date} {latitude_input} {longitude_input} {first_api_input} {second_api_input}") # ���� ���� ��ũ��Ʈ ����.  �ü������ ���ڿ��� ������ ����� �����ϴ� �����Լ� os.system() �޼��� ���� py���� �����ϴ� ��ũ��Ʈ ������ f-string���� ����.
    except Exception as e:
        print(f"�۾� ���� �� ���� �߻�: {e}")



# ������ py ������ ���
py_files = ["API.py","API_2.py","get_battery_charge_DB.py"] # �� ��ũ��Ʈ ����(Auto_Script)�� ���� ���丮 �� ���ϵ���, �����̸��� �����ص� ���� ������.

# Ŀ�ǵ� ���� ���ڷ� ���� ��¥�� ���� ��¥ �ޱ� (���ڰ� ������ None)

start_date_input = sys.argv[1] if len(sys.argv) > 1 else None # execute_py �Լ��� ���� ����Ǵ� ���̽� ��ũ��Ʈ �� ����(��¥,���浵,��ġ)�� ����Ʈ�� ���·� ����Ǵµ�, �� ����Ʈ�� �ǹ���. execute�Լ� ������ ���� 2��°��Ҵ� ���۳�¥ ��
end_date_input = sys.argv[2] if len(sys.argv) >=2  else None # execute_py �Լ��� ���� ����Ǵ� ���̽� ��ũ��Ʈ �� ����(��¥,���浵,��ġ)�� ����Ʈ�� ���·� ����Ǵµ�, �� ����Ʈ�� �ǹ���. execute�Լ� ������ ���� 3��°��Ҵ� ���ᳯ¥ ��
latitude_input = sys.argv[3] if len(sys.argv) >=3  else None # execute_py �Լ��� ���� ����Ǵ� ���̽� ��ũ��Ʈ �� ����(��¥,���浵,��ġ)�� ����Ʈ�� ���·� ����Ǵµ�, �� ����Ʈ�� �ǹ���. execute�Լ� ������ ���� 4��°��Ҵ� ���� ��
longitude_input = sys.argv[4] if len(sys.argv) >=4  else None# execute_py �Լ��� ���� ����Ǵ� ���̽� ��ũ��Ʈ �� ����(��¥,���浵,��ġ)�� ����Ʈ�� ���·� ����Ǵµ�, �� ����Ʈ�� �ǹ���. execute�Լ� ������ ���� 5��°��Ҵ� �浵 ��
first_api_input = sys.argv[5] if len(sys.argv) >=5  else None# ù ��° api�� īī�� api Ű ��
second_api_input = sys.argv[6] if len(sys.argv) >=6  else None # �� ��° api�� ���û api Ű ��



# ������ ��¥ ������Ʈ
start_date = update_date(start_date_input) # update_date �Լ��� ����Ͽ� command line���� ���޵� ���۳�¥ ���� ���� ��ó��
end_date = update_date(end_date_input) # update_date �Լ��� ����Ͽ� command line���� ���޵� ���ᳯ¥ ���� ���� ��ó��


print(sys.argv)
print(start_date)
print(end_date)
print(latitude_input)
print(longitude_input)
print(first_api_input)
print(second_api_input)

# ���� ��¥���� ���� ��¥���� �� py ������ ����
# previous_six_date = start_date # 6�� �� ��¥�� ���۳�¥�� ���߰�(�ݺ� ���� ����)
# current_date = end_date # ���� ��¥�� ���� ��¥�� ����
print(start_date)
print(end_date)

try:
    while start_date <= end_date: #���糯¥�� ���ᳯ¥�� ������ ������
        for file in py_files:# ������ py���ϵ鿡 ���ؼ�
            # execute_py(file, current_date.strftime("%Y%m%d"), end_date.strftime("%Y%m%d"), float(latitude_input), float(longitude_input),  first_api_input.replace(" ", "").replace(".", ""),second_api_input.replace(" ", "").replace(".", "") )  # execute�Լ� �����Ͽ� �� py���Ͽ� ���Ͽ� ���� �� ���̽� ��ũ��Ʈ ����. ���ڿ� ���·� ��ȯ�Ͽ� ����
            execute_py(file, start_date.strftime("%Y%m%d"), end_date.strftime("%Y%m%d"), latitude_input, longitude_input,  first_api_input,second_api_input )
        start_date += datetime.timedelta(days=1)  # �� ���� ���� ������ �Ϸ�Ǹ� timedelta Ȱ���Ͽ� ���� ��¥�� �̵�
except:

    paths = [r"C:\Users\\user1\Desktop\Battery_Charging_Algorithm\Algorithm_1_3\API.py",r"C:\Users\user1\Desktop\Battery_Charging_Algorithm\Algorithm_1_3\API_2.py",r"C:\Users\user1\Desktop\Battery_Charging_Algorithm\Algorithm_1_3\get_battery_charge_DB.py" ]

    while start_date <= end_date: #���糯¥�� ���ᳯ¥�� ������ ������
          for path in paths:# ������ py���ϵ鿡 ���ؼ�
                # execute_py(file, current_date.strftime("%Y%m%d"), end_date.strftime("%Y%m%d"), float(latitude_input), float(longitude_input),  first_api_input.replace(" ", "").replace(".", ""),second_api_input.replace(" ", "").replace(".", "") )  # execute�Լ� �����Ͽ� �� py���Ͽ� ���Ͽ� ���� �� ���̽� ��ũ��Ʈ ����. ���ڿ� ���·� ��ȯ�Ͽ� ����
                execute_py(path, start_date.strftime("%Y%m%d"), end_date.strftime("%Y%m%d"), latitude_input, longitude_input,  first_api_input,second_api_input )
                start_date += datetime.timedelta(days=1)  # �� ���� ���� ������ �Ϸ�Ǹ� timedelta Ȱ���Ͽ� ���� ��¥�� �̵�

 
print(sys.argv)
print('chainsmokers')