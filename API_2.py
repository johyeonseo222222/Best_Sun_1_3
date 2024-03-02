# -*- coding: utf-8 -*-   # ���� ���ڵ� ������ ���� ù�Ӹ��� ������־� ���ڵ� ������ ������

import pandas as pd
from haversine import haversine # pip install �ʿ�
import requests
import json
import datetime
import os
import sys

print(" API ���� 2 ��ũ��Ʈ ����")
    
if len(sys.argv) >=7:  
    start_date_input = sys.argv[1] # ���۳�¥�� ����Ʈ�� �� ��° ����̹Ƿ� [1] �ε����� ���� ������ ����
    start_date_input = start_date_input.replace('-', '') # �Էµ� ��¥�� ������ ��ó�� �ϴ� ����. �������� �ִٸ� �̸� �����϶�� ���
    print(start_date_input) # ���۳�¥�� �ùٸ� �������� �ٲ������ Ȯ���ϱ� ���� print��
    end_date_input = sys.argv[2] # ���ᳯ¥�� ����Ʈ�� �� ��° ����̹Ƿ� [2] �ε����� ���� ������ ����
    end_date_input = end_date_input.replace('-', '') # �Էµ� ��¥�� ������ ��ó�� �ϴ� ����. �������� �ִٸ� �̸� �����϶�� ���
    print(end_date_input) # ���ᳯ¥��  �ùٸ� �������� �ٲ������ Ȯ���ϱ� ���� print��




else: # ���� Ŀ�ǵ������ ���ڰ� 7�� ���϶��, �� ���ڰ� �����Ǿ��ų� �߸��� �������� �Էµ� ��Ȳ�̶�� => ������ ���ڸ� �Է��Ͽ� �����ؾ� ��. �����ƾ: �����Է�
    print("Ŀ�ǵ���� ������ ������ �ٽ� Ȯ���ϼ���. ���� ��¥�� ���� ��¥�� YYYYMMDD �������� �Է��ϼ���.������ �浵�� �Ҽ������� �Է��ϼ���. ") # �Է������� �ٽ� Ȯ���غ���� �޽����� ����ϰ�
    
    
    start_date_input = input("������ ������ ���� ��¥�� YYYYMMDD �������� �Է��ϼ���(ex. 2�� 22��~2�� 25���� �������� �����Ϸ��� YYYY0221):") # ���۳�¥�� YYYMMDD�� �Է��ϴ� input �κ�. �����ϰ��� �ϴ� �� �Ϸ� �� ��¥�� �Է�(T-1����)
    start_date_input = start_date_input.replace('-', '') # �Է°��� �������� �ִٸ� �����϶�� ���. �������� �ִ� ���·� �ԷµǸ� ȣ�� URL�� ����� �� ����
       
    end_date_input = input("������ ������ ���� ��¥�� YYYYMMDD �������� �Է��ϼ���(ex. 2�� 22��~2�� 25���� �������� �����Ϸ��� YYYY0224):") # ���ᳯ¥�� YYYMMDD�� �Է��ϴ� input �κ�. �����ϰ��� �ϴ� �� �Ϸ� �� ��¥�� �Է�(T-1����)
    end_date_input = end_date_input.replace('-', '')  # �Է°��� �������� �ִٸ� �����϶�� ���. �������� �ִ� ���·� �ԷµǸ� ȣ�� URL�� ����� �� ����     
    
    
   

try: # ���⼭�� try - except ����� ���� if len(sys.argv) >=7 ������ ������ ��� �ٷ� ����Ǵ� ��������. else�� ���� �ڵ�� ����� ������, if�� else �� ��� ��� ��ó���� �����ؾ� �ϱ� ������.
    start_date = datetime.datetime.strptime(start_date_input, '%Y%m%d') # ���۳�¥�� ȣ��URL�� ����� �� �ִ� �������� ��ó��. �̸� ���� ���ڿ��� datetime��ü�� ��ȯ�ϴ� strptime �޼��� Ȱ�� 
    end_date = datetime.datetime.strptime(end_date_input, '%Y%m%d')  # ���ᳯ¥��  ȣ��URL�� ����� �� �ִ� �������� ��ó��. �̸� ���� ���ڿ��� datetime��ü�� ��ȯ�ϴ� strptime �޼��� Ȱ��


except ValueError:# Ŀ�ǵ���ο��� ���ڰ� �߸� �ԷµǾ��ų�, ��ó���� �ùٸ��� ������� ������
    print("��¥ Ȥ�� ���浵 �� ���Ŀ� ������ �ֽ��ϴ�. ��¥�� ��� YYYYMMDD �������� �Է��ϼ���. ���浵�� ��� �Ҽ������� ��Ȯ�� �Է��ϼ��� ") # ��� �޽��� ����ϰ�
    sys.exit() # ��ũ��Ʈ ����. �̷��� ��쿡�� �ٽ� �����Ͽ� Ŀ�ǵ������ ����� �Է����־�� ��
   

start_date = datetime.datetime.strptime(start_date_input, '%Y%m%d') # ���۳�¥�� ȣ��URL�� ����� �� �ִ� �������� ��ó��. �̸� ���� ���ڿ��� datetime��ü�� ��ȯ�ϴ� strptime �޼��� Ȱ�� 
end_date = datetime.datetime.strptime(end_date_input, '%Y%m%d')  # ���ᳯ¥��  ȣ��URL�� ����� �� �ִ� �������� ��ó��. �̸� ���� ���ڿ��� datetime��ü�� ��ȯ�ϴ� strptime �޼��� Ȱ��


# �Է¹��� ��¥�� ������� ���� ��¥ �� ���� ��¥ ���

current_date = datetime.date.today()
# current_date = end_date # ���糯¥�� ���۳�¥�� ����. �̴� ���۳�¥���� ���ᳯ¥���� �ݺ��ϱ� ����.
current_date_str = current_date.strftime('%Y%m%d') # datetime��ü�� YYYYMMDD������ ���ڿ��� �ٲ� ���۳�¥(���ó�¥)�� ����
end_date_str = end_date.strftime('%Y%m%d') # datetime��ü�� YYYYMMDD������ ���ڿ��� �ٲ� ���۳�¥(���ó�¥)�� ����
previous_six_date = end_date - datetime.timedelta(days=6) 
# previous_six_date = start_date
previous_six_date_str = previous_six_date.strftime('%Y%m%d') # datetime��ü�� YYYYMMDD������ ���ڿ��� �ٲ�
previous9_date = end_date - datetime.timedelta(days=9) #ȣ�� URL���� tm1_value�� ������ ��. ���������ʹ� 3~10�� ������ ������ �����ϱ� ������ 9�� �� �������� Ȯ��
previous9_date_str = previous9_date.strftime('%Y%m%d') # datetime��ü�� YYYYMMDD������ ���ڿ��� �ٲ�
later_date = end_date + datetime.timedelta(days=1) # ȣ�� url���� tm3_value�� ����� ��. T������ ������ �о�� ����
later_date_str = later_date.strftime('%Y%m%d')# datetime��ü�� YYYYMMDD������ ���ڿ��� �ٲ�
later3_date = end_date + datetime.timedelta(days=3) # ȣ�� url���� tm4_value�� ����� ��. T���� ��ü �ð����� ������ �о�� ����
later3_date_str = later3_date.strftime('%Y%m%d')# datetime��ü�� YYYYMMDD������ ���ڿ��� �ٲ� 




def preprocess_weather_data(input_path, output_path): # ��ó���ϴ� �Լ� ����, save_file_path�� ���� �ҷ��ͼ� ��ó���� �� ouput_path�� ������ ��ȹ
    df_weather_API_pre = pd.read_csv(input_path, sep='\s+', comment='#', encoding='cp949') # input_path�� Ȱ���� ������ �ҷ�����, ������ �����ڷ� �ν��ϰ� #�κ��� �ּ�ó���� ������ ������
    
   
    with open(input_path, "r", encoding="cp949") as file: # �б���� ����
        lines = file.readlines() # ���� �� ��ҵ��� ����Ʈ �� ��ҷ� ��ȯ. �ε��� �� ��ҿ� �����Ͽ� ��ó���ϱ� ���� �ٲ��ִ� readlines()�޼��� Ȱ��

    column_index = [i for i, line in enumerate(lines) if line.startswith("# YYMMDDHHMI")][0] # ������ ���� list �� line���� ��ȸ�ϸ鼭 # YYMMDDHHMI�� �����ϴ� �κ��� ��� �ε����� ��ȯ�ϴ� enumerate����, �ε����� ��ȯ(i for i, line )
    column_names = lines[column_index].split() # ���� �������� split�ؼ� �÷������� ��ȯ�� �� column_names�� ����
    column_names.remove('#') # ó���� ������� �÷� ���̿� ����� ������ �÷� ���̰� ���� ����. �̿� ����� ���� �� �÷� ���̸� ���������� �ٿ��� ���� �����ֱ�
    df_weather_API_pre.columns = column_names # ����� ���� �޼���� �÷��� ����

    df_weather_API = df_weather_API_pre[['YYMMDDHHMI', 'STN', 'WS', 'HM', 'CA', 'SS', 'SI', 'TS']] # ����� �÷��� �߷��� ���������� ���� �����
    df_weather_API['YYMMDDHHMI'] = df_weather_API['YYMMDDHHMI'].astype(str) # ��¥ �� �ð������� ������ YYMMDDHHM ���� �������� ���͸��ϱ� ����, ���� ���ڿ��� �ٲ���
    df_weather_API['datetime'] = pd.to_datetime(df_weather_API['YYMMDDHHMI']).dt.strftime('%Y-%m-%d %H:00') # datetime64ns �������� ����ȯ ���� ��, strftime�޼���� ��-��-��- ��:00 �������� ����������.
    df_weather_API['datetime'] = pd.to_datetime(df_weather_API['datetime']) # strftime �޼��带 Ȱ���ϸ� ���ڿ��� �ٽ� �ٲ�Ƿ�, datetime���� �纯ȯ
    df_weather_API = df_weather_API[(df_weather_API['datetime'].dt.hour >= 7) & (df_weather_API['datetime'].dt.hour <= 20)] # dt.hour���� ��-��-��-��:00 ���� '��' �κи� �̾Ƴ� ��, �̸� �������� 7�ÿ� 20�� ���̸� ���͸�
    
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



    df_weather_API.to_csv(output_path, index=False) # ��ó�� �Ϸ�. output_path�� ���� ��������
    return df_weather_API



# current_date = datetime.date.today()
# current_date = end_date # ���糯¥�� ���۳�¥�� ����. �̴� ���۳�¥���� ���ᳯ¥���� �ݺ��ϱ� ����.
# previous_six_date = current_date - datetime.timedelta(days=6) 
# previous_six_date = start_date

print(start_date)
print(end_date)

while start_date <= end_date:

    start_date_str = start_date.strftime('%Y%m%d')

    # ��ó�� �Լ� ȣ��

    input_path_weather = fr"C:\Users\user1\Desktop\Battery_Charging_Algorithm\API\Weather_{start_date_str}_file_pre.csv" # ��ó�� ������  ����������� ���
    output_path_weather = fr"C:\Users\user1\Desktop\Battery_Charging_Algorithm\API\Weather_{start_date_str}_file.csv"  # ��ó�� �Ϸ��� ����������� ���


    try:
     df_weather_API= preprocess_weather_data(input_path_weather , output_path_weather) # ��ó�� �Ϸ�� ����API������ ������ df_weather_API�� ����
     print(f"����api��ó�� �Ϸ�")# ��ó�� �Ϸ� �޽��� ���                                         
                                         
                                         
    except:
     print(f"����api��ó���Լ� ó�� �� ���� �߻�") # ��ó���Լ� ó�� �� ���� �߻�. API_pre ������ Ȯ���غ� ��. �÷��� ������ ����� �Ǿ����� �ʾ� ������ ���� ���ɼ��� ����. �÷��� ������ ����� �������� ������ with_open�޼��� �κп��� sep, comment �ɼ��� �߸��Ǿ� ���� ���ɼ�
     sys.exit() # ��ũ��Ʈ ����. �̷��� ��쿡�� �ٽ� �����Ͽ� Ŀ�ǵ������ ����� �Է����־�� ��

    start_date += datetime.timedelta(days=1) 

#�������



#����api ��ó������



def preprocess_forecast_data(input_path, output_path): # ���� ��ó�� �ϴ� �Լ� ����. �ҷ��� �� ���ϰ��, ������ �� ���ϰ�θ� ���ڷ� �����
 
    df_forecast_API = pd.read_csv(input_path, sep='\s+', comment='#', encoding='cp949') # �ҷ��� �� ����, ������ �����ڷ� �ν�, #�� �ּ�ó���ϰ�, cp949 ���ڵ� ��� ���



# �÷����� ����
    with open(input_path, "r", encoding="cp949") as file: # ���� ������ �б� ������� ����
     lines = file.readlines() # ���� �� ��ҵ��� ����Ʈ �� ��ҷ� ��ȯ. �ε��� �� ��ҿ� �����Ͽ� ��ó���ϱ� ���� �ٲ��ִ� readlines()�޼��� Ȱ��


    # "reg_id"�� �����ϴ� ������ ã�� �÷������� ���
    column_index = [i for i, line in enumerate(lines) if line.startswith("# REG_ID")][0] # ������ ���� list �� line���� ��ȸ�ϸ鼭 #reg_id�� �����ϴ� �κ��� ��� �ε����� ��ȯ�ϴ� enumerate����, �ุ ��ȯ(0)
    column_names = lines[column_index].split() # ���� �������� split�ؼ� �÷������� ��ȯ�� �� column_names�� ����


    # '#' �÷� ����
    column_names.remove('#') # ó���� ������� �÷� ���̿� ����� ������ �÷� ���̰� ���� ����. �̿� ����� ���� �� �÷� ���̸� ���������� �ٿ��� ���� �����ֱ�
    
   

    # ������ �������� �÷����� ����
    df_forecast_API.columns = column_names # ����� ���� �޼���� �÷��� ����

    print(df_forecast_API.columns)


    df_forecast_API['TM_EF'] = df_forecast_API['TM_EF'].astype(str) # TM_EF �÷��� ���ڿ��� ��ȯ. �̴� TM_EF (������ȿ�ð�)�� ������ ���͸��� �ϱ� ����.

    df_forecast_API['datetime'] = pd.to_datetime(df_forecast_API['TM_EF']) #  TM_EF �÷��� datetime�������� ��ȯ
   


    def season_searching(month): # �������� �����ϴ� �Լ�.

      if 3<= month <=5:
        return 'spring'
      elif 6<= month <= 8:
        return 'summer'
      elif 9<= month <=11:
        return 'fall'
      else:
        return 'winter'

  
    df_forecast_API['season'] = df_forecast_API['datetime'].dt.month.apply(season_searching) # apply �޼��带 ���� season_searching �Լ� ����

    df_forecast_API.to_csv(output_path, index = False) # ��ó�� �Ϸ�. ���� ��������.


input_path_forecast = fr"C:\Users\user1\Desktop\Battery_Charging_Algorithm\API\forecast_{end_date_str}_pre.csv" # ��ó�� ���� ����API������ ���
output_path_forecast = fr"C:\Users\user1\Desktop\Battery_Charging_Algorithm\API\forecast_{end_date_str}.csv" #��ó������ ����API������ ���


try:
 df_forecast_API = preprocess_forecast_data(input_path_forecast, output_path_forecast) # ��ó���Ϸ��� ����api������ df_forecast_API�� ����
 print(f"����api��ó���Ϸ�")
except:
 print(f"����api��ó���Լ� ó�� �� ���� �߻�")
 sys.exit() # ��ũ��Ʈ ����. �̷��� ��쿡�� �ٽ� �����Ͽ� Ŀ�ǵ������ ����� �Է����־�� ��

print("API ���� 2 ��ũ��Ʈ ����")

# os.system("pause")

