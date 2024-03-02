# -*- coding: utf-8 -*-   # ���� ���ڵ� ������ ���� ù�Ӹ��� ������־� ���ڵ� ������ ������

import pandas as pd
from haversine import haversine # pip install �ʿ�
import requests
import json
import datetime
import os
import sys



print("API ���� 1 ��ũ��Ʈ ����")



print(sys.argv) # Ŀ�ǵ������ ���� py���� ������ ���� ���ڵ��� ����� ���޵Ǿ����� ����Ͽ� Ȯ���غ��� ���� print��


### �Է¾� Ȯ�� �� ���� ����(��ƾ 1)

## ���� �ڵ� ������ ���� �ʿ��� ���ڵ��� Ŀ�ǵ����(sys.argv)���� ���������� ���޵Ǿ����� Ȯ��
## ���� �̸�(Auto_Script.py), ���۳�¥(���糯¥ ���� 7������ ��), ���ᳯ¥(���糯¥),����,�浵, īī��APIŰ, ���ûAPIŰ

# ���� ���� ���� 7���� ����� ���޵Ǿ����� Ȯ���ϴ� �κ�(��ƾ 1-1)
if len(sys.argv) >=7:

  
    start_date_input = sys.argv[1] 
    start_date_input = start_date_input.replace('-', '') 
    end_date_input = sys.argv[2] 
    end_date_input = end_date_input.replace('-', '') 
    latitude_input = sys.argv[3] 
    longitude_input = sys.argv[4]
    first_api_input = sys.argv[5] 
    second_api_input = sys.argv[6] 



# ���� Ŀ�ǵ������ ���ڰ� 7�� ���϶��, �� ���ڰ� �����Ǿ��ų� �߸��� �������� �Էµ� ��Ȳ���� �����Է����� ��ȯ (�����ƾ1)
else:  
    print("Ŀ�ǵ���� ������ ������ �ٽ� Ȯ���ϼ���. ���� ��¥�� ���� ��¥�� YYYYMMDD �������� �Է��ϼ���.������ �浵�� �Ҽ��� �״�� �Է��ϼ���. �Ǵ� api Ű ���� �ٽ� Ȯ���ϼ��� ") 
    try:  
        
        start_date_input = input("������ ������ ���� ��¥�� YYYYMMDD �������� �Է��ϼ���(ex. 2�� 22���� �������� �����Ϸ��� YYYY0215):")
        start_date_input = start_date_input.replace('-', '') 
       
        end_date_input = input("������ ������ ���� ��¥�� YYYYMMDD �������� �Է��ϼ���(ex. 2�� 22���� �������� �����Ϸ��� YYYY0221):") 
        end_date_input = end_date_input.replace('-', '')  


        latitude_input = float(input("������ �Է��ϼ���: "))
        longitude_input = float(input("�浵�� �Է��ϼ���: ")) 
        if not (-90 <= latitude_input <= 90) or not (-180 <= longitude_input <= 180): 
            print("�߸��� ���� �Ǵ� �浵 ���Դϴ�.") 
         
        first_api_input = input("KAKAO REST API KEY ���� �Է��ϼ���. : ")
        first_api_input =  first_api_input.replace(" ", "").replace(".", "")
        second_api_input = input("���û API KEY ���� �Է��ϼ���. : ")
        second_api_input =  second_api_input.replace(" ", "").replace(".", "")
       

    # �����ƾ 1 ������ ���, ��ũ��Ʈ ����     
    except ValueError as e: 
        print(f"��ȿ�� �Է°��� �Է����ּ���. {e}") 
        exit() 
        sys.exit() 
# ���۳�¥�� ���ᳯ¥ ��ó��(��ƾ 1-2)    
try: 
    start_date = datetime.datetime.strptime(start_date_input, '%Y%m%d') 
    end_date = datetime.datetime.strptime(end_date_input, '%Y%m%d')  
    latitude_input = float(latitude_input) 
    longitude_input = float(longitude_input) 
    first_api_input =  first_api_input.replace(" ", "").replace(".", "") 
    second_api_input =  second_api_input.replace(" ", "").replace(".", "")


# �Է°� ������ ���� ��ũ��Ʈ ����(��ƾ 1-2-1) 
except ValueError as e:
    print(f"��¥ Ȥ�� ���浵 �� ���Ŀ� ������ �ֽ��ϴ�. ��¥�� ��� YYYYMMDD �������� �Է��ϼ���. ���浵�� ��� �Ҽ������� ��Ȯ�� �Է��ϼ���. �Ǵ� api Ű ���� �ٽ� Ȯ���ϼ��� {e} ") 
    sys.exit() 
    

# �Է°� ��ó�� ����, �˰��� ������ ���̴� ��¥ ���� ����(��ƾ1-3)
# try ��ϰ� �Բ� ���̴� else���� ���, try-except���� ���ܰ� �߻����� �ʾ��� �� �ٷ� �����
else:
    current_date = datetime.date.today()
    # end_date = current_date 
    current_date_str = current_date.strftime('%Y%m%d') # datetime��ü�� YYYYMMDD������ ���ڿ��� �ٲ� ���۳�¥(���ó�¥)�� ����
    end_date_str = end_date.strftime('%Y%m%d') # datetime��ü�� YYYYMMDD������ ���ڿ��� �ٲ� ���۳�¥(���ó�¥)�� ����
    previous_six_date = end_date - datetime.timedelta(days=6) 
    # start_date = previous_six_date 
    previous_six_date_str = previous_six_date.strftime('%Y%m%d') # datetime��ü�� YYYYMMDD������ ���ڿ��� �ٲ�
    previous9_date = end_date - datetime.timedelta(days=9) #ȣ�� URL���� tm1_value�� ������ ��. ���������ʹ� 3~10�� ������ ������ �����ϱ� ������ 9�� �� �������� Ȯ��
    previous9_date_str = previous9_date.strftime('%Y%m%d') # datetime��ü�� YYYYMMDD������ ���ڿ��� �ٲ�
    later_date = end_date + datetime.timedelta(days=1) # ȣ�� url���� tm3_value�� ����� ��. T������ ������ �о�� ����
    later_date_str = later_date.strftime('%Y%m%d')# datetime��ü�� YYYYMMDD������ ���ڿ��� �ٲ�
    later3_date = end_date + datetime.timedelta(days=3) # ȣ�� url���� tm4_value�� ����� ��. T���� ��ü �ð����� ������ �о�� ����
    later3_date_str = later3_date.strftime('%Y%m%d')# datetime��ü�� YYYYMMDD������ ���ڿ��� �ٲ� 
  



print(start_date, end_date, latitude_input, longitude_input, first_api_input, second_api_input ) 

print("Ŀ�ǵ���� ���� Ȯ�� �Ϸ�") # ��ó���� ������ ����Ǹ� Ȯ�� �Ϸ� �޽��� ���

base_url = f"https://dapi.kakao.com/v2/local/geo/coord2regioncode.json?x={longitude_input}&y={latitude_input}&input_coord=WGS84" 

print(base_url) #������


headers = {'Authorization' : 'KakaoAK '+first_api_input} 

api_req = requests.get(base_url,headers=headers) 
print(api_req) #������. 200 


input_data = json.loads(api_req.text) 



try: # API�� ���������� �ҷ����� ���
 input_lat = input_data['documents'][0]["y"] # ������ ����
 input_lon = input_data['documents'][0]["x"] # �浵�� ����
 input_region_1depth_name = input_data['documents'][0]['region_1depth_name'] # �� ���� �������� ����
 input_region_2depth_name = input_data['documents'][0]['region_2depth_name'] # �ñ��� ���� �������� ����
except:
  print(f"API���� ��ǥ���� �Ǵ� �������� ���� ����") # ������ �߻��ϴ� �� API �����Ͱ� ����� �ҷ������� �ʾҴٸ� �����޽��� ���
  sys.exit() # ��ũ��Ʈ ����. �̷��� ��쿡�� �ٽ� �����Ͽ� Ŀ�ǵ������ ����� �Է����־�� ��

print(input_region_1depth_name) # �� ���� �������� Ȯ��. �����
print(input_region_2depth_name) # �ñ��� ���� �������� Ȯ��. �����


extracted_location = (input_lat,input_lon) # ������ ���浵���� Ʃ�����·� ������ ����. �� ������ ���� �־��� ��ġ�� ���� ����� ���� ������ ID�� ���ؾ� ��.

API_LOCATION = pd.read_csv(r"C:\Users\user1\Desktop\Battery_Charging_Algorithm\API_LOCATION_V2.csv" )


# �Է°� ���� �ִܰŸ��� ������� ������ ��� �Լ� 
def find_nearest_observatory(extracted_location, API_LOCATION): # API�� ���� ������ ���浵 ���� ���� ���� �������� ���浵 ���� ���ڷ� Ȱ���ϴ� find_nearest_observatory �Լ� ����. ��ǥ �� �Ÿ��� ����ϰ� �ִܰŸ��� �ִ� ���������� ID�� ����.



    min_distance = float('inf') # �ִܰŸ��� ������ ���� ����. �ʱⰪ�� inf�� �����ϴ� ������ �ݺ����� �� ����Ǵ� ���� nearest_location ���� �����Ǹ� �� �Ǳ� ������. ex. ���� �� 5���ε� 3��° ������ �ִܰŸ��� �������� ������ 2�� ���� �񱳰� �Ұ����ϴϱ�, ���Ƿ� inf���� �ּҰŸ��� �����ؼ� ��� ���� ���� �񱳰� �̷��� �� �ֵ��� �ϴ°���.
    nearest_location = None # ���� ����� ���� ������ ������ ���� ����. �ʱⰪ�� None

    # API_LOCATION���� ������ ���� ���� �� ���
    for index, row in API_LOCATION.iterrows(): # ������ ���浵 ���Ͽ��� �ε����� ��ȸ�ϴ� �ݺ�����
        obs_latitude = row['LAT']  # API_LOCATION���� �������� ����
        obs_longitude = row['LON']  # API_LOCATION���� �������� �浵
        endpoint = (obs_latitude, obs_longitude)  # ������ ��ǥ Ʃ�� ����

        # Haversine �޼��带 ����Ͽ� �Ÿ� ���
        try:
         distance = haversine(extracted_location, endpoint, unit = 'km') # haversine �޼��带 ����Ͽ� �Է���ǥ���� ���������� ��ǥ �� �Ÿ� ���ϱ�
        except:
         print(f"�ִܰŸ� ��� �� ���� �߻�") # haversine �޼��尡 ����� ������� �ʾ��� ���� ������ ���ɼ� ����
         # sys.exit() # ��ũ��Ʈ ����. �̷��� ��쿡�� �ٽ� �����Ͽ� Ŀ�ǵ������ ����� �Է����־�� ��/ �̰� Ȱ��ȭ��Ű�� find_nearest_observatory �Լ��� ������� ����. �ͱ�?


        if distance < min_distance: # �ݺ��� �����Ͽ� �ִܰŸ��� ���ϴ� ����. ���� �ִܰŸ��� �����ϸ�
            min_distance = distance # min_distance��  �ִܰŸ� ���� ����
            nearest_location = row['STN_2']  # �ִܰŸ��� ���� endpoint�� �ִ� �࿡�� STN_2 �÷��� ������ ������ ID�� nearest_location�� ����

    return min_distance, nearest_location # �ִܰŸ�


# ���� ����� ������������ ID ã��
nearest_location = find_nearest_observatory(extracted_location, API_LOCATION) #�Լ��� �����Ͽ� �ִܰŸ��� ���������� ID ã��

print(f" �Է°� ���� �ִܰŸ������������� : {nearest_location}")

def download_file(file_url, save_path): # ���� ������ API ���� �ٿ�ε带 ���� download_file�Լ�. ���û API��꿡�� �����ϴ� �Լ���.
    try:
        # ���丮 ����
        if not os.path.exists(os.path.dirname(save_path)): # os.path.dirname�޼���� ���丮 �κ��� ������ �� os.path.exists()�޼���� ���丮 ������ Ȯ��
            os.makedirs(os.path.dirname(save_path)) # os.makedirs �޼���� ���丮 ����

        # ���� �ٿ�ε�
        with open(save_path, 'wb') as f:
            response = requests.get(file_url)
            if response.status_code == 200:
                f.write(response.content)
            else:
                print(f"�ٿ�ε� ����. HTTP ���� �ڵ�: {response.status_code}")
    except Exception as e:
        print(f"���� �߻�: {e}")
        sys.exit() # ��ũ��Ʈ ����. �̷��� ��쿡�� �ٽ� �����Ͽ� Ŀ�ǵ������ ����� �Է����־�� ��






# URL�� ���� ��� �������� �����ϱ�
base_url_weather = "https://apihub.kma.go.kr/api/typ01/url/kma_sfctm3.php" # url�� �Һ� ����
# auth_key = "ghBQh0IXRimQUIdCFzYp4w" # url�� ���� ����- ����Ű �κ�


# �ݺ��� ����. tm1_value, tm2_value�� ���� 7���� �� 14�� ���;� ��. url_weather�� �� 7�� ������, 7�� ȣ���� �Ǿ����


# current_date = datetime.date.today()
# current_date = end_date # ���糯¥�� ���۳�¥�� ����. �̴� ���۳�¥���� ���ᳯ¥���� �ݺ��ϱ� ����.
# previous_six_date = current_date - datetime.timedelta(days=6) 
# previous_six_date = start_date


while start_date <= end_date:
    start_date_str = start_date.strftime('%Y%m%d')
    # RL�� tm1�� tm2 ���� ����
    tm1_value = start_date_str + "0600" # url�� ���� ���� - ���� �ð� ����
    tm2_value = start_date_str + "2100" # url�� ���� ���� - ���� �ð� ����

    # �ϼ��� URL
    url_weather = f"{base_url_weather}?tm1={tm1_value}&tm2={tm2_value}&stn={nearest_location[1]}&help=0&authKey={second_api_input}" # url_weather ����


    print("����URL �ϼ�")
    print(url_weather)



    # ���� ���

    weather_save_file_path = fr"C:\Users\user1\Desktop\Battery_Charging_Algorithm\API\Weather_{start_date_str}_file_pre.csv" # ����, ���� ���� ������ ���ϵ��� ��� ���糯¥)_csv�� �̸����� �����. 
   

    try:
       # ���� �ٿ�ε� �Լ� ȣ��
        download_file(url_weather, weather_save_file_path)
    except Exception as e:
        print(f"���� �ٿ�ε� �� ���� �߻�: {e}")
        sys.exit() # ��ũ��Ʈ ����. �̷��� ��쿡�� �ٽ� �����Ͽ� Ŀ�ǵ������ ����� �Է����־�� ��



    response = requests.get(url_weather)

    # ���� ���� �ڵ� Ȯ��
    print(response.status_code)

    # ���� ���� ���
    print(response.text)



    print(f"����API ȣ�� �Ϸ�")

    start_date += datetime.timedelta(days=1) 





# ���������Ͱ����� ��� �Լ�


API_LOCATION_FORECAST = pd.read_csv(r"C:\Users\user1\Desktop\Battery_Charging_Algorithm\API_FORECAST_REGION_V2.csv") # ���������� ������ ������ ��� �ִ� ����

list_east = ['������', '��ô��', '�¹��', '���ʽ�', '��籺', '���ؽ�', '����'] #  �������� ����������.  input_region_2depth_name�� ����Ʈ �� ���ԵǴ� ���ڿ��� ������ '��������' �����Ҹ� ��ȯ��.
list_west = ['ö����', 'ȭõ��', '�籸��', '������', '��õ��', 'ȫõ��', 'Ⱦ����', '���ֽ�', '��â��', '������', '������'] # �������� ����������.  input_region_2depth_name�� ����Ʈ �� ���ԵǴ� ���ڿ��� ������ '��������' �����Ҹ� ��ȯ��.

def find_nearest_obesrvatory_forecast_first(depth1_name, depth2_name): # ���浵 ���� ���� �������� ��ȯ�ϴ� �Լ�. ��ȯ�� ������������ ��󿹺������� ������ ���ϰ� ���� ��ġ�ϴ� ���� ���� ������ reg_value�� ��ȯ
    if depth2_name in list_east:# depth2_name�� �ñ������� ��������, depth2_name ������ list_east �� ���ڿ��� ��ġ�ϴ� ���� �ִٸ� 
        return '��������' # ���������� ��ȯ��. �� ��� '��������' ��󿹺������� reg_value�� ��ȯ��
    elif depth2_name in list_west: # depth2_name ������ list_west �� ���ڿ��� ��ġ�ϴ� ���� �ִٸ�
        return '��������' # �������� ��ȯ
    elif '����' in depth1_name or '���' in depth1_name or '��õ' in depth1_name: # depth1_name�� �����̳� ��⳪ ��õ�� ���ԵǾ��ִٸ�
        return '����.��õ.���' # ����.��õ.��� ��ȯ
    
    elif '����Ư����ġ��' in depth1_name: # ����Ư����ġ���� ���
        return '���ֵ�' # ���ֵ� ��󿹺������� reg_value ��ȯ

    else: # �� ���� ���
        return depth1_name # depth1_name ��ȯ

extracted_location_forecast = find_nearest_obesrvatory_forecast_first(input_region_1depth_name, input_region_2depth_name) #�Լ� ���� ����� extracted_location_forecast�� ����.  �Ʒ� ���� find_nearest_obesrvatory_forecast_second �Լ����� �Է°����� Ȱ��

def find_nearest_obesrvatory_forecast_second(extracted_location_forecast, location): # ���浵 ���� ���� ��ȯ�� ������������ ��󿹺������� ���� �����͸� ���� �ش� ��󿹺������� reg_value�� ��ȯ�ϴ� �Լ�
    matched_location = location[location['REG_NAME'] == extracted_location_forecast] # ��󿹺������Ϳ��� REG_NAME�� ������ ����extracted_location_forecast �̸��� ������ �ش� �������� reg_name ��ȯ
    if not matched_location.empty: #����� �ҷ����� �����ߴٸ�
        return matched_location.iloc[0]['REG_ID'] #REG_ID ��ȯ
    else:
        print(f"API_LOCATION_FORECAST���� {extracted_location_forecast} ���� ã�µ� ����") # �ҷ����� ������ ��� 
        return None # none�� ��ȯ
        sys.exit() # ��ũ��Ʈ ����. �̷��� ��쿡�� �ٽ� �����Ͽ� Ŀ�ǵ������ ����� �Է����־�� ��

nearest_location_forecast = find_nearest_obesrvatory_forecast_second(extracted_location_forecast, API_LOCATION_FORECAST) # �������� ���� �Է°� ���� �ִܰŸ��� ��󿹺������� ����
print(nearest_location_forecast)





def download_file(file_url, save_path): #API �޾ƿͼ� ���� �ٿ�ε� �ϴ� �Լ�
    with open(save_path, 'wb') as f: #API �޾ƿͼ� ���� �ٿ�ε� �ϴ� �Լ�
        response = requests.get(file_url) #API �޾ƿͼ� ���� �ٿ�ε� �ϴ� �Լ�
        f.write(response.content) #API �޾ƿͼ� ���� �ٿ�ε� �ϴ� �Լ�

# URL�� ���� ��� �������� �����ϱ�

base_url_forecast = "https://apihub.kma.go.kr/api/typ01/url/fct_afs_wl.php" # url�� �Һ� ����
# auth_key = "ghBQh0IXRimQUIdCFzYp4w" # url�� ���� ����, ����Ű �κ�

reg_value = nearest_location_forecast # url�� ���� ���� - ������ id �κ�

# URL�� tm1�� tm2 ���� ����
tm1_value_forecast = previous9_date_str + "0600" # url�� ���� ���� - ���� �ð� ����
tm2_value_forecast = end_date_str + "1800" # url�� ���� ���� - ���� �ð� ����
tm3_value_forecast = later_date_str + "0000" # url�� ���� ���� - ���� �ð� ����
tm4_value_forecast = later3_date_str + "0000" # url�� ���� ���� - ���� �ð� ����




# �ϼ��� URL

url_forecast = f"{base_url_forecast}?reg={reg_value}&tmfc1={tm1_value_forecast}&tmfc2={tm2_value_forecast}&tmef1={tm3_value_forecast}&tmef2={tm4_value_forecast}&mode=0&disp=0&help=1&authKey={second_api_input}"  # ȣ�� URL ���� 




print("URL Ȯ�οϷ�")
print(url_forecast)



# ���� ���
forecast_save_file_path = fr"C:\Users\user1\Desktop\Battery_Charging_Algorithm\API\forecast_{end_date_str}_pre.csv" # ����, ���� ���� ������ ���ϵ��� ��� ���糯¥)_csv�� �̸����� �����. 



try:
    # ���� �ٿ�ε� �Լ� ȣ��
    download_file(url_forecast, forecast_save_file_path)  # ���յ� url�� ������ ���ϰ�θ� Ȱ���� download_file �Լ� ������ ���������� api ���� ����

except Exception as e:
        print(f"���� �ٿ�ε� �� ���� �߻�: {e}")
        sys.exit() # ��ũ��Ʈ ����. �̷��� ��쿡�� �ٽ� �����Ͽ� Ŀ�ǵ������ ����� �Է����־�� �� 


response = requests.get(url_forecast) # �ϼ���  URL�� request.get �޼��带 ������ api������ ������

# ���� ���� �ڵ� Ȯ��
print(response.status_code) # 200�� ����. 

# ���� ���� ���
print(response.text)


print(f"����API ȣ�� �Ϸ�")

print("API���� 1 ��ũ��Ʈ ����")


# os.system("pause")


