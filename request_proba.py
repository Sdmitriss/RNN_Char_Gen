import requests
data= {
    'name_model':'model_2',
    'prime_str':  'привет',
     'predict_len': 300,
     'temperature': 0.5 
}

response = requests.post("http://127.0.0.1:8000/rnn/inferense/", json=data)
print(response.json())


