from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles

from pydantic import BaseModel, Field

from fastapi.responses import HTMLResponse
import uvicorn



import requests
import os
import random
import pickle

import torch
import torch.nn as nn



# index_character  возвращает  индекс  по токену  либо токен  по индексу
def index_character( data: str | int):
    try:
        if isinstance(data, int):
            if data in  range(0, size_character):
                return set_character[data]
        if isinstance(data, str):
            if data in set_character:
                return set_character.index(data)
        return print ('некорркектный ввод')
    except:
          return print ('ошибка ввода')
    





# num_embeddings - размер словаря, embedding_dim -размер ембеддинга, n_layers - число слоев gru

class RNN(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, n_layers=1):
        super().__init__()
    
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        # self.output_size = output_size
        self.n_layers = n_layers

        
        self.emb = nn.Embedding(num_embeddings, embedding_dim)
        self.gru = nn.GRU(embedding_dim, embedding_dim, n_layers,batch_first=True)
        self.decoder = nn.Linear(embedding_dim, num_embeddings)

    def forward (self,input, hidden=None):

        emb = self.emb(input)  
        gru = self.gru(emb, hidden)
        proba = self.decoder(gru[0])  
# proba тензор с логитами пример; gpu[1]  тензор  hidden state
        return proba, gru[1] 

                                 
    def init_hidden(self,batch_size):
        return torch.zeros(self.n_layers, batch_size, self.embedding_dim)
    



# Инференс на cpu!
#set_character -  словарь токенов
# num_embeddings - размер  словаря
# embedding_dim = размер  вектора hidden  state
# n_layers  количество слоев RNN
# param_model =(num_embeddings, embedding_dim, n_layers)
             
#  пример : name_model = 'model_2' строка  с именем модели без расширения
def load_model(name_model : str):
    global set_character , size_character

    path = os.path.join(os.getcwd(),'model', f'{name_model}.pkl')
    if os.path.exists(path):
        
        with open(path,'rb') as file:
            set_character, param_model = pickle.load(file)
    else:
        raise HTTPException(status_code=404, detail="File 'pkl' not found")        

    size_character=len(set_character) 


    model =  RNN(*param_model)

    path_model =os.path.join(os.getcwd(),'model',f'{name_model}.pth')

    if os.path.exists(path_model):
        model.load_state_dict(torch.load(path_model, map_location=torch.device('cpu'),weights_only=True))
        return model
    else:
        raise HTTPException(status_code=404, detail="File 'pth' not found")

# функция записи фразы  в output
def write_phrase( phrase):
    file = os.listdir(os.path.join(os.getcwd(),'output'))
    file =[x for x in file if x.endswith('.txt')]


    if file :
        number= max([int(x.replace('.txt','').split('_')[1]) for x in file])
        new_name = f'text_{number+1}.txt'
    else: 
        new_name = f'text_1.txt'

    with open(os.path.join(os.getcwd(),'output', new_name), 'w',encoding='utf-8') as file:
          file.write( phrase)
    return   new_name 

# evaluete - функция для генрациии фразы
def evaluate(model, prime_str='a', predict_len=500, temperature=0.5):
    prime_str=prime_str.lower()
    model.eval()
 # Инициализация  тензора для генерации
    batch_size=1
    hidden = model.init_hidden(batch_size)
    prime_text=[index_character('<sos>')] + [index_character(x) for x in prime_str]
    prime_tensor =torch.tensor( prime_text, dtype=torch.int64).reshape(batch_size,-1)
    hidden = model.init_hidden(batch_size)

    
    for _ in range(0, predict_len): #  цикл для генерации  символов в количестве predict_len  
        
        with torch.no_grad():
            proba, hidden = model(prime_tensor,hidden) # тушим градиенты и делаем предсказание
 #Блок выбора  индекса для предсказанного символа        
        proba = proba / temperature 
        proba = torch.softmax(proba [:,-1,:-1],dim=1).squeeze() # -1 в dim=2- убираем из  предсказания токен <sos>
        proba= torch.multinomial(proba,1).item() # вобор индекса согласно вероятному расперделению
        
# Формироем  тензор для предсказания, присоеденив к нему индекс  предсказанного токена
        prime_tensor = torch.cat((prime_tensor,torch.tensor([proba],dtype=torch.int64).reshape(batch_size,-1)), dim=1)

    # return proba, hidden, prime_tensor

# Формируем  строку  с предсказанной фразой
    phrase =[index_character(x.item()) for x in  prime_tensor.squeeze()]
    phrase=''.join(phrase[1:]) #  [1:] - убираем  токен <sos>

    name = write_phrase(phrase)

    
    return  phrase, name       

    
# name_model_2 = 'model_2'
# model_train_2 = load_model(name_model_2)
# evaluate(model_train_2, 'привет')


app = FastAPI()
path_output = os.path.join(os.getcwd(),'output')
app.mount('/rnn/output', StaticFiles(directory=path_output))




class Item(BaseModel):
    name_model: str ='model_1' # строка со значением по умолчанию
    prime_str: str = 'a'  # строка с значением по умолчанию
    predict_len: int = Field(default=200, le=10000)  # ограничение на максимальное значение 10000 и значение по умолчанию
    temperature: float = Field(default=0.5, ge=0.1, le=2.0)  # ограничение от 0.1 до 2.0 и значение по умолчанию

@app.post('/rnn/inferense/')
def model_inference(item: Item):
    model = load_model(item.name_model)
    phrase, name = evaluate(model, item.prime_str, item.predict_len, item.temperature)

    save_url = f'http://127.0.0.1:8000/rnn/output/{name}' 
    save_path =os.path.join(os.getcwd(),'output',name)
    return {'save_url':save_url, 'save_path': save_path }


if __name__=='__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)



