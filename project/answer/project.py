import pandas as pd
import numpy as np
import nltk
import os
import torch

from sklearn.utils import shuffle
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from tensorflow.keras.preprocessing.text import one_hot, Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Embedding, Input, LSTM, Conv1D, MaxPool1D, Bidirectional, Dropout, BatchNormalization
from sklearn.metrics import accuracy_score ,confusion_matrix, classification_report, ConfusionMatrixDisplay
from transformers import BertForSequenceClassification, BertTokenizer
from IPython.display import display, clear_output
from sklearn.metrics import accuracy_score
from tqdm.notebook import tqdm
from torch.utils.data import Dataset,DataLoader
from torch.nn.utils.rnn import pad_sequence



nltk.download('punkt')
nltk.download("stopwords")

def preprocess(text):
    t = word_tokenize(text)
    filtered_t = [token.lower() for token in t if token.lower() not in stop_words and len(token) > 3 and token.isalpha()]
    return filtered_t
class news_dataset(Dataset):
    def __init__(self, mode, tokenizer):
        assert mode in ['train', 'val', 'test']
        self.mode = mode
        self.df = pd.read_csv('input/' + mode + '.tsv', sep='\t').fillna("")
        self.len = len(self.df)
        self.tokenizer = tokenizer  

    def __getitem__(self, idx):
        if self.mode == 'test':
            statement, label = self.df.iloc[idx, :].values
            label_tensor = torch.tensor(label)
        else:
            statement, label = self.df.iloc[idx, :].values
            label_tensor = torch.tensor(label)
            
        word_pieces = ['[CLS]']
        statement = self.tokenizer.tokenize(statement)
        word_pieces += statement + ['[SEP]']
        len_st = len(word_pieces)
        ids = self.tokenizer.convert_tokens_to_ids(word_pieces)
        tokens_tensor = torch.tensor(ids)
        
        segments_tensor = torch.tensor([0] * len_st, dtype=torch.long)
        
        return (tokens_tensor, segments_tensor, label_tensor)
    
    def __len__(self):
        return self.len
def mini_batch(samples):
    tokens_tensors = [s[0] for s in samples]
    segments_tensors = [s[1] for s in samples]
    
    
    if samples[0][2] is not None:
        label_ids = torch.stack([s[2] for s in samples])
    else:
        label_ids = None
    
    
    tokens_tensors = pad_sequence(tokens_tensors, batch_first=True)
    segments_tensors = pad_sequence(segments_tensors, batch_first=True)
    
    
    masks_tensors = torch.zeros(tokens_tensors.shape, dtype=torch.long)
    masks_tensors = masks_tensors.masked_fill(tokens_tensors != 0, 1)
    
    return tokens_tensors, segments_tensors, masks_tensors, label_ids

#input
true = pd.read_csv('input/True.csv')
fake = pd.read_csv('input/Fake.csv')

#markers and mixing true and false
true['true'] = 1
fake['true'] = 0
df = pd.concat([true, fake])

#shuffle
df = df.iloc[:,[0, -1]]
df = shuffle(df).reset_index(drop=True)

#spliting
train_val = df.sample(frac = 0.8)
test = df.drop(train_val.index)
train = train_val.sample(frac = 0.8)
val = train_val.drop(train.index)
train = train.reset_index(drop=True)
val = val.reset_index(drop=True)
test = test.reset_index(drop=True)
df = pd.concat([train, val, test])

#Cleaning
stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use'])
df['clean'] = df['title'].apply(preprocess)

#colleting all words
l = []
for i in df.clean:
    for j in i:
        l.append(j)
words = len(list(set(l)))

#Tokenization and padding
tokenizer = Tokenizer(num_words = words)
tokenizer.fit_on_texts(train['title'])
train_seq = tokenizer.texts_to_sequences(train['title'])
val_seq = tokenizer.texts_to_sequences(val['title'])
test_seq = tokenizer.texts_to_sequences(test['title'])

padded_train = pad_sequences(train_seq,maxlen = 42, padding = 'post', truncating = 'post')
padded_val = pad_sequences(val_seq,maxlen = 42, padding = 'post', truncating = 'post')
padded_test = pad_sequences(test_seq,maxlen = 42, padding = 'post', truncating = 'post')

total_words = 100000
embedding_vector_features=40
y_train = np.asarray(train['true'])
y_val = np.asarray(val['true'])

#LSTM
model_L=Sequential()
model_L.add(Embedding(input_dim=total_words,output_dim=embedding_vector_features,input_length=42))
model_L.add(Dropout(0.3))
model_L.add(LSTM(100))
model_L.add(Dropout(0.3))
model_L.add(Dense(1,activation='sigmoid'))
model_L.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model_L.fit(padded_train, y_train, batch_size = 64, validation_data=(padded_val, y_val), epochs = 3)
prediction_L=(model_L.predict(padded_test) > 0.5).astype("int32")

#BiLSTM
model_bi=Sequential()
model_bi.add(Embedding(total_words,embedding_vector_features,input_length=42))
model_bi.add(Dropout(0.3))
model_bi.add(Bidirectional(LSTM(100)))
model_bi.add(Dropout(0.3))
model_bi.add(Dense(1,activation='sigmoid'))
model_bi.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model_bi.fit(padded_train, y_train, batch_size = 64, validation_data=(padded_val, y_val), epochs = 3)
prediction_bi=(model_bi.predict(padded_test) > 0.5).astype("int32")

#CNN
model_C=Sequential()
model_C.add(Embedding(total_words,embedding_vector_features,input_length=42))
model_C.add(Dropout(0.3))
model_C.add(Conv1D(32, 5, activation='relu'))
model_C.add(MaxPool1D())
model_C.add(Conv1D(32, 5, activation='relu'))
model_C.add(MaxPool1D())
model_C.add(Bidirectional(LSTM(100)))
model_C.add(Dropout(0.3))
model_C.add(Dense(1,activation='sigmoid'))
model_C.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model_C.fit(padded_train, y_train, batch_size = 64, validation_data=(padded_val, y_val), epochs = 3)
prediction_C=(model_C.predict(padded_test) > 0.5).astype("int32")

train.to_csv('input/train.tsv', sep='\t', index=False)
val.to_csv('input/val.tsv', sep='\t', index=False)
test.to_csv('input/test.tsv', sep='\t', index=False)

#BERT
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)

train_set = news_dataset('train', tokenizer=tokenizer)
val_set = news_dataset('val', tokenizer=tokenizer)
test_set = news_dataset('test', tokenizer=tokenizer)

BATCH_SIZE = 16
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, collate_fn=mini_batch)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, collate_fn=mini_batch)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE,collate_fn=mini_batch)

data = next(iter(train_loader))

tokens_tensors, segments_tensors, masks_tensors, label_ids = data

NUM_LABELS = 2
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=NUM_LABELS)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device:", device)
model = model.to(device)
file_path='input/fake_news_detection.pth'
if os.path.isfile(file_path):
    print("trained model found")
    model = torch.load('input/fake_news_detection.pth')
    model = model.to(device)
else:
    print('Training model')
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    NUM_EPOCHS = 3

    for epoch in range(NUM_EPOCHS):
        train_loss = 0.0
        train_acc = 0.0

        loop = tqdm(train_loader)
        for batch_idx, data in enumerate(loop):
            tokens_tensors, segments_tensors, masks_tensors, labels = [t.to(device) for t in data]

        
            optimizer.zero_grad()
            
            outputs = model(input_ids=tokens_tensors, 
                            token_type_ids=segments_tensors, 
                            attention_mask=masks_tensors, 
                            labels=labels)

            loss = outputs[0]
            loss.backward()
            optimizer.step()

            logits = outputs[1]
            _, pred = torch.max(logits.data, 1)
            train_acc = accuracy_score(pred.cpu().tolist() , labels.cpu().tolist())

            
            train_loss += loss.item()

            loop.set_description(f"Epoch [{epoch+1}/{NUM_EPOCHS}]")
            loop.set_postfix(acc = train_acc, loss = train_loss)
    
    torch.save(model, 'input/fake_news_detection.pth')
    print('Model saved!')

true_b=[]
predictions_b=[]
with torch.no_grad():
    model.eval()
    for data in test_loader:
        if next(model.parameters()).is_cuda:
            data = [t.to(device) for t in data if t is not None]
            
        tokens_tensors, segments_tensors, masks_tensors = data[:3]
        test_outputs = model(input_ids=tokens_tensors, 
                    token_type_ids=segments_tensors, 
                    attention_mask=masks_tensors)

        logits = test_outputs[0]
        _, pred = torch.max(logits.data, 1)

        labels = data[3]
        true_b.extend(labels.cpu().tolist())
        predictions_b.extend(pred.cpu().tolist())    
#Getting The Accuracy
y_test = np.asarray(test['true'])
accuracy_L = accuracy_score(list(y_test), prediction_L)
accuracy_bi = accuracy_score(list(y_test), prediction_bi)
accuracy_C = accuracy_score(list(y_test), prediction_C)
accuracy_b=accuracy_score(predictions_b,true_b)
print("LSTM Model Accuracy : ", accuracy_L)
print("BiLSTM Model Accuracy : ", accuracy_bi)
print("CNN Model Accuracy : ", accuracy_C)
print("Bert Model Accuracy : ", accuracy_b)

#save
df = pd.DataFrame({"pred_label": predictions_b})
df_pred = pd.concat([test_set.df.loc[:, ['title']], test_set.df.loc[:, ['true']], df.loc[:, 'pred_label']], axis=1)
df_pred.to_csv('output/trained.csv', index=False)


