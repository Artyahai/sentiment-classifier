from dataload import tokenizer
from classifier import Classifier
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = Classifier(d_model=512, vocab_size=30522, seq_len=100, h=8, d_ff=2048,
                   num_classes=2, num_layers=6, dropout=0.1)
model.load_state_dict(torch.load('classifier_model.pth', map_location=device))
model.to(device)
model.eval()
while True:
    sentence = input('Test a model: ')
    encoding = tokenizer(
        sentence, 
        return_tensors='pt',
        padding = 'max_length',
        truncation=True,
        max_length = 100
    )
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    if sentence == 'exit':
        break
    with torch.no_grad():
        outputs = model(input_ids)
        _, predicted = torch.max(outputs, dim=1)
    print(f'Predicted class: {predicted.item()}')

