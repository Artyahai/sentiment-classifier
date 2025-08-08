from dataload import train_dataloader, test_dataloader
from classifier import Classifier
import torch
from dataload import tokenizer
device = 'cuda' if torch.cuda.is_available() else 'cpu'

d_model= 512
vocab_size= len(tokenizer)
seq_len= 100
h= 8
d_ff= 2048
num_classes= 2
num_layers= 6
dropout= 0.1


classifier_model = Classifier(d_model=d_model, vocab_size= vocab_size, seq_len=seq_len, h=h, d_ff=d_ff, num_classes=num_classes, num_layers=num_layers, dropout=dropout)
classifier_model.to(device)
optimizer = torch.optim.AdamW(classifier_model.parameters(), lr=0.001, weight_decay=0.01)
classifier_model.train()
num_epochs = 10
for epoch in range(num_epochs):
    for batch in iter(train_dataloader):
        input_ids, attention_masks, labels = [item.to(device) for item in batch]
        optimizer.zero_grad()
        outputs = classifier_model(input_ids)
        loss = torch.nn.functional.cross_entropy(outputs, labels)
        loss.backward()
        optimizer.step()
    
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")
    
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")
total = 0 
correct = 0 
classifier_model.eval()
with torch.no_grad():
    for batch in iter(test_dataloader):
        input_ids, attention_masks, labels = [item.to(device) for item in batch]
        outputs = classifier_model(input_ids)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f"Accuracy on test set: {100 * correct / total}%")


torch.save(classifier_model.state_dict(), 'classifier_model.pth')