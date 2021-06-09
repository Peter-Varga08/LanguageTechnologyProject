from torch.utils.data import DataLoader
from transformers import SqueezeBertConfig, SqueezeBertForMultipleChoice, SqueezeBertTokenizer
import torch
import time
from dataloader_movies import movie_dataset


def evaluate(model, loader):
    model.eval()
    with torch.no_grad():
        correct = 0.0
        total = 0.0
        for batch in loader:
            data, labels = batch
            out = model(input_ids=data, labels=labels)
            preds = out.logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.numel()

    print(correct, total)
    model.train()
    return correct/total


def train(model, train_loader, valid_loader, test_loader, epochs):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    for epoch in range(epochs):
        start_time = time.time()
        total_loss = 0.0
        total_tokens = 0.0
        for i, batch in enumerate(train_loader):

            if i % 5 == 0 and i > 0:
                tps = total_tokens/ (time.time()-start_time)
                print("[Epoch %d, Iter %d] loss: %.4f, toks/sec: %d" % (epoch, i, total_loss/5, tps))
                start_time = time.time()
                total_loss = 0.0
                total_tokens = 0.0

            data, labels = batch
            optimizer.zero_grad()
            out = model(input_ids=data, labels=labels)
            out.loss.backward()
            optimizer.step()
            total_loss += out.loss.item()
            total_tokens += data.numel()

        acc = evaluate(model, valid_loader)
        print("[Epoch %d] Acc (valid): %.4f" % (epoch, acc))
        acc = evaluate(model, test_loader)
        print("[Epoch %d] Acc (test): %.4f" % (epoch, acc))
        start_time = time.time() # so tps stays consistent

    print("############## END OF TRAINING ##############")
    acc = evaluate(model, valid_loader)
    print("Final Acc (valid): %.4f" % (acc))
    acc = evaluate(model, test_loader)
    print("Final Acc (test): %.4f" % (acc))


pretrained = 'squeezebert/squeezebert-uncased'
tokenizer = SqueezeBertTokenizer.from_pretrained(pretrained)
tokenizer.do_basic_tokenize = False
train_dataset = movie_dataset(tokenizer)
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True,collate_fn=padding_collate_fn,)
configuration = SqueezeBertConfig.from_pretrained(pretrained)
configuration.num_labels = 16 #movies_df['Genre'].nunique()
configuration.num_hidden_layers = 1
configuration.num_attention_heads = 1
configuration.output_attentions = True
model = SqueezeBertForMultipleChoice(configuration)
train(model, train_dataloader, train_dataloader, train_dataloader, 16)