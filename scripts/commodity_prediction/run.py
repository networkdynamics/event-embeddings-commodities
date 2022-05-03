import random

import torch
import tqdm

TEACHER_FORCING_RATIO = 0.5
MAX_EPOCHS = 10000
EARLY_STOPPING_PATIENCE = 25

class AttnDecoderRNN(torch.nn.Module):
    def __init__(self, embedding_size, hidden_size, dropout_p=0.1):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.dropout_p = dropout_p

        self.attn = torch.nn.Linear(embedding_size + hidden_size + 1, 1)
        self.attn_combine = torch.nn.Linear(embedding_size + 1, hidden_size)
        self.dropout = torch.nn.Dropout(self.dropout_p)
        self.gru = torch.nn.GRU(self.hidden_size, self.hidden_size)
        self.out = torch.nn.Linear(self.hidden_size, 1)

    def forward(self, input, hidden, encoder_outputs):

        attn_weights = self.attn(torch.cat((encoder_outputs, hidden, input), 1))
        attn_weights = torch.functional.softmax(attn_weights, dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((input[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = torch.functional.relu(output)
        output, hidden = self.gru(output, hidden)

        output = torch.functional.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def init_hidden(self, device):
        return torch.zeros(1, 1, self.hidden_size, device=device)

def run_model(encoder_outputs, target_tensor, decoder, optimizer, criterion, device):
    
    optimizer.zero_grad()

    target_length = target_tensor.size(0)

    loss = 0

    decoder_input = torch.tensor([0], device=device)
    decoder_hidden = decoder.init_hidden()

    use_teacher_forcing = random.random() < TEACHER_FORCING_RATIO

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for idx in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs[idx])
            loss += criterion(decoder_output, target_tensor[idx])
            decoder_input = target_tensor[idx]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for idx in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs[idx])
            decoder_input = decoder_output.detach() # detach from history as input

            loss += criterion(decoder_output, target_tensor[idx])

    loss.backward()
    optimizer.step()

    return loss.item() / target_length

def train(model, train_data, val_data, device, checkpoint_path, resume):

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    criterion = torch.nn.MSELoss()
    model.train()

    min_val_loss = 999999
    epochs_since_best = 0

    if resume:
        model.load_state_dict(torch.load(checkpoint_path))

    for epoch in range(MAX_EPOCHS):

        # train on training set
        train_loss = 0
        progress_bar_data = tqdm.tqdm(enumerate(train_data), total=len(train_data))
        for batch_idx, batch in progress_bar_data:
            encoder_outputs = batch['encoder_outputs'].to(device)
            targets = batch['targets'].to(device)
            
            batch_loss = train_model(encoder_outputs, targets, model, optimizer, criterion, device)
            progress_bar_data.set_description(f"Current Loss: {batch_loss:.4f}")
            train_loss += batch_loss
        
        train_loss /= len(train_data)

        # trail on validation set
        val_loss = 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_data):
                encoder_outputs = batch['encoder_outputs'].to(device)
                targets = batch['targets'].to(device)

                batch_loss = train_model(encoder_outputs, targets, model, optimizer, criterion, device)
                val_loss += batch_loss

        val_loss /= len(val_data)

        # potentially update learning rate
        scheduler.step(val_loss)

        # save best model so far
        if val_loss < min_val_loss:
            # checkpoint model
            torch.save(model.state_dict(), checkpoint_path)
            min_val_loss = val_loss
            epochs_since_best = 0
        else:
            epochs_since_best += 1

        # early stopping
        if epochs_since_best > EARLY_STOPPING_PATIENCE:
            break

        print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

def main():
    pass

if __name__ == '__main__':
    main()