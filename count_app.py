import streamlit as st
import torch
import numpy as np
from functools import partial

alphabet = 'NACGT'
dna2int = {a: i for a, i in zip(alphabet, range(1, 6))}
int2dna = {i: a for a, i in zip(alphabet, range(1, 6))}
dna2int.update({"pad": 0})
int2dna.update({0: "<pad>"})

intseq_to_dnaseq = partial(map, int2dna.get)
dnaseq_to_intseq = partial(map, dna2int.get)

max_len = 128
multiply_factor = 64
LSTM_HIDDEN = 64
LSTM_LAYER = 2
INPUT_SIZE = 26
OUTPUT_SIZE = 1

class CpGPredictor(torch.nn.Module):
    ''' Simple model that uses a LSTM to count the number of CpGs in a sequence '''
    def __init__(self):
        super(CpGPredictor, self).__init__()
        # TODO complete model, you are free to add whatever layers you need here
        # We do need a lstm and a classifier layer here but you are free to implement them in your way
        self.lstm = torch.nn.LSTM(INPUT_SIZE, LSTM_HIDDEN, LSTM_LAYER, batch_first=True)
        self.classifier = torch.nn.Linear(LSTM_HIDDEN, OUTPUT_SIZE)

    def forward(self, x):
        # TODO complete forward function
        h0 = torch.zeros(LSTM_LAYER, x.size(0), LSTM_HIDDEN).to(x.device)
        c0 = torch.zeros(LSTM_LAYER, x.size(0), LSTM_HIDDEN).to(x.device)
        lstm_out, _  = self.lstm(x, (h0, c0))
        output = self.classifier(lstm_out[:, -1, :])
        logits = output.squeeze(1)
        return logits

model = torch.load('model_variable_length.pth')
model.eval()

def main():
    st.title("Count CG")
    query = st.text_input("Enter your input (in capital letters, any length upto 128 chars): ")
    if st.button("Count"):
        query_int = list(dnaseq_to_intseq(query))
        padded_query = [0]*max_len
        padded_query[:len(query_int)] = query_int
        one_hot_seq = np.zeros((len(padded_query), 26))
        for i, char in enumerate(padded_query):
            if i == len(padded_query)-1 or padded_query[i] == 0:
                break
            one_hot_seq[i, 5*(char-1)+padded_query[i+1]] = 1

        query_tensor = torch.tensor([one_hot_seq], dtype=torch.float32)

        with torch.no_grad():
            output = model(query_tensor)

        st.write(f"No. of CG's are: {int(np.round(output.item()*multiply_factor))}")


if __name__ == "__main__":
    main()