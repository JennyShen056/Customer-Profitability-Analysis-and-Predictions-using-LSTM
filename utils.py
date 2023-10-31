ts = pd.read_csv("/Users/jennyshen/Desktop/retail_fianl/ts.csv", index_col=0)
ts = ts.dropna(subset=["2010-01-01"])
ts.to_csv("/Users/jennyshen/Desktop/retail_fianl/ts_updated.csv")
from torch.nn.functional import one_hot
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from models import RNN, LSTMModel
from torch import nn


def create_inout_sequences(input_data, tw, input_data_no_one):
    inout_seq = []
    # L = len(input_data)
    L = input_data.shape[1]
    for i in range(L - tw):
        train_seq = input_data[:, i : i + tw, :]
        train_label = input_data_no_one[:, i + tw : i + tw + 1]
        inout_seq.append((train_seq, train_label))
    return inout_seq


def data_preprocessing(input_data, tw=5):
    input_data_one_hot = one_hot(torch.Tensor(input_data).long())
    print(input_data_one_hot.shape)
    # input_data = torch.Tensor(input_data.values)
    # print(input_data.shape)
    # input_data = input_data.type(torch.LongTensor)
    inout_seq = create_inout_sequences(input_data_one_hot, tw, input_data)
    print(inout_seq[0][0].shape, inout_seq[0][1].shape)
    return inout_seq


def load_data(pth):
    data = pd.read_csv(pth, header=0, index_col=0)
    data = data - 1
    # print(data.shape)
    # print(data.head(10))
    return data


if __name__ == "__main__":
    tw = 5
    test_data_size = 1
    pth = "ts_updated.csv"
    data = load_data(pth)
    print(data.shape)
    data = data.values
    train_data = data[:, 0 : data.shape[1] - test_data_size]
    test_data = data[:, data.shape[1] - test_data_size - tw : data.shape[1]]
    print(test_data.shape)
    print(train_data.shape)
    data_onehot = data_preprocessing(train_data)

    data_onehot_test = data_preprocessing(test_data)

    input_size = data_onehot[0][0].shape[2]
    hidden_size = 32
    num_layers = 2
    output_size = data_onehot[0][0].shape[2]
    # rnn_model = LSTM(input_size=4, hidden_layer_size=16, output_size=4)
    rnn_model = LSTMModel()

    loss_function = nn.CrossEntropyLoss()
    # loss_function = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(rnn_model.parameters(), lr=0.001)

    epochs = 25
    loss = []
    ind = []

    for i in range(epochs):
        for seq, labels in data_onehot:
            optimizer.zero_grad()

            rnn_model.hidden_cell = (
                torch.zeros(1, 1, rnn_model.hidden_layer_size),
                torch.zeros(1, 1, rnn_model.hidden_layer_size),
            )

            y_pred = rnn_model(seq.float())
            labels = torch.tensor(labels).view(-1)

            single_loss = loss_function(y_pred, labels.long())
            single_loss.backward()
            optimizer.step()

        if i % 2 == 1:
            print(f"epoch: {i:3} loss: {single_loss.item():10.8f}")
            ind.append(i)
            loss.append(single_loss.item())

    plt.plot(ind, loss)
    plt.title("Training loss of LSTM")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.xticks()
    #     plt.legend()
    plt.savefig("Training loss of LSTM.pdf")

    print(f"epoch: {i:3} loss: {single_loss.item():10.10f}")

    rnn_model.eval()
    test_pred = []
    test_gt = []

    for seq, labels in data_onehot_test:
        rnn_model.hidden_cell = (
            torch.zeros(1, 1, rnn_model.hidden_layer_size),
            torch.zeros(1, 1, rnn_model.hidden_layer_size),
        )

        y_pred = rnn_model(seq.float())
        test_pred.append(y_pred)
        test_gt.append(labels)
    print(test_pred)

    test_pred = torch.stack(test_pred, dim=1)
    test_gt = np.stack(test_gt, axis=1).squeeze()
    print(test_pred.shape)
    test_pred = torch.argmax(test_pred, dim=2)
    print(test_pred.shape)

    from sklearn.metrics import accuracy_score

    print(accuracy_score(y_true=test_gt, y_pred=test_pred.view(-1).numpy()))

    # for i in range(test_data):
    #     seq = torch.FloatTensor(test_inputs[-train_window:])
    #     with torch.no_grad():
    #         rnn_model.hidden = (torch.zeros(1, 1, model.hidden_layer_size),
    #                         torch.zeros(1, 1, model.hidden_layer_size))
    #         test_inputs.append(model(seq).item())

from sklearn.metrics import confusion_matrix

y_true = test_gt
y_pred = test_pred.view(-1).numpy()
cm = confusion_matrix(y_true + 1, y_pred + 1)
print(np.unique(y_pred + 1))
print(cm)

sns.set()
f, ax = plt.subplots()
sns.heatmap(
    cm,
    annot=True,
    fmt=".20g",
    cmap=plt.cm.Blues,
    ax=ax,
    xticklabels=[1, 2, 3],
    yticklabels=[1, 2, 3],
)  # 画热力图

ax.set_title("confusion matrix")  # 标题
ax.set_xlabel("predict")  # x轴
ax.set_ylabel("true")  # y轴
plt.savefig("confusion matrix.pdf")
