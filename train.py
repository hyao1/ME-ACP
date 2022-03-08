import torch
import torch.nn as nn
import torch.optim as opt
from torch.optim import lr_scheduler
from evaluate import evaluate
import model
torch.set_default_dtype(torch.float64)


def train(device, train_dataset, test_dataset, batch_size, epochs, lr):
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=len(test_dataset),
                                              pin_memory=True,
                                              shuffle=False,
                                              num_workers=4)
    loss_function = nn.BCELoss()

    net = model.MeACP().to(device)
    optimizer = opt.Adam(net.parameters(), lr=lr)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[5, 15, 25], gamma=0.1)
    max_acc = 0.0
    max_epoch = 0
    all_evaluation = []

    for epoch in range(epochs):
        net.train()
        for step, data in enumerate(train_loader, start=0):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = net(inputs)

            predict = torch.ones(logits.shape[0]).to(device)
            for i in range(logits.shape[0]):
                if logits[i] < 0.5:
                    predict[i] = 0

            loss = loss_function(logits, labels)
            loss.backward()
            optimizer.step()
        scheduler.step()

        with torch.no_grad():
            net.eval()
            for test_data in test_loader:
                inputs, labels = test_data
                inputs = inputs.to(device)
                labels = labels.to(device)

                logits = net(inputs)
                predict = torch.ones(logits.shape[0]).to(device)
                for i in range(logits.shape[0]):
                    if logits[i] < 0.5:
                        predict[i] = 0.0
                evaluation = evaluate(labels.detach().cpu().numpy(), predict.detach().cpu().numpy(),
                                      logits.detach().cpu().numpy())
                evaluation = [evaluation['acc'], evaluation['sen'], evaluation['spec'], evaluation["precision"], evaluation['f1_score'],
                              evaluation['mcc'], evaluation['auc']]

                all_evaluation.append(evaluation)

                if evaluation[0] > max_acc:
                    max_acc = evaluation[0]
                    max_epoch = epoch
    return max_epoch, all_evaluation[max_epoch]
