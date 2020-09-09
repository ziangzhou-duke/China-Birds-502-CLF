import torch
from sklearn.metrics import auc, roc_curve, roc_auc_score

def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)


# def AUC(output, target):
#     with torch.no_grad():
#         output_tmp = torch.softmax(output, dim=1)
#         output_max = torch.Tensor([max(x) for x in output_tmp]).cpu().numpy()
# #         pred = torch.argmax(output_tmp, dim=1).cpu().numpy()
# #         print(pred)
#         assert output_max.shape[0] == len(target)
#         fpr, tpr, thresholds = roc_curve(target.cpu().numpy(), output_max)
#     return auc(fpr, tpr)

def AUC(output, target):
    with torch.no_grad():
        output_tmp = torch.softmax(output, dim=1)
        output_max = torch.Tensor([max(x) for x in output_tmp]).cpu().numpy()
#         pred = torch.argmax(output_tmp, dim=1).cpu().numpy()
#         print(pred)
#         assert output_max.shape[0] == len(target)
#         fpr, tpr, thresholds = roc_curve(target.cpu().numpy(), output_max)
    return roc_auc_score(target.cpu().numpy(), output_max)