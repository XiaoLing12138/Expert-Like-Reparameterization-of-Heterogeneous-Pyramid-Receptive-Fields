import torch
import torch.nn.functional as F


def assemble_labels(y_true, y_pred, label, out):
    if y_true == None:
        y_true = label
        y_pred = out
    else:
        y_true = torch.cat((y_true, label), 0)
        y_pred = torch.cat((y_pred, out))

    return y_true, y_pred


def get_ece(logits, labels, n_bins=15):
    # This function is based on https://github.com/gpleiss/temperature_scaling
    softmaxes = F.softmax(logits, dim=1)
    confidences, predictions = torch.max(softmaxes, 1)
    accuracies = predictions.eq(labels)
    ece = torch.zeros(1, device=logits.device)

    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Calculated |confidence - accuracy| in each bin
        in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
        prop_in_bin = in_bin.float().mean()  # weight of current bin

        if prop_in_bin.item() > 0:
            accuracy_in_bin = accuracies[in_bin].float().mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return ece


def get_cece(logits, labels, n_bins=15):
    n_classes = logits.shape[1]
    classwise_ece = []
    for i in range(n_classes):
        classwise_logits = [logits[x] for x in range(len(labels)) if labels[x] == i]
        classwise_labels = [x for x in labels if x == i]

        classwise_logits = torch.stack(classwise_logits).cuda()
        classwise_labels = torch.stack(classwise_labels).cuda()

        softmaxes = F.softmax(classwise_logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(classwise_labels)
        ece = torch.zeros(1, device=logits.device)

        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()  # weight of current bin

            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        classwise_ece.append(ece)

    classwise_ece = torch.mean(torch.tensor(classwise_ece))
    return classwise_ece


def get_mce(logits, labels, n_bins=15):
    # This function is based on https://github.com/gpleiss/temperature_scaling
    softmaxes = F.softmax(logits, dim=1)
    confidences, predictions = torch.max(softmaxes, 1)
    accuracies = predictions.eq(labels)
    mce = torch.zeros(1, device=logits.device)

    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Calculated |confidence - accuracy| in each bin
        in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
        prop_in_bin = in_bin.float().mean()

        if prop_in_bin.item() > 0:
            accuracy_in_bin = accuracies[in_bin].float().mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            error = torch.abs(avg_confidence_in_bin - accuracy_in_bin)
            if error > mce:
                mce = error

    return mce


def get_bs(logits, labels):
    softmaxes = F.softmax(logits, dim=1)
    bs_score = 0.0
    for i in range(len(labels)):
        for j in range(len(softmaxes[0])):
            if labels[i] == j:
                bs_score += (softmaxes[i][j] - 1) ** 2
            else:
                bs_score += (softmaxes[i][j] - 0) ** 2

    bs_score /= len(labels)
    return bs_score


def classify_report(result):
    result = result.split()
    x = result.index('accuracy')
    n = (x + 1) // 5 - 1
    tag = result[:4]
    m = {}
    for i in range(n):
        start = result.index(str(i)) + 1
        for j in range(4):
            m["grade" + str(i) + "_" + tag[j]] = result[start + j]
    m[result[x]] = result[x + 1]
    m_x = result.index("macro")
    for i in range(3):
        m[result[m_x] + "_" + tag[i]] = result[m_x + 2 + i]
    w_x = result.index("weighted")
    for i in range(3):
        m[result[w_x] + "_" + tag[i]] = result[w_x + 2 + i]
    m["total"] = result[-1]
    return m

