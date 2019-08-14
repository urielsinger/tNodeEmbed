import numpy as np

from sklearn.metrics import roc_curve, auc, precision_recall_curve, f1_score

def auc_calc(y, y_pred, ax_roc=None, ax_pr=None):

    fpr, tpr, roc_threshold = roc_curve(y, y_pred)
    roc_auc = auc(fpr, tpr)
    if ax_roc is not None: ax_roc.plot(fpr, tpr, label='area = %0.3f, treatment = %s' % (roc_auc, 'all'))
    precision, recall, pr_threshold = precision_recall_curve(y, y_pred, pos_label=1)
    if ax_pr is not None: ax_pr.plot(recall, precision, label='treatment = %s' % 'all')

    if ax_roc is not None:
        handles, labels = ax_roc.get_legend_handles_labels()
        ax_roc.legend(handles, labels, loc=4, fontsize=14)
        ax_roc.set_xlim([0.0, 1.0])
        ax_roc.set_ylim([0.0, 1.05])
        ax_roc.set_xlabel('False Positive Rate', fontsize=14)
        ax_roc.set_ylabel('True Positive Rate', fontsize=14)

    if ax_pr is not None:
        handles, labels = ax_pr.get_legend_handles_labels()
        ax_pr.legend(handles, labels, loc=4, fontsize=14)
        ax_pr.set_xlim([0.0, 1.0])
        ax_pr.set_ylim([0.0, 1.0])
        ax_pr.set_xlabel('Recall', fontsize=14)
        ax_pr.set_ylabel('Precision', fontsize=14)

def get_auc(y_true, y_pred):
    if len(y_pred.shape) > 1:
        if len(y_true.shape) == 1:
            y = np.zeros_like(y_pred)
            y[np.arange(len(y_true)), y_true] = 1
            y_true = y
        y_true = y_true.ravel()
        y_pred = y_pred.ravel()

    fpr, tpr, _ = roc_curve(y_true, y_pred)
    return auc(fpr, tpr)

def get_f1_score(y_true, y_pred, average):
    if len(y_pred.shape) > 1:
        y_pred = np.argmax(y_pred, axis=1)
    if len(y_true.shape) > 1:
        y_true = np.argmax(y_true, axis=1)

    return f1_score(y_true, y_pred, average=average)

def get_metrics(y_true, y_pred):
    return {'auc': get_auc(y_true, y_pred),
            'f1_micro': get_f1_score(y_true, y_pred, average='micro'),
            'f1_macro': get_f1_score(y_true, y_pred, average='macro')}
