from metrics import dice_score, sm_score, em_score, wfm_score, iou_score

def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def adjust_lr(optimizer, init_lr, epoch, decay_rate=0.1, decay_epoch=30):
    decay = decay_rate ** (epoch // decay_epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = decay*init_lr
        lr=param_group['lr']
    return lr

def calculate_metrics(y_true, y_pred):
    score_smeasure = sm_score(y_true, y_pred)
    score_wfmeasure = wfm_score(y_true, y_pred)
    score_emeasure = em_score(y_true, y_pred)
    score_f1 = dice_score(y_true, y_pred)
    score_iou = iou_score(y_true, y_pred)

    return [score_f1, score_iou, score_smeasure, score_emeasure, score_wfmeasure]