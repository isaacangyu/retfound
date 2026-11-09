import os
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import Iterable, Optional
from timm.data import Mixup
from timm.utils import accuracy
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score, average_precision_score,
    hamming_loss, jaccard_score, recall_score, precision_score, cohen_kappa_score
)
from pycm import ConfusionMatrix
import util.misc as misc
import util.lr_sched as lr_sched
import util.oct_tasks as oct_tasks


def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    max_norm: float = 0,
    mixup_fn: Optional[Mixup] = None,
    log_writer=None,
    args=None
):
    """Train the model for one epoch."""
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    print_freq, accum_iter = 20, args.accum_iter
    optimizer.zero_grad()
    
    if log_writer:
        print(f'log_dir: {log_writer.log_dir}')
    
    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, f'Epoch: [{epoch}]')):
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)
        
        targets = targets.to(device, non_blocking=True)
        if mixup_fn:
            samples = samples.to(device, non_blocking=True)
            samples, targets = mixup_fn(samples, targets)
        
        with torch.cuda.amp.autocast():
            outputs = oct_tasks.forward_samples(model, samples, device)  # MIL bags are mean-pooled
            loss = criterion(outputs, targets)
        loss_value = loss.item()
        loss /= accum_iter
        
        loss_scaler(loss, optimizer, clip_grad=max_norm, parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        metric_logger.update(loss=loss_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss/train', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)
    
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate(data_loader, model, device, args, epoch, mode, num_class, log_writer):
    """Evaluate the model."""
    if getattr(args, 'task_type', None) is not None:
        return evaluate_task(data_loader, model, device, args, epoch, mode, log_writer)
    criterion = nn.CrossEntropyLoss()
    metric_logger = misc.MetricLogger(delimiter="  ")
    os.makedirs(os.path.join(args.output_dir, args.task), exist_ok=True)
    
    model.eval()
    true_onehot, pred_onehot, true_labels, pred_labels, pred_softmax = [], [], [], [], []
    
    for batch in metric_logger.log_every(data_loader, 10, f'{mode}:'):
        images, target = batch[0].to(device, non_blocking=True), batch[-1].to(device, non_blocking=True)
        target_onehot = F.one_hot(target.to(torch.int64), num_classes=num_class)
        
        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output, target)
        output_ = nn.Softmax(dim=1)(output)
        output_label = output_.argmax(dim=1)
        output_onehot = F.one_hot(output_label.to(torch.int64), num_classes=num_class)
        
        metric_logger.update(loss=loss.item())
        true_onehot.extend(target_onehot.cpu().numpy())
        pred_onehot.extend(output_onehot.detach().cpu().numpy())
        true_labels.extend(target.cpu().numpy())
        pred_labels.extend(output_label.detach().cpu().numpy())
        pred_softmax.extend(output_.detach().cpu().numpy())
    
    accuracy = accuracy_score(true_labels, pred_labels)
    hamming = hamming_loss(true_onehot, pred_onehot)
    jaccard = jaccard_score(true_onehot, pred_onehot, average='macro')
    average_precision = average_precision_score(true_onehot, pred_softmax, average='macro')
    kappa = cohen_kappa_score(true_labels, pred_labels)
    f1 = f1_score(true_onehot, pred_onehot, zero_division=0, average='macro')
    roc_auc = roc_auc_score(true_onehot, pred_softmax, multi_class='ovr', average='macro')
    precision = precision_score(true_onehot, pred_onehot, zero_division=0, average='macro')
    recall = recall_score(true_onehot, pred_onehot, zero_division=0, average='macro')
    
    score = (f1 + roc_auc + kappa) / 3
    if log_writer:
        for metric_name, value in zip(['accuracy', 'f1', 'roc_auc', 'hamming', 'jaccard', 'precision', 'recall', 'average_precision', 'kappa', 'score'],
                                       [accuracy, f1, roc_auc, hamming, jaccard, precision, recall, average_precision, kappa, score]):
            log_writer.add_scalar(f'perf/{metric_name}', value, epoch)
    
    print(f'val loss: {metric_logger.meters["loss"].global_avg}')
    print(f'Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}, ROC AUC: {roc_auc:.4f}, Hamming Loss: {hamming:.4f},\n'
          f' Jaccard Score: {jaccard:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f},\n'
          f' Average Precision: {average_precision:.4f}, Kappa: {kappa:.4f}, Score: {score:.4f}')
    
    metric_logger.synchronize_between_processes()
    
    results_path = os.path.join(args.output_dir, args.task, f'metrics_{mode}.csv')
    file_exists = os.path.isfile(results_path)
    with open(results_path, 'a', newline='', encoding='utf8') as cfa:
        wf = csv.writer(cfa)
        if not file_exists:
            wf.writerow(['val_loss', 'accuracy', 'f1', 'roc_auc', 'hamming', 'jaccard', 'precision', 'recall', 'average_precision', 'kappa'])
        wf.writerow([metric_logger.meters["loss"].global_avg, accuracy, f1, roc_auc, hamming, jaccard, precision, recall, average_precision, kappa])
    
    if mode == 'test':
        cm = ConfusionMatrix(actual_vector=true_labels, predict_vector=pred_labels)
        cm.plot(cmap=plt.cm.Blues, number_label=True, normalized=True, plot_lib="matplotlib")
        plt.savefig(os.path.join(args.output_dir, args.task, 'confusion_matrix_test.jpg'), dpi=600, bbox_inches='tight')
    
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, score


@torch.no_grad()
def evaluate_task(data_loader, model, device, args, epoch, mode, log_writer):
    """Evaluation for the OCT task flags (--multi-binary / --single-real / --response).

    Appends per-target + overall rows to output_dir/<task>/metrics_<mode>.csv. In test
    mode also writes benchmark/<run>/{analysis.tsv, predictions_test.csv} and updates
    benchmark/analysis.tsv. Returns (stats, score); see oct_tasks.task_metrics for score.
    """
    columns = oct_tasks.task_columns(args)
    split = oct_tasks.select_and_split(args)
    criterion = oct_tasks.OctTaskLoss([k for _, k in columns], smoothing=0.0)
    metric_logger = misc.MetricLogger(delimiter="  ")
    out_dir = os.path.join(args.output_dir, args.task)
    os.makedirs(out_dir, exist_ok=True)

    model.eval()
    outs, tgts = [], []
    use_amp = torch.device(device).type == 'cuda'
    for batch in metric_logger.log_every(data_loader, 10, f'{mode}:'):
        target = batch[-1].to(device, non_blocking=True).float()
        with torch.autocast(device_type='cuda', enabled=use_amp):
            output = oct_tasks.forward_samples(model, batch[0], device)
        loss = criterion(output, target)
        metric_logger.update(loss=loss.item())
        outs.append(output.float().cpu().numpy())
        tgts.append(target.cpu().numpy())
    metric_logger.synchronize_between_processes()

    pred, tgt = oct_tasks.decode_outputs(np.concatenate(outs), np.concatenate(tgts), columns, split['norm'])
    rows = oct_tasks.task_metrics(pred, tgt, columns, mode=oct_tasks.label_mode(args))
    score = rows[-1]['score']
    score = float('-inf') if np.isnan(score) else score
    loss_avg = metric_logger.meters['loss'].global_avg if 'loss' in metric_logger.meters else float('nan')

    print(f'{mode} loss: {loss_avg:.4f}')
    print(oct_tasks.format_rows(rows))
    print(f'Score: {score:.4f}')

    if log_writer:
        for k, v in rows[-1].items():
            if isinstance(v, float) and not np.isnan(v):
                log_writer.add_scalar(f'perf/{k}', v, epoch)

    if misc.is_main_process():
        path = os.path.join(out_dir, f'metrics_{mode}.csv')
        exists = os.path.isfile(path)
        with open(path, 'a', newline='', encoding='utf8') as f:
            wf = csv.writer(f)
            if not exists:
                wf.writerow(['epoch', 'loss'] + oct_tasks.METRIC_COLS)
            for r in rows:
                wf.writerow([epoch, loss_avg] + [r.get(c, '') for c in oct_tasks.METRIC_COLS])

        if mode == 'test':
            frame = data_loader.dataset.frame
            if args.dist_eval and misc.get_world_size() > 1:
                print('Warning: --dist_eval test predictions cover only this rank; skipping benchmark outputs.')
            else:
                oct_tasks.write_test_outputs(args, rows, pred, tgt, frame,
                                             best_epoch=getattr(args, 'best_epoch', None))

    return {'loss': loss_avg, 'score': score}, score
