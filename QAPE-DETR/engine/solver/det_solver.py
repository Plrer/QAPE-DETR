"""
DEIM: DETR with Improved Matching for Fast Convergence
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from D-FINE (https://github.com/Peterande/D-FINE)
Copyright (c) 2024 D-FINE authors. All Rights Reserved.
"""
import os
import time
import json
import datetime

import torch

from ..misc import dist_utils, stats

from ._solver import BaseSolver
from .det_engine import train_one_epoch, evaluate
from ..optim.lr_scheduler import FlatCosineLRScheduler


class DetSolver(BaseSolver):

    def fit(self, ):
        self.train()
        args = self.cfg

        n_parameters, model_stats = stats(self.cfg)
        print(model_stats)
        print("-"*42 + "Start training" + "-"*43)

        for i, (name, param) in enumerate(self.model.named_parameters()):
            if i in [194, 195]:
                print(f"Index {i}: {name} - requires_grad: {param.requires_grad}")

        self.self_lr_scheduler = False
        if args.lrsheduler is not None:
            iter_per_epoch = len(self.train_dataloader)
            print("     ## Using Self-defined Scheduler-{} ## ".format(args.lrsheduler))
            self.lr_scheduler = FlatCosineLRScheduler(self.optimizer, args.lr_gamma, iter_per_epoch, total_epochs=args.epoches, 
                                                warmup_iter=args.warmup_iter, flat_epochs=args.flat_epoch, no_aug_epochs=args.no_aug_epoch)
            self.self_lr_scheduler = True
        n_parameters = sum([p.numel() for p in self.model.parameters() if p.requires_grad])
        print(f'number of trainable parameters: {n_parameters}')

        n_parameters = sum([p.numel() for p in self.model.parameters() if not p.requires_grad])
        print(f'number of non-trainable parameters: {n_parameters}')

        top1 = 0
        best_stat = {'epoch': -1, }
        # evaluate again before resume training
        if self.last_epoch > 0:
            module = self.ema.module if self.ema else self.model
            test_stats, coco_evaluator = evaluate(
                module,
                self.criterion,
                self.postprocessor,
                self.val_dataloader,
                self.evaluator,
                self.device
            )
            for k in test_stats:
                best_stat['epoch'] = self.last_epoch
                best_stat[k] = test_stats[k][0]
                top1 = test_stats[k][0]
                print(f'best_stat: {best_stat}')

        best_stat_print = best_stat.copy()
        start_time = time.time()
        start_epoch = self.last_epoch + 1
        for epoch in range(start_epoch, args.epoches):

            self.train_dataloader.set_epoch(epoch)
            # self.train_dataloader.dataset.set_epoch(epoch)
            if dist_utils.is_dist_available_and_initialized():
                self.train_dataloader.sampler.set_epoch(epoch)

            if epoch == self.train_dataloader.collate_fn.stop_epoch:
                self.load_resume_state(str(self.output_dir / 'best_stg1.pth'))
                self.ema.decay = self.train_dataloader.collate_fn.ema_restart_decay
                print(f'Refresh EMA at epoch {epoch} with decay {self.ema.decay}')

            train_stats = train_one_epoch(
                self.self_lr_scheduler,
                self.lr_scheduler,
                self.model, 
                self.criterion, 
                self.train_dataloader, 
                self.optimizer, 
                self.device, 
                epoch, 
                max_norm=args.clip_max_norm, 
                print_freq=args.print_freq, 
                ema=self.ema, 
                scaler=self.scaler, 
                lr_warmup_scheduler=self.lr_warmup_scheduler,
                writer=self.writer
            )

            if not self.self_lr_scheduler:  # update by epoch 
                if self.lr_warmup_scheduler is None or self.lr_warmup_scheduler.finished():
                    self.lr_scheduler.step()

            self.last_epoch += 1

            if self.output_dir and epoch < self.train_dataloader.collate_fn.stop_epoch:
                checkpoint_paths = [self.output_dir / 'last.pth']
                # extra checkpoint before LR drop and every 100 epochs
                if (epoch + 1) % args.checkpoint_freq == 0:
                    checkpoint_paths.append(self.output_dir / f'checkpoint{epoch:04}.pth')
                for checkpoint_path in checkpoint_paths:
                    dist_utils.save_on_master(self.state_dict(), checkpoint_path)

            eval_freq = getattr(args, 'eval_freq', args.yaml_cfg.get('eval_freq', 1))
            do_eval = ((epoch + 1) % eval_freq == 0) or ((epoch + 1) == args.epoches)
            if not do_eval:
                continue

            module = self.ema.module if self.ema else self.model
            test_stats, coco_evaluator = evaluate(
                module,
                self.criterion,
                self.postprocessor,
                self.val_dataloader,
                self.evaluator,
                self.device
            )

            for k in test_stats:
                if self.writer and dist_utils.is_main_process():
                    for i, v in enumerate(test_stats[k]):
                        self.writer.add_scalar(f'Test/{k}_{i}'.format(k), v, epoch)

                metric_value = test_stats[k][0]

                if k in best_stat:
                    if metric_value > best_stat[k]:
                        best_stat['epoch'] = epoch
                        best_stat[k] = metric_value
                    else:
                        best_stat[k] = max(best_stat[k], metric_value)
                else:
                    best_stat['epoch'] = epoch
                    best_stat[k] = metric_value

                if best_stat[k] > top1:
                    top1 = best_stat[k]
                    best_stat_print['epoch'] = epoch
                    best_stat_print[k] = top1

                    if self.output_dir:
                        if epoch >= self.train_dataloader.collate_fn.stop_epoch:
                            stage_name = 'best_stg2'
                        else:
                            stage_name = 'best_stg1'

                        named_path = self.output_dir / f'{stage_name}_e{epoch:03d}_{k}{top1:.4f}.pth'
                        self._save_best_checkpoint(stage_name, epoch, k, top1)
                        self._remove_old_named_best(stage_name, named_path)

                else:
                    best_stat_print[k] = max(best_stat.get(k, metric_value), top1)

                print(f'best_stat: {best_stat_print}')

                if epoch >= self.train_dataloader.collate_fn.stop_epoch and best_stat['epoch'] != epoch:
                    best_stat = {'epoch': -1, }
                    self.ema.decay -= 0.0001
                    self.load_resume_state(str(self.output_dir / 'best_stg1.pth'))
                    print(f'Refresh EMA at epoch {epoch} with decay {self.ema.decay}')


            log_stats = {
                **{f'train_{k}': v for k, v in train_stats.items()},
                **{f'test_{k}': v for k, v in test_stats.items()},
                'epoch': epoch,
                'n_parameters': n_parameters
            }

            if self.output_dir and dist_utils.is_main_process():
                with (self.output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

                # for evaluation logs
                if coco_evaluator is not None:
                    (self.output_dir / 'eval').mkdir(exist_ok=True)
                    if "bbox" in coco_evaluator.coco_eval:
                        filenames = ['latest.pth']
                        if epoch % 50 == 0:
                            filenames.append(f'{epoch:03}.pth')
                        for name in filenames:
                            torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                    self.output_dir / "eval" / name)

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))

    def _save_best_checkpoint(self, stage_name, epoch, metric_name, metric_value):
        if not self.output_dir:
            return None

        # 固定名 best
        fixed_path = self.output_dir / f'{stage_name}.pth'
        dist_utils.save_on_master(self.state_dict(), fixed_path)

        # 带 epoch 和指标名的 best
        named_path = self.output_dir / f'{stage_name}_e{epoch:03d}_{metric_name}{metric_value:.4f}.pth'
        dist_utils.save_on_master(self.state_dict(), named_path)

        # 记录信息
        info = {
            'stage': stage_name,
            'epoch': int(epoch),
            'metric_name': metric_name,
            'metric_value': float(metric_value),
            'fixed_path': str(fixed_path),
            'named_path': str(named_path),
            'time': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        info_path = self.output_dir / f'{stage_name}_info.json'
        if dist_utils.is_main_process():
            with open(info_path, 'w', encoding='utf-8') as f:
                json.dump(info, f, ensure_ascii=False, indent=2)

        print(f'[Best Saved] {stage_name}: epoch={epoch}, {metric_name}={metric_value:.4f}')
        print(f'[Best Saved] fixed: {fixed_path}')
        print(f'[Best Saved] named: {named_path}')

        return named_path

    def _remove_old_named_best(self, stage_name, keep_path):
        if not self.output_dir or not dist_utils.is_main_process():
            return

        prefix = f'{stage_name}_e'
        for p in self.output_dir.glob(f'{prefix}*.pth'):
            if str(p) != str(keep_path):
                try:
                    os.remove(p)
                except Exception as e:
                    print(f'Warning: failed to remove old checkpoint {p}, error: {e}')

    def val(self, ):
        self.eval()

        module = self.ema.module if self.ema else self.model
        test_stats, coco_evaluator = evaluate(module, self.criterion, self.postprocessor,
                self.val_dataloader, self.evaluator, self.device)

        if self.output_dir:
            dist_utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, self.output_dir / "eval.pth")

        return
