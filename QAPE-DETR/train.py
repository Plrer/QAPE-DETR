"""
DEIMv2: Real-Time Object Detection Meets DINOv3
Copyright (c) 2025 The DEIMv2 Authors. All Rights Reserved.
---------------------------------------------------------------------------------
DEIM: DETR with Improved Matching for Fast Convergence
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright (c) 2023 lyuwenyu. All Rights Reserved.
"""

import os
import sys
import json
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import argparse

from engine.misc import dist_utils
from engine.core import YAMLConfig, yaml_utils
from engine.solver import TASKS

debug = False

if debug:
    import torch

    def custom_repr(self):
        return f'{{Tensor:{tuple(self.shape)}}} {original_repr(self)}'
    original_repr = torch.Tensor.__repr__
    torch.Tensor.__repr__ = custom_repr


def switch_eval_set(cfg, eval_set='val'):
    """
    eval_set='val'  -> 默认验证集
    eval_set='test' -> 把 val_dataloader 临时替换成 test_dataloader
    """
    if eval_set == 'test':
        assert 'test_dataloader' in cfg.yaml_cfg, \
            'test_dataloader is not defined in yaml config.'
        cfg.yaml_cfg['val_dataloader'] = cfg.yaml_cfg['test_dataloader']


def find_best_checkpoint(output_dir):
    """
    优先从 best_stg1_info.json / best_stg2_info.json 中找真正最优权重；
    如果没有 info 文件，则退化到 best_stg2.pth / best_stg1.pth。
    """
    output_dir = Path(output_dir)

    candidates = []

    for stage_name in ['best_stg1', 'best_stg2']:
        info_path = output_dir / f'{stage_name}_info.json'
        if info_path.exists():
            try:
                with open(info_path, 'r', encoding='utf-8') as f:
                    info = json.load(f)

                metric_value = float(info.get('metric_value', -1))
                named_path = info.get('named_path', None)
                fixed_path = info.get('fixed_path', None)

                ckpt_path = None
                if named_path and Path(named_path).exists():
                    ckpt_path = Path(named_path)
                elif fixed_path and Path(fixed_path).exists():
                    ckpt_path = Path(fixed_path)

                if ckpt_path is not None:
                    candidates.append((metric_value, ckpt_path))
            except Exception as e:
                print(f'Warning: failed to parse {info_path}, error: {e}')

    if len(candidates) > 0:
        candidates = sorted(candidates, key=lambda x: x[0], reverse=True)
        return candidates[0][1]

    # fallback
    for p in [
        output_dir / 'best_stg2.pth',
        output_dir / 'best_stg1.pth',
        output_dir / 'best.pth',
    ]:
        if p.exists():
            return p

    return None


def build_cfg(args, update_dict):
    cfg = YAMLConfig(args.config, **update_dict)

    if args.resume or args.tuning:
        if 'HGNetv2' in cfg.yaml_cfg:
            cfg.yaml_cfg['HGNetv2']['pretrained'] = False

    return cfg


def run_final_test(args, update_dict, best_ckpt):
    """
    训练结束后，自动用 best checkpoint 在 test_dataloader 上跑一次最终测试
    """
    print('\n' + '=' * 80)
    print(f'Auto final test with best checkpoint: {best_ckpt}')
    print('=' * 80)

    test_update_dict = dict(update_dict)
    test_update_dict['resume'] = str(best_ckpt)

    test_cfg = YAMLConfig(args.config, **test_update_dict)

    if 'HGNetv2' in test_cfg.yaml_cfg:
        test_cfg.yaml_cfg['HGNetv2']['pretrained'] = False

    switch_eval_set(test_cfg, eval_set='test')

    print('test cfg: ', test_cfg.__dict__)

    test_solver = TASKS[test_cfg.yaml_cfg['task']](test_cfg)
    test_solver.val()


def main(args) -> None:
    dist_utils.setup_distributed(args.print_rank, args.print_method, seed=args.seed)

    assert not all([args.tuning, args.resume]), \
        'Only support from_scrach or resume or tuning at one time'

    update_dict = yaml_utils.parse_cli(args.update)
    update_dict.update({
        k: v for k, v in args.__dict__.items()
        if k not in ['update'] and v is not None
    })

    cfg = build_cfg(args, update_dict)

    # test_only 时，支持 val / test 切换
    if args.test_only:
        switch_eval_set(cfg, eval_set=args.eval_set)

    print('cfg: ', cfg.__dict__)

    solver = TASKS[cfg.yaml_cfg['task']](cfg)

    if args.test_only:
        solver.val()
    else:
        solver.fit()

        # 训练结束后自动用最优模型跑 test
        if 'test_dataloader' in cfg.yaml_cfg:
            output_dir = solver.output_dir if hasattr(solver, 'output_dir') else cfg.yaml_cfg.get('output_dir', None)

            if output_dir is None:
                print('Warning: output_dir is None, skip final test.')
            else:
                best_ckpt = find_best_checkpoint(output_dir)
                if best_ckpt is None:
                    print(f'Warning: no best checkpoint found in {output_dir}, skip final test.')
                else:
                    run_final_test(args, update_dict, best_ckpt)
        else:
            print('No test_dataloader found in config, skip final test.')

    dist_utils.cleanup()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-c', '--config',
        type=str,
        default='configs/qape_detr.yml'
    )
    parser.add_argument('-r', '--resume', type=str, help='resume from checkpoint')
    parser.add_argument('-t', '--tuning', type=str, help='tuning from checkpoint')
    parser.add_argument('-d', '--device', type=str, help='device')
    parser.add_argument('--seed', type=int, default=0, help='exp reproducibility')
    parser.add_argument('--use-amp', action='store_true', help='auto mixed precision training')
    parser.add_argument('--output-dir', type=str, help='output directoy')
    parser.add_argument('--summary-dir', type=str, help='tensorboard summry')
    parser.add_argument('--test-only', action='store_true', default=False)

    parser.add_argument('--eval-set', type=str, default='val', choices=['val', 'test'],
                        help='which split to evaluate when using --test-only')

    parser.add_argument('-u', '--update', nargs='+', help='update yaml config')

    parser.add_argument('--print-method', type=str, default='builtin', help='print method')
    parser.add_argument('--print-rank', type=int, default=0, help='print rank id')
    parser.add_argument('--local-rank', type=int, help='local rank id')

    args = parser.parse_args()
    main(args)