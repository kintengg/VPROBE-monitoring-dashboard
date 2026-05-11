'''
by lyuwenyu
'''
import time 
import json
import datetime

import torch 
import numpy as np

from src.misc import dist
from src.data import get_coco_api_from_dataset

from .solver import BaseSolver
from .det_engine import train_one_epoch, evaluate


class DetSolver(BaseSolver):
    
    def __init__(self, cfg):
        super().__init__(cfg)
        self.save_visualizations = False  # Default to False, can be set via command line
    
    def fit(self, ):
        print("Start training")
        self.train()

        args = self.cfg 
        
        n_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print('number of params:', n_parameters)
        
        # Store as instance variables for later use
        self.n_parameters = n_parameters
        self.n_parameters_M = n_parameters / 1e6  # In millions
        
        # Calculate GFLOPs using thop library
        try:
            from thop import profile, clever_format
            import copy
            
            dummy_input = torch.randn(1, 3, 640, 640).to(next(self.model.parameters()).device)
            
            # Create a deep copy to avoid modifying the original model
            # thop adds attributes to modules which can interfere with training
            model_copy = copy.deepcopy(self.model)
            
            flops, params = profile(model_copy, inputs=(dummy_input,), verbose=False)
            gflops = flops / 1e9
            
            # Delete the copy to free memory
            del model_copy
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            # Format for display
            flops_str, params_str = clever_format([flops, params], "%.3f")
            
            print(f'GFLOPs: {gflops:.2f} ({flops_str})')
            self.gflops = gflops
        except ImportError:
            print('Warning: thop library not found. Install it with: pip install thop')
            print('GFLOPs calculation skipped.')
            self.gflops = None
        except Exception as e:
            print(f'Warning: Could not calculate GFLOPs: {e}')
            print('GFLOPs calculation skipped.')
            self.gflops = None
        
        # Print model complexity summary
        print(f'\n[Model Complexity]')
        print(f'  Parameters: {self.n_parameters_M:.2f}M ({self.n_parameters:,})')
        if self.gflops is not None:
            print(f'  GFLOPs: {self.gflops:.2f}')
        print()

        base_ds = get_coco_api_from_dataset(self.val_dataloader.dataset)
        # best_stat = {'coco_eval_bbox': 0, 'coco_eval_masks': 0, 'epoch': -1, }
        best_stat = {'epoch': -1, }

        start_time = time.time()
        for epoch in range(self.last_epoch + 1, args.epoches):
            if dist.is_dist_available_and_initialized():
                self.train_dataloader.sampler.set_epoch(epoch)
            
            train_stats = train_one_epoch(
                self.model, self.criterion, self.train_dataloader, self.optimizer, self.device, epoch,
                args.clip_max_norm, print_freq=args.log_step, ema=self.ema, scaler=self.scaler)

            self.lr_scheduler.step()
            
            if self.output_dir:
                checkpoint_paths = [self.output_dir / 'checkpoint.pth']
                # extra checkpoint before LR drop and every 100 epochs
                if (epoch + 1) % args.checkpoint_step == 0:
                    checkpoint_paths.append(self.output_dir / f'checkpoint{epoch:04}.pth')
                for checkpoint_path in checkpoint_paths:
                    dist.save_on_master(self.state_dict(epoch), checkpoint_path)

            module = self.ema.module if self.ema else self.model
            test_stats, coco_evaluator = evaluate(
                module, self.criterion, self.postprocessor, self.val_dataloader, base_ds, self.device, self.output_dir, 
                epoch=epoch, save_visualizations=self.save_visualizations, save_logs=False
            )

            # TODO 
            for k in test_stats.keys():
                # Handle both scalar values and array values
                if isinstance(test_stats[k], (list, tuple, np.ndarray)):
                    current_value = test_stats[k][0]
                else:
                    current_value = test_stats[k]
                
                if k in best_stat:
                    if current_value > best_stat[k]:
                        best_stat['epoch'] = epoch
                        best_stat[k] = current_value
                else:
                    best_stat['epoch'] = epoch
                    best_stat[k] = current_value
            print('best_stat: ', best_stat)


            additional_metrics = {}
            if coco_evaluator and 'bbox' in coco_evaluator.coco_eval:
                stats = coco_evaluator.coco_eval['bbox'].stats
                additional_metrics.update({
                    'test_mAP@0.5:0.95': stats[0],
                    'test_mAP@0.5': stats[1],
                    'test_mAP@0.75': stats[2],
                    'test_mAP_small': stats[3],
                    'test_mAP_medium': stats[4],
                    'test_mAP_large': stats[5],
                    'test_AR@1': stats[6],
                    'test_AR@10': stats[7],
                    'test_AR@100': stats[8],
                    'test_AR_small': stats[9],
                    'test_AR_medium': stats[10],
                    'test_AR_large': stats[11]
                })

                coco_eval = coco_evaluator.coco_eval['bbox']
                precision = coco_eval.eval['precision']
                recall = coco_eval.eval['recall']

                # Precision shape: [T, R, K, A, M]
                # T=IoU thresholds, R=recall thresholds, K=categories, A=area ranges, M=maxDets
                
                # Use metrics from test_stats if available (computed in det_engine.py)
                # Otherwise compute them here for backward compatibility
                if 'precision_0.5' in test_stats:
                    additional_metrics.update({
                        'test_precision_0.5': test_stats['precision_0.5'],
                        'test_recall_0.5': test_stats['recall_0.5'],
                        'test_f1_score_0.5': test_stats['f1_score_0.5']
                    })
                else:
                    # Fallback: compute manually
                    # Average Precision @ IoU=0.5 (first IoU threshold, all categories, all areas, maxDet=100)
                    precision_50 = precision[0, :, :, 0, 2]  # IoU=0.5, all recall, all cats, all areas, maxDet=100
                    valid_precision_50 = precision_50[precision_50 > -1]
                    avg_precision_50 = valid_precision_50.mean() if len(valid_precision_50) > 0 else 0.0

                    # Recall @ IoU=0.5 (from recall array)
                    recall_50 = recall[0, :, 0, 2]  # IoU=0.5, all categories, all areas, maxDet=100
                    valid_recall_50 = recall_50[recall_50 > -1]
                    avg_recall_50 = valid_recall_50.mean() if len(valid_recall_50) > 0 else 0.0
                    
                    # F1 Score @ IoU=0.5
                    if avg_precision_50 > 0 and avg_recall_50 > 0:
                        f1_score_50 = 2 * (avg_precision_50 * avg_recall_50) / (avg_precision_50 + avg_recall_50)
                    else:
                        f1_score_50 = 0.0
                    
                    additional_metrics.update({
                        'test_precision_0.5': avg_precision_50,
                        'test_recall_0.5': avg_recall_50,
                        'test_f1_score_0.5': f1_score_50
                    })
                
                # Use metrics from test_stats for IoU=0.75 if available
                if 'precision_0.75' in test_stats:
                    additional_metrics.update({
                        'test_precision_0.75': test_stats['precision_0.75'],
                        'test_recall_0.75': test_stats['recall_0.75'],
                        'test_f1_score_0.75': test_stats['f1_score_0.75']
                    })
                
                # Average Precision @ IoU=0.5:0.95 (average across all IoU thresholds)
                precision_all_iou = precision[:, :, :, 0, 2]  # All IoU, all recall, all cats, all areas, maxDet=100
                valid_precision_all = precision_all_iou[precision_all_iou > -1]
                avg_precision_all = valid_precision_all.mean() if len(valid_precision_all) > 0 else 0.0

                # Recall @ IoU=0.5:0.95 (average across all IoU thresholds)
                recall_all_iou = recall[:, :, 0, 2]  # All IoU, all cats, all areas, maxDet=100
                valid_recall_all = recall_all_iou[recall_all_iou > -1]
                avg_recall_all = valid_recall_all.mean() if len(valid_recall_all) > 0 else 0.0

                # F1 Score @ IoU=0.5:0.95
                if avg_precision_all > 0 and avg_recall_all > 0:
                    f1_score_all = 2 * (avg_precision_all * avg_recall_all) / (avg_precision_all + avg_recall_all)
                else:
                    f1_score_all = 0.0

                additional_metrics.update({
                    'test_precision_0.5:0.95': avg_precision_all,
                    'test_recall_0.5:0.95': avg_recall_all,
                    'test_f1_score_0.5:0.95': f1_score_all
                })
                
                # Add TP/FP/FN counts if available
                if 'total_tp' in test_stats:
                    additional_metrics.update({
                        'test_total_tp': test_stats['total_tp'],
                        'test_total_fp': test_stats['total_fp'],
                        'test_total_fn': test_stats['total_fn'],
                        'test_total_gt': test_stats['total_gt']
                    })


                # Per-class AP for detailed analysis
                per_class_ap = []
                for cat_idx in range(precision.shape[2]):  # Loop over categories
                    cat_precision = precision[:, :, cat_idx, 0, 2]  # All IoU, all recall, this cat
                    valid_cat_precision = cat_precision[cat_precision > -1]
                    if len(valid_cat_precision) > 0:
                        per_class_ap.append(valid_cat_precision.mean())
                    else:
                        per_class_ap.append(0.0)
                
                # Log per-class metrics
                if hasattr(base_ds, 'loadCats'):
                    try:
                        cat_ids = base_ds.getCatIds()
                        cats = base_ds.loadCats(cat_ids)
                        for i, (ap, cat) in enumerate(zip(per_class_ap, cats)):
                            additional_metrics[f"test_AP_{cat['name']}"] = ap
                    except:
                        pass

            # Convert numpy types to Python native types for JSON serialization
            def convert_to_python_type(obj):
                if isinstance(obj, (np.integer, np.floating)):
                    return obj.item()
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: convert_to_python_type(v) for k, v in obj.items()}
                elif isinstance(obj, (list, tuple)):
                    return [convert_to_python_type(v) for v in obj]
                return obj
            
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        **{f'test_{k}': v for k, v in test_stats.items()},
                        **additional_metrics,
                        'epoch': epoch,
                        'n_parameters': n_parameters,
                        'n_parameters_M': self.n_parameters_M,
                        'gflops': self.gflops if self.gflops is not None else 0.0}
            
            # Convert all numpy types to Python native types
            log_stats = convert_to_python_type(log_stats)

            if self.output_dir and dist.is_main_process():
                with (self.output_dir / "logs.txt").open("a") as f:
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


    def val(self, save_logs=True):
        self.eval()

        base_ds = get_coco_api_from_dataset(self.val_dataloader.dataset)
        
        module = self.ema.module if self.ema else self.model
        # Pass save_logs=True when called directly (test-only mode)
        test_stats, coco_evaluator = evaluate(module, self.criterion, self.postprocessor,
                self.val_dataloader, base_ds, self.device, self.output_dir, epoch=0, 
                save_visualizations=self.save_visualizations, save_logs=save_logs)
                
        if self.output_dir:
            dist.save_on_master(coco_evaluator.coco_eval["bbox"].eval, self.output_dir / "eval.pth")
        
        return