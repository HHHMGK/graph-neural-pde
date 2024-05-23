import os, sys, time, gc
module_path = os.path.abspath(os.path.join('./src'))
if module_path not in sys.path:
    sys.path.append(module_path)
from run_GNN import main
import torch

opt = {'use_cora_defaults': False, 'dataset': 'Cora', 'data_norm': 'rw', 'self_loop_weight': 1.0, 'use_labels': False, 'geom_gcn_splits': False, 'num_splits': 1, 'label_rate': 0.5, 'planetoid_split': False, 'hidden_dim': 16, 'fc_out': False, 'input_dropout': 0.5, 'dropout': 0.0, 'batch_norm': False, 'optimizer': 'adam', 'lr': 0.01, 'decay': 0.0005, 'epoch': 100, 'alpha': 1.0, 'alpha_dim': 'sc', 'no_alpha_sigmoid': False, 'beta_dim': 'sc', 'block': 'constant', 'function': 'laplacian', 'use_mlp': False, 'add_source': False, 'cgnn': False, 'time': 1.0, 'augment': False, 'method': None, 'step_size': 1, 'max_iters': 100, 'adjoint_method': 'adaptive_heun', 'adjoint': False, 'adjoint_step_size': 1, 'tol_scale': 1.0, 'tol_scale_adjoint': 1.0, 'ode_blocks': 1, 'max_nfe': 1000, 'no_early': False, 'earlystopxT': 3, 'max_test_steps': 100, 'leaky_relu_slope': 0.2, 'attention_dropout': 0.0, 'heads': 4, 'attention_norm_idx': 0, 'attention_dim': 64, 'mix_features': False, 'reweight_attention': False, 'attention_type': 'scaled_dot', 'square_plus': False, 'jacobian_norm2': None, 'total_deriv': None, 'kinetic_energy': None, 'directional_penalty': None, 'not_lcc': True, 'rewiring': None, 'gdc_method': 'ppr', 'gdc_sparsification': 'topk', 'gdc_k': 64, 'gdc_threshold': 0.0001, 'gdc_avg_degree': 64, 'ppr_alpha': 0.05, 'heat_time': 3.0, 'att_samp_pct': 1, 'use_flux': False, 'exact': False, 'M_nodes': 64, 'new_edges': 'random', 'sparsify': 'S_hat', 'threshold_type': 'topk_adj', 'rw_addD': 0.02, 'rw_rmvR': 0.02, 'rewire_KNN': False, 'rewire_KNN_T': 'T0', 'rewire_KNN_epoch': 5, 'rewire_KNN_k': 64, 'rewire_KNN_sym': False, 'KNN_online': False, 'KNN_online_reps': 4, 'KNN_space': 'pos_distance', 'beltrami': False, 'fa_layer': False, 'pos_enc_type': 'DW64', 'pos_enc_orientation': 'row', 'feat_hidden_dim': 64, 'pos_enc_hidden_dim': 32, 'edge_sampling': False, 'edge_sampling_T': 'T0', 'edge_sampling_epoch': 5, 'edge_sampling_add': 0.64, 'edge_sampling_add_type': 'importance', 'edge_sampling_rmv': 0.32, 'edge_sampling_sym': False, 'edge_sampling_online': False, 'edge_sampling_online_reps': 4, 'edge_sampling_space': 'attention', 'symmetric_attention': False, 'fa_layer_edge_sampling_rmv': 0.8, 'gpu': 0, 'pos_enc_csv': False, 'pos_dist_quantile': 0.001}
opt_fnc = ['laplacian', 'transformer']
opt_dts = ['Cora', 'Citeseer', 'Pubmed', 'Computers', 'Photo', 'CoauthorCS', 'ogbn-arxiv']
# opt_dts = ['Cora', 'Pubmed', 'Computers', 'Photo', 'CoauthorCS', 'ogbn-arxiv']

torch.cuda.empty_cache()
gc.collect()

for dts in opt_dts:
    for fnc in opt_fnc:    
        torch.cuda.reset_peak_memory_stats()
        opt['function'] = fnc
        opt['dataset'] = dts
        starttime = time.time()
        train_acc, val_acc, test_acc = main(opt, printing=False)
        print('Function:', fnc, 'Dataset:', dts)
        print('Train accuracy:', train_acc)
        print('Validation accuracy:', val_acc)
        print('Test accuracy:', test_acc)
        print('Execution time', time.time()-starttime)
        print('Peak memory usage:', torch.cuda.max_memory_allocated()/1024/1024, 'MB')
        print('---------------------------------')
        torch.cuda.empty_cache()
        gc.collect()
