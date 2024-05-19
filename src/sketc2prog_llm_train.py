# link to VirtualHome
import sys
sys.path.append('../simulation')

import time
import resource
import numpy as np
from termcolor import colored
from multiprocessing import Pool
import torch
from helper import Constant, LinearStep, LCS
from program.utils import setup, log, summary, save, save_dict, load
from program.dataset import get_prog_dataset
from program.dataset import prog_collate_fn as collate_fn
from program.dataset import prog_to_cuda_fn as to_cuda_fn
from network.encoder_decoder import Desc2ProgramGeo
import json 
from sketc2prog_llm_test import test
import os

def view_process(iter, all_iter, loss, lr):
    message = "\rIteration:[{}/{}] Loss: {} LR:{:.6f}".format(iter, all_iter, loss, lr)
    sys.stdout.write(message)
    sys.stdout.flush()

def run_one_iteration(model, optim, scheduler, batch_data, train_args, args, dset, pool):

    model.train()
    train_args['p'].update()

    optim.zero_grad()
    loss = model(batch_data, inference=False, sampling='argmax', **train_args)
    loss = torch.mean(loss)
    loss.backward()
    optim.step()
    scheduler.step()
    return batch_data, loss.item()

def train(
        args,
        model,
        optim,
        scheduler,
        train_loader,
        test_loader,
        get_next_data_fn,
        checkpoint_dir,
        writer):

    # Train
    print(colored('Start training...', 'red'))
    pool = Pool(processes=max(args.n_workers, 1))

    train_args = {}
    if args.training_scheme == 'schedule_sampling':
        p = LinearStep(args.schedule_sampling_p_max,
                       args.schedule_sampling_p_min,
                       args.schedule_sampling_steps)
    else:
        p = Constant(1.)
    train_args.update({'p': p})
    dset = train_loader.dataset

    best_f1 = 0.0
    max_iter_idx = 0
    # best mode
    continuous_mode = 'best'
    contiguous_mode_id = -1 if continuous_mode == "best" else -2
    ckpt_paths = '{}/desc2program-{}.ckpt'.format(checkpoint_dir, continuous_mode)
    if os.path.exists(ckpt_paths):
        results_prefix = '{}/testing_results-desc2program-{}'.format(checkpoint_dir, continuous_mode)
        results = json.load(open('{}.json'.format(results_prefix), 'r'))
        iteration = results['iteration']
        best_f1 = results['total_f1']
        best_lcs = results['lcs']
        parasability = results['parsibility']
        executability = results['executability']
        max_iter_idx = int(iteration)
        load(args, contiguous_mode_id, checkpoint_dir, model.module)
        print("Iteration:{} LCS:{} F1:{} Prasability:{} Executability:{}".format(max_iter_idx, best_lcs, best_f1, parasability, executability))
        
    def _train_loop(init_iter=0, checkpoint_dir="", best_f1=0.0):
        iter = init_iter + 1

        for i in range(init_iter):
            train_args['p'].update()
            scheduler.step()
        
        while iter <= args.train_iters:
            for batch_data in train_loader:
                results, loss = run_one_iteration(model, optim, scheduler, batch_data, train_args, args, dset, pool)
                lr = optim.state_dict()['param_groups'][0]['lr']
                view_process(iter, args.train_iters, loss, lr)
                if iter % 250 == 0:
                    args.checkpoint = '{}/desc2program-{}.ckpt'.format(checkpoint_dir, iter)
                    results = test(args, model.module, test_loader, checkpoint_dir, word=False)
                    f1 = results['total_f1']
                    save(args, -2, checkpoint_dir, model.module)
                    save(args, -3, checkpoint_dir, model.module)
                    results_prefix = '{}/testing_results-desc2program-now'.format(checkpoint_dir)
                    json.dump(results, open('{}.json'.format(results_prefix), 'w'))
                    if f1 >= best_f1:
                        save(args, -1, checkpoint_dir, model.module)
                        results_prefix = '{}/testing_results-desc2program-best'.format(checkpoint_dir)
                        json.dump(results, open('{}.json'.format(results_prefix), 'w'))
                        best_f1 = f1
                iter += 1

    _train_loop(max_iter_idx, checkpoint_dir, best_f1)
    pool.close()
    pool.join()

def main():
    args, checkpoint_dir, model_config = setup(train=True)

    # get dataset
    if not os.path.exists("{}/dict.json".format(checkpoint_dir)):
        train_dset, test_dset = get_prog_dataset(args, train=True)
        save_dict(checkpoint_dir, train_dset)
    else:
        args.checkpoint = checkpoint_dir
        args.checkpoint_dir = checkpoint_dir
        train_dset, test_dset = get_prog_dataset(args, train=False)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.n_workers,
        collate_fn=collate_fn,
        drop_last=True)

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
        drop_last=False)

    # loader for the testing set
    def _loader():
        while True:
            for batch_data in test_loader:
                yield batch_data
    get_next_data_fn = _loader().__iter__().__next__

    # initialize model and optim
    model = Desc2ProgramGeo(train_dset, **model_config)

    ### additional
    optim = torch.optim.Adam(model.parameters(), args.model_lr_rate)
    milestones = [5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000, 13000, 14000, 15000, 16000, 17000, 18000, 19000, 20000]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, milestones, gamma=0.9, last_epoch=-1)

    lst = []
    for para in model.parameters():
        if para.requires_grad == True:
            lst.append(para.nelement())
    print(f"trainable paras number: {sum(lst)}")

    if args.gpu_id is not None:
        model.cuda()
        # model.set_to_cuda_fn(to_cuda_fn)
        model = torch.nn.DataParallel(model)
        model.module.set_to_cuda_fn(to_cuda_fn)

    # loading pre-computed future link relationship
    import pickle as pkl
    with open("../../Data/future_link.pkl", 'rb') as f:
        gt_link_collector = pkl.load(f)
    model.module.set_gt_link(gt_link_collector)

    train(args, model, optim, scheduler, train_loader, test_loader, get_next_data_fn, checkpoint_dir, writer=None)

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (1024 * 4, rlimit[1]))

if __name__ == '__main__':
    from multiprocessing import set_start_method
    try:
        set_start_method('spawn')
    except RuntimeError:
        pass
    torch.multiprocessing.set_sharing_strategy('file_system')
    main()
