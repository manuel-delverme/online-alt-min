import argparse
import random

import numpy as np
import torch
import torch.nn as nn

import experiment_buddy
from altmin import get_mods, get_codes, update_codes, update_last_layer_, update_hidden_weights_adam_
from altmin import scheduler_step
from models import LeNet
from models import test
from utils import get_devices, load_dataset


def main():
    # global device, n_inputs, step, data
    # Check cuda
    device, num_gpus = get_devices("cuda:0" if not args.no_cuda and torch.cuda.is_available() else "cpu", seed=args.seed)
    # Load data and model
    model_name = "lenet"
    if model_name == 'feedforward' or model_name == 'binary':
        model_name += '_' + str(args.n_hidden_layers) + 'x' + str(args.n_hiddens)
    print('\nOnline alternating-minimization with sgd')
    print('* Loading dataset {}'.format(args.dataset))
    print('* Loading model {}'.format(model_name))
    train_loader, test_loader, n_inputs = load_dataset(args.dataset, batch_size=args.batch_size, conv_net=True, num_workers=0)
    window_size = train_loader.dataset.data[0].shape[0]
    if len(train_loader.dataset.data[0].shape) == 3:
        num_input_channels = train_loader.dataset.data[0].shape[2]
    else:
        num_input_channels = 1
    model = LeNet(num_input_channels=num_input_channels, window_size=window_size, bias=True).to(device)
    criterion = nn.CrossEntropyLoss()
    if __name__ == "__main__":
        # Save everything in a `ddict`
        # SAV = ddict(args=args.__dict__)

        # Store training and test performance after each training epoch
        # SAV.perf = ddict(tr=[], te=[])

        # Store test performance after each iteration in first epoch
        # SAV.perf.first_epoch = []

        # Store test performance after each args.save_interval iterations
        # SAV.perf.te_vs_iterations = []

        # Expose model modules that has_codes
        model = get_mods(model, optimizer='Adam', optimizer_params={'lr': args.lr_weights}, scheduler=lambda epoch: 1 / 2 ** (epoch // args.lr_half_epochs))

        model[-1].optimizer.param_groups[0]['lr'] = args.lr_out

        # Initial mu and increment after every mini-batch
        mu = args.mu
        mu_max = 10 * args.mu

        step = 0
        for epoch in range(1, args.epochs + 1):
            print('\nEpoch {} of {}. mu = {:.4f}, lr_out = {}'.format(epoch, args.epochs, mu, model[-1].scheduler.get_lr()))

            for batch_idx, (data, targets) in enumerate(train_loader):
                data, targets = data.to(device), targets.to(device)

                # (1) Forward
                model.train()
                if step == 0:
                    test_accuracy, test_loss = test(model, data_loader=test_loader, label=" - Test")
                    tb.add_scalar("test/accuracy", test_accuracy, step)
                    tb.add_scalar("test/loss", test_loss, step)

                    # Outputs to terminal
                    loss = criterion(outputs, targets)
                    tb.add_scalar("train/epoch", epoch, step)
                    tb.add_scalar("train/loss", loss, step)

                with torch.no_grad():
                    outputs, codes = get_codes(model, data)

                # (2) Update codes
                codes, num_gradients = update_codes(codes, model, targets, criterion, mu, lambda_c=args.lambda_c, n_iter=args.n_iter_codes, lr=args.lr_codes)
                step += num_gradients

                # (3) Update weights
                num_gradients = update_last_layer_(model[-1], codes[-1], targets, criterion, n_iter=args.n_iter_weights)
                step += num_gradients

                num_gradients = update_hidden_weights_adam_(model, data, codes, lambda_w=args.lambda_w, n_iter=args.n_iter_weights)
                step += num_gradients

                # Store all iterations of first epoch
                test_accuracy, test_loss = test(model, data_loader=test_loader, label=" - Test")
                tb.add_scalar("test/accuracy", test_accuracy, step)
                tb.add_scalar("test/loss", test_loss, step)

                # Outputs to terminal
                loss = criterion(outputs, targets)
                tb.add_scalar("train/epoch", epoch, step)
                tb.add_scalar("train/loss", loss, step)

                # Increment mu
                if mu < mu_max:
                    mu = mu + args.d_mu

            scheduler_step(model)

            # Print performances
            # SAV.perf.tr += [test(model, data_loader=train_loader, label="Training")]
            # SAV.perf.te += [test(model, data_loader=test_loader, label="Test")]
            train_accuracy, train_loss = test(model, data_loader=train_loader)
            tb.add_scalar("train/accuracy", train_accuracy, step)
            tb.add_scalar("train/loss", train_loss, step)

            test_accuracy, test_loss = test(model, data_loader=test_loader)
            tb.add_scalar("test/accuracy", test_accuracy, step)
            tb.add_scalar("test/loss", test_loss, step)


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Online Alternating-Minimization with SGD')
    parser.add_argument('--dataset', default='mnist', metavar='D', help='name of dataset')
    parser.add_argument('--data-augmentation', action='store_true', default=False, help='enables data augmentation')
    parser.add_argument('--batch-size', type=int, default=200, metavar='B', help='input batch size for training')
    parser.add_argument('--epochs', type=int, default=50, metavar='E', help='number of epochs to train (default: 50)')
    parser.add_argument('--n-iter-codes', type=int, default=5, metavar='N', help='number of internal iterations for codes optimization')
    parser.add_argument('--n-iter-weights', type=int, default=1, metavar='N', help='number of internal iterations in learning weights')
    parser.add_argument('--lr-codes', type=float, default=0.3, metavar='LR', help='learning rate for codes updates')
    parser.add_argument('--lr-out', type=float, default=0.008, metavar='LR', help='learning rate for last layer weights updates')
    parser.add_argument('--lr-weights', type=float, default=0.008, metavar='LR', help='learning rate for hidden weights updates')
    parser.add_argument('--lr-half-epochs', type=int, default=8, metavar='LH', help='number of epochs after which learning rate if halfed')
    parser.add_argument('--no-batchnorm', action='store_true', default=True, help='disables batchnormalization')
    parser.add_argument('--lambda_c', type=float, default=0.0, metavar='L', help='codes sparsity')
    parser.add_argument('--lambda_w', type=float, default=0.0, metavar='L', help='weight sparsity')
    parser.add_argument('--mu', type=float, default=0.003, metavar='M', help='initial mu parameter')
    parser.add_argument('--d-mu', type=float, default=0.0 / 300, metavar='M', help='increase in mu after every mini-batch')
    parser.add_argument('--random_seed', type=int, default=6, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N', help='how many batches to wait before logging training status')
    parser.add_argument('--save-interval', type=int, default=1000, metavar='N', help='how many batches to wait before saving test performance (if set to zero, it does not save)')
    parser.add_argument('--log-first-epoch', action='store_true', default=False, help='whether or not it should test and log after every mini-batch in first epoch')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='enables CUDA training')
    args = parser.parse_args()

    experiment_buddy.register_defaults(vars(args))
    tb = experiment_buddy.deploy("mila")
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)
    main()
