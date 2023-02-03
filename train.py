import os
import time
import torch
from tqdm import tqdm
import torch.utils.data
import torch.optim as optim
import torch.optim.lr_scheduler as LS

import config
import models
from test import testing


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def check_path(path):
    if not os.path.isdir(path):
        os.mkdir(path)
        print(f"checking paths, mkdir: {path}")


def main():
    check_path(config.para.save_path)
    check_path(config.para.folder)
    set_seed(996007)

    net = models.TCS_Net().train().to(config.para.device)
    optimizer = optim.Adam(filter(lambda x: x.requires_grad, net.parameters()), lr=1e-3)
    scheduler = LS.MultiStepLR(optimizer, milestones=[51, 101], gamma=0.1)
    if os.path.exists(config.para.my_state_dict):
        if torch.cuda.is_available():
            net.load_state_dict(torch.load(config.para.my_state_dict, map_location=config.para.device))
            info = torch.load(config.para.my_info, map_location=config.para.device)
        else:
            raise Exception(f"No GPU.")

        start_epoch = info["epoch"]
        current_best = info["res"]
        print(f"Loaded trained model of epoch {start_epoch}, res: {current_best}.")
    else:
        start_epoch = 1
        current_best = 0
        print("No saved model, start epoch = 1.")

    print("Data loading...")
    # train_set = utils.TrainDatasetFromFolder('./dataset/train/', block_size=utils.para.block_size)
    # dataset_train = torch.utils.data.DataLoader(
    #     dataset=train_set, num_workers=8, batch_size=utils.para.batch_size, shuffle=True)
    dataset_train = config.train_loader()

    over_all_time = time.time()
    for epoch in range(start_epoch, int(200)):
        print("Lr: {}.".format(optimizer.param_groups[0]['lr']))

        epoch_loss = 0.
        dic = {"epoch": epoch, "device": config.para.device, "rate": config.para.rate, "lr": optimizer.param_groups[0]['lr']}
        for idx, xi in enumerate(tqdm(dataset_train, desc="Now training: ", postfix=dic)):
            xi = xi.to(config.para.device)

            optimizer.zero_grad()
            xo, _ = net(xi)
            batch_loss = torch.mean(torch.pow(xo - xi, 2)).to(config.para.device)

            if epoch != 1 and batch_loss > 2:
                print("\nWarning: your loss > 2 !")

            epoch_loss += batch_loss.item()

            batch_loss.backward()
            optimizer.step()

            if idx % 10 == 0:
                tqdm.write("\r[{:5}/{:5}], Loss: [{:8.6f}]".format(
                    config.para.batch_size * (idx + 1),
                    dataset_train.__len__() * config.para.batch_size,
                    batch_loss.item()))

        avg_loss = epoch_loss / dataset_train.__len__()
        print("\n=> Epoch of {:2}, Epoch Loss: [{:8.6f}]".format(epoch, avg_loss))

        # Make a log note.
        if epoch == 1:
            if not os.path.isfile(config.para.my_log):
                output_file = open(config.para.my_log, 'w')
                output_file.write("=" * 120 + "\n")
                output_file.close()
            output_file = open(config.para.my_log, 'r+')
            old = output_file.read()
            output_file.seek(0)
            output_file.write("\nAbove is {} test. Note：{}.\n"
                              .format("???", None) + "=" * 120 + "\n")
            output_file.write(old)
            output_file.close()

        with torch.no_grad():
            p, s = testing(net.eval(), val=True, manner="grey", save_img=False)
        print("{:5.3f}".format(p))
        if p > current_best:
            epoch_info = {"epoch": epoch, "res": p}
            torch.save(net.state_dict(), config.para.my_state_dict)
            torch.save(epoch_info, config.para.my_info)
            print("Check point saved\n")
            current_best = p
            output_file = open(config.para.my_log, 'r+')
            old = output_file.read()
            output_file.seek(0)

            output_file.write(f"Epoch {epoch}, Loss of train {round(avg_loss, 6)}, Res {round(current_best, 2)}, {round(s, 4)}\n")
            output_file.write(old)
            output_file.close()

        scheduler.step()
        print("Over all time: {:.3f}s".format(time.time() - over_all_time))

    print("Train end.")


if __name__ == "__main__":
    main()
