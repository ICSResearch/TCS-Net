import argparse


parser = argparse.ArgumentParser(description="Args of this repo.")
parser.add_argument("--rate", default=0.1, type=float)
parser.add_argument("--lr", default=0.001, type=float)
parser.add_argument("--device", default="0")
parser.add_argument("--time", default=0, type=int)
parser.add_argument("--block_size", default=96, type=int)
parser.add_argument("--batch_size", default=32, type=int)

parser.add_argument("--save", default=False)
parser.add_argument("--manner", default="grey")

parser.add_argument("--save_path", default=f"./trained_models")
parser.add_argument("--folder")
parser.add_argument("--my_state_dict")
parser.add_argument("--my_log")
parser.add_argument("--my_info")
para = parser.parse_args()
para.device = f"cuda:{para.device}"
para.folder = f"{para.save_path}/{str(int(para.rate * 100))}/"
para.my_state_dict = f"{para.folder}/state_dict.pth"
para.my_log = f"{para.folder}/log.txt"
para.my_info = f"{para.folder}/info.pth"
