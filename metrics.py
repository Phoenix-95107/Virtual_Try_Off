from typing import Set

import os
import torch
from cleanfid import fid as CleanFID
from DISTS_pytorch import DISTS
from prettytable import PrettyTable
from pytorch_fid import fid_score as PytorchFID
from PIL import Image
from torch.utils.data import Dataset
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchvision import transforms
from tqdm import tqdm
import shutil
from pathlib import Path

class EvalDataset(Dataset):
    def __init__(self, gt_folder, pred_files, height=1024):
        self.gt_folder = gt_folder
        self.pred_files = pred_files
        self.height = height
        self.data = self.prepare_data()
        self.to_tensor = transforms.ToTensor()
        self.pred_transform = transforms.Compose([
            transforms.Resize((1024, 768)),
            transforms.ToTensor(),
        ])
        self.gt_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def extract_id_from_filename(self, filename):
        if "inshop" in filename:
            filename = filename.split("_")[0]
            return filename
        filename = filename.split("_")[0]
        return filename

    def prepare_data(self):
        gt_files = scan_files_in_dir(self.gt_folder, postfix={'.jpg', '.png'})
        gt_dict = {self.extract_id_from_filename(
            file.name): file for file in gt_files}
        pred_files = self.pred_files
        pred_dict = {self.extract_id_from_filename(file.name): file for file in pred_files}
        pred_files = [pred_dict[pred_id] for pred_id in gt_dict.keys()]
        assert len(pred_files) == len(
            gt_dict), f"Number of gt files {len(gt_dict)} and pred files {len(pred_files)} do not match"

        tuples = []
        for pred_file in pred_files:
            pred_id = self.extract_id_from_filename(pred_file.name)
            tuples.append((gt_dict[pred_id].path, pred_file.path))
        return tuples

    def resize(self, img):
        w, h = img.size
        new_w = int(w * self.height / h)
        return img.resize((new_w, self.height), Image.LANCZOS)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        gt_path, pred_path = self.data[idx]
        gt, pred = self.resize(Image.open(gt_path)), self.resize(
            Image.open(pred_path))
        if gt.height != self.height:
            gt = self.resize(gt)
        if pred.height != self.height:
            pred = self.resize(pred)
        gt = self.gt_transform(gt)
        gt = (gt+1)/2
        pred = self.pred_transform(pred)
        return gt, pred


def scan_files_in_dir(directory, postfix: Set[str] = None, progress_bar: tqdm = None) -> list:
    """Scan a directory and return a list of files with the given postfix. Can accept also directory with subdirectories.
    WARNING: Put the folder containing only needed files and subfolders to avoid iterating over unecessary subfolders and files
    """
    file_list = []
    progress_bar = tqdm(total=0, desc=f"Scanning",
                        ncols=100) if progress_bar is None else progress_bar
    for entry in os.scandir(directory):
        if entry.is_file():
            if postfix is None or os.path.splitext(entry.path)[1] in postfix:
                file_list.append(entry)
                progress_bar.total += 1
                progress_bar.update(1)
        elif entry.is_dir():
            file_list += scan_files_in_dir(entry.path,
                                           postfix=postfix, progress_bar=progress_bar)
    return file_list


def copy_resize_gt(gt_folder, height, width):
    new_folder = f"{gt_folder}_{height}"
    if not os.path.exists(new_folder):
        os.makedirs(new_folder, exist_ok=True)
    for file in tqdm(os.listdir(gt_folder)):
        if os.path.exists(os.path.join(new_folder, file)):
            continue
        img = Image.open(os.path.join(gt_folder, file))
        img = img.resize((width, height), Image.LANCZOS)
        img.save(os.path.join(new_folder, file))
    return new_folder


@torch.no_grad()
def psnr(dataloader):
    psnr_score = 0
    psnr = PeakSignalNoiseRatio(data_range=1.0).to("cuda")
    for gt, pred in tqdm(dataloader, desc="Calculating PSNR"):
        batch_size = gt.size(0)
        gt, pred = gt.to("cuda"), pred.to("cuda")
        psnr_score += psnr(pred, gt) * batch_size
    return psnr_score / len(dataloader.dataset)


@torch.no_grad()
def ssim(dataloader):
    ssim_score = 0
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to("cuda")
    for gt, pred in tqdm(dataloader, desc="Calculating SSIM"):
        batch_size = gt.size(0)
        gt, pred = gt.to("cuda"), pred.to("cuda")
        ssim.update(pred, gt)
    return ssim.compute().item()


@torch.no_grad()
def lpips(dataloader):
    lpips_score = LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=True).to("cuda")
    score = 0
    for gt, pred in tqdm(dataloader, desc="Calculating LPIPS"):
        batch_size = gt.size(0)
        pred = pred.to("cuda")
        gt = gt.to("cuda")
        lpips_score.update(gt, pred)
    return lpips_score.compute().item()

def dists(dataloader):
    dists = DISTS().to("cuda")
    dists_score = 0
    for gt, pred in tqdm(dataloader, desc="DISTS"):
        batch_size = gt.size(0)
        pred = pred.to("cuda")
        gt = gt.to("cuda")
        score = dists(gt, pred, batch_average=True)
        dists_score+=score
    dists_score = dists_score / len(dataloader)
    return dists_score.item()

def structure_dirs(dataset, dataroot, category="all"):
    """_summary_

    Args:
        dataroot (_type_): _description_
        category (str, optional): _description_. Defaults to "all".
        Accumulate image paths into a temporary folder to generate stats of different categories. For example, tmp_folder = /tmp/dresscode/dresses"
        line.strip().split()[1] returns the cloth name, line.strip().split()[0] returns the vton image name
    """
    if dataset=="dresscode":
        dresscode_filesplit = os.path.join(dataroot, f"test_pairs_paired.txt")
        with open(dresscode_filesplit, 'r') as f:
            lines = f.read().splitlines()
        paths = []
        if category in ['lower_body', 'upper_body', 'dresses']:
            for line in lines:
                path = os.path.join(dataroot, category, 'cleaned_inshop_imgs', line.strip().split()[1]).replace(".jpg", "_cleaned.jpg")
                path_2 = os.path.join(dataroot, category, 'images', line.strip().split()[1]).replace(".jpg", "_cleaned.jpg")
                path_3 = os.path.join(dataroot, category, 'images', line.strip().split()[1])
                if os.path.exists(path):
                    paths.append(path)
                elif os.path.exists(path_2):
                    paths.append(path_2)
                elif os.path.exists(path_3):
                    paths.append(path_3)
            tmp_folder = f"/tmp/dresscode/{category}"
            shutil.rmtree(tmp_folder, ignore_errors=True)
            os.makedirs(tmp_folder, exist_ok=True)
            for path in tqdm(paths):
                shutil.copy(path, tmp_folder)
            return tmp_folder
        
        elif category == "all":
            for category in ['lower_body', 'upper_body', 'dresses']:
                for line in lines:
                    path = os.path.join(dataroot, category, 'cleaned_inshop_imgs', line.strip().split()[1]).replace(".jpg", "_cleaned.jpg")
                    path_2 = os.path.join(dataroot, category, 'images', line.strip().split()[1]).replace(".jpg", "_cleaned.jpg")
                    path_3 = os.path.join(dataroot, category, 'images', line.strip().split()[1])
                    if os.path.exists(path):
                        paths.append(path)
                    elif os.path.exists(path_2):
                        paths.append(path_2)
                    elif os.path.exists(path_3):
                        paths.append(path_3)
            tmp_folder = f"/tmp/dresscode/all"
            shutil.rmtree(tmp_folder, ignore_errors=True)
            os.makedirs(tmp_folder, exist_ok=True)
            for path in tqdm(paths):
                shutil.copy(path, tmp_folder)
            return tmp_folder
        
        else: raise NotImplementedError("category must be in ['lower_body', 'upper_body', 'dresses', 'all']")

    elif dataset=="vitonhd":
        return os.path.join(dataroot, 'test', 'cloth')
    
    else: raise NotImplementedError("dataset must be in ['dresscode', 'vitonhd']")


def compute_metrics(args):
    args.gt_folder = structure_dirs(args.dataset, args.gt_folder, args.category)
    pred_files = scan_files_in_dir(
        args.pred_folder, postfix={'.jpg', '.png'})
    if args.category == "all":
        for pred_file in tqdm(pred_files):
            new_pred_folder = f"/tmp/{args.dataset}/predictions"
            os.makedirs(new_pred_folder, exist_ok=True)
            shutil.copy(pred_file.path, new_pred_folder)
        pred_folder = new_pred_folder
    elif os.path.exists(os.path.join(args.pred_folder, args.category)):
        pred_folder = os.path.join(args.pred_folder, args.category)
    else:
        pred_folder = args.pred_folder
        
    
    gt_sample = os.listdir(args.gt_folder)[0]
    gt_img = Image.open(os.path.join(args.gt_folder, gt_sample))
    if args.height != gt_img.height:
        title = "--"*30 + \
            f"Resizing GT Images to height {args.height}" + "--"*30
        print(title)
        args.gt_folder = copy_resize_gt(args.gt_folder, args.height, args.width)
        print("-"*len(title))

    dataset = EvalDataset(args.gt_folder, pred_files, args.height)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, drop_last=False
    )

    header = []
    row = []
    print("FID and KID")
    header += ["PyTorch-FID"]
    pytorch_fid_ = PytorchFID.calculate_fid_given_paths(
        [args.gt_folder, pred_folder], batch_size=args.batch_size, device="cuda", dims=2048, num_workers=args.num_workers)
    row += ["{:.4f}".format(pytorch_fid_)]
    print("PyTorch-FID: {:.4f}".format(pytorch_fid_))
    header += ["Clean-FID", "Clean-KID"]
    clean_fid_ = CleanFID.compute_fid(args.gt_folder, pred_folder)
    print("Clean-FID: {:.4f}".format(clean_fid_))
    row += ["{:.4f}".format(clean_fid_)]
    clean_kid_ = CleanFID.compute_kid(args.gt_folder, pred_folder) * 1000
    print("Clean-KID: {:.4f}".format(clean_kid_))
    row += ["{:.4f}".format(clean_kid_)]
    
    if args.paired:
        print("SSIM, LPIPS, DISTS")
        header += ["SSIM", "LPIPS", "DISTS"]
        ssim_ = ssim(dataloader)
        print("SSIM: {:.4f}".format(ssim_))
        lpips_ = lpips(dataloader)
        print("LPIPS: {:.4f}".format(lpips_))
        dists_ = dists(dataloader)
        print("DISTS: {:.4f}".format(dists_))
        row += ["{:.4f}".format(ssim_),
                "{:.4f}".format(lpips_),"{:.4f}".format(dists_),]

    print("GT Folder  : ", args.gt_folder)
    print("Pred Folder: ", args.pred_folder)
    table = PrettyTable()
    table.field_names = header
    table.add_row(row)
    print(table)
    
    with open(os.path.join(Path(args.pred_folder), f"metrics_{args.category}.txt"), "w") as f:
        f.write(str(table))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_folder", type=str, required=True)
    parser.add_argument("--pred_folder", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--category", type=str, choices=["lower_body", "upper_body", "dresses", "all"], required=True)
    parser.add_argument("--paired", action="store_true")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--height", type=str, default=1024)
    parser.add_argument("--width", type=str, default=768)
    args = parser.parse_args()

    if args.gt_folder.endswith("/"):
        args.gt_folder = args.gt_folder[:-1]
    if args.pred_folder.endswith("/"):
        args.pred_folder = args.pred_folder[:-1]

    compute_metrics(args)
