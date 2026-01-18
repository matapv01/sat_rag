import torch.optim as optim

from OpenKE.openke.data import TestDataLoader
from OpenKE.openke.config import Tester
from OpenKE.openke.module.model import RotatE
from src.text_encoder import TextEncoderPretrained
from src.hka_trainer import train_hka_joint

import time
import torch
from src.data_loader import FB15K237Graph


from torch.utils.data import DataLoader
from src.data_loader import LocalAlignmentDataset, collate_fn


def log(msg: str):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)



device = "cuda" if torch.cuda.is_available() else "cpu"



graph_obj = FB15K237Graph(
    data_dir="data/fb15k237",
    entity2id_path="/home/naver/MinhPV/sat_rag/OpenKE/benchmarks/FB15K237/entity2id.txt",
    relation2id_path="/home/naver/MinhPV/sat_rag/OpenKE/benchmarks/FB15K237/relation2id.txt",
)


dataset = LocalAlignmentDataset(graph_obj, k_hop=2)

dataloader = DataLoader(
    dataset,
    batch_size=2,          # giữ nhỏ cho joint TE+GE
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=0,
    pin_memory=(device == "cuda"),
)
grad_accum_steps = 8
doc_max_length = 96


def eval_openke_linkpred(tag, rotate_model, test_dataloader, use_gpu=True):
    tester = Tester(model=rotate_model, data_loader=test_dataloader, use_gpu=use_gpu)
    mrr, mr, hit10, hit3, hit1 = tester.run_link_prediction(type_constrain=False)
    log(f"{tag} | MRR={mrr:.6f} MR={mr:.3f} H@10={hit10:.6f} H@3={hit3:.6f} H@1={hit1:.6f}")
    return {"MRR": float(mrr), "MR": float(mr), "H@10": float(hit10), "H@3": float(hit3), "H@1": float(hit1)}

def make_opt_joint(rotate, text_encoder, lr_ge=1e-4, lr_te=2e-6):
    return optim.AdamW([
        {"params": rotate.parameters(), "lr": lr_ge},
        {"params": text_encoder.parameters(), "lr": lr_te},
    ], weight_decay=0.01)

def make_opt_ge_only_all(rotate, lr_ge=1e-4):
    return optim.AdamW([{"params": rotate.parameters(), "lr": lr_ge}], weight_decay=0.0)

def make_opt_ge_only_entity(rotate, lr_ge=1e-4):
    # freeze all, train only entity embedding
    for p in rotate.parameters():
        p.requires_grad = False
    ent_w = rotate.ent_embeddings.weight
    ent_w.requires_grad = True
    return optim.AdamW([ent_w], lr=lr_ge, weight_decay=0.0)

# ---- OpenKE loaders/models ----
test_dataloader = TestDataLoader("/home/naver/MinhPV/sat_rag/OpenKE/benchmarks/FB15K237/", "link")

def load_rotate_from_ckpt(ckpt_path):
    m = RotatE(
        ent_tot=test_dataloader.get_ent_tot(),
        rel_tot=test_dataloader.get_rel_tot(),
        dim=64, margin=12.0, epsilon=2.0
    )
    m.load_checkpoint(ckpt_path)
    return m

ckpt = "/home/naver/MinhPV/sat_rag/OpenKE/checkpoint/rotate.ckpt"

results = {}

# MODE 1: metric for GE
log("\n==================== MODE 1: eval_GE(OpenKE Tester) ====================")
rotate = load_rotate_from_ckpt(ckpt).to(device)
print("GE entity emb shape:", rotate.ent_embeddings.weight.shape)
results["mode1_eval_ge"] = eval_openke_linkpred("MODE 1", rotate, test_dataloader, use_gpu=(device=="cuda"))

# helper: fresh TE each mode (fair)
def fresh_te():
    return TextEncoderPretrained(model_name="prajjwal1/bert-tiny", entity2text=graph_obj.entity2text).to(device)

# MODE 2: train TE+GE local
log("\n==================== MODE 2: train_TE+GE_local ====================")
rotate = load_rotate_from_ckpt(ckpt).to(device)
text_encoder = fresh_te()

print("GE entity emb shape:", rotate.ent_embeddings.weight.shape)
print("TE hidden size:", text_encoder.hidden_size)


opt = make_opt_joint(rotate, text_encoder, lr_ge=1e-4, lr_te=2e-6)
train_hka_joint(rotate, text_encoder, dataloader, opt, device=device,
                epochs_local=3, epochs_global=0, train_te=True, train_ge=True,
                grad_accum_steps=grad_accum_steps, doc_max_length=doc_max_length)
results["mode2_tege_local"] = eval_openke_linkpred("MODE 2", rotate, test_dataloader, use_gpu=(device=="cuda"))

# MODE 3: train TE+GE (GE only) local
log("\n==================== MODE 3: train_GE_only_local ====================")
rotate = load_rotate_from_ckpt(ckpt).to(device)
text_encoder = fresh_te()  # sẽ bị freeze trong trainer vì train_te=False
opt = make_opt_ge_only_entity(rotate, lr_ge=1e-4)   # hoặc make_opt_ge_only_all
train_hka_joint(rotate, text_encoder, dataloader, opt, device=device,
                epochs_local=3, epochs_global=0, train_te=False, train_ge=True,
                grad_accum_steps=grad_accum_steps, doc_max_length=doc_max_length)
results["mode3_geonly_local"] = eval_openke_linkpred("MODE 3", rotate, test_dataloader, use_gpu=(device=="cuda"))

# MODE 4: train TE+GE global
log("\n==================== MODE 4: train_TE+GE_global ====================")
rotate = load_rotate_from_ckpt(ckpt).to(device)
text_encoder = fresh_te()
opt = make_opt_joint(rotate, text_encoder, lr_ge=1e-4, lr_te=2e-6)
train_hka_joint(rotate, text_encoder, dataloader, opt, device=device,
                epochs_local=0, epochs_global=3, train_te=True, train_ge=True,
                grad_accum_steps=grad_accum_steps, doc_max_length=doc_max_length)
results["mode4_tege_global"] = eval_openke_linkpred("MODE 4", rotate, test_dataloader, use_gpu=(device=="cuda"))

# MODE 5: train TE+GE (GE only) global
log("\n==================== MODE 5: train_GE_only_global ====================")
rotate = load_rotate_from_ckpt(ckpt).to(device)
text_encoder = fresh_te()
opt = make_opt_ge_only_entity(rotate, lr_ge=1e-4)
train_hka_joint(rotate, text_encoder, dataloader, opt, device=device,
                epochs_local=0, epochs_global=3, train_te=False, train_ge=True,
                grad_accum_steps=grad_accum_steps, doc_max_length=doc_max_length)
results["mode5_geonly_global"] = eval_openke_linkpred("MODE 5", rotate, test_dataloader, use_gpu=(device=="cuda"))

# MODE 6: train TE+GE HKA
log("\n==================== MODE 6: train_TE+GE_HKA ====================")
rotate = load_rotate_from_ckpt(ckpt).to(device)
text_encoder = fresh_te()
opt = make_opt_joint(rotate, text_encoder, lr_ge=1e-4, lr_te=2e-6)
train_hka_joint(rotate, text_encoder, dataloader, opt, device=device,
                epochs_local=2, epochs_global=1, train_te=True, train_ge=True,
                grad_accum_steps=grad_accum_steps, doc_max_length=doc_max_length)
results["mode6_tege_hka"] = eval_openke_linkpred("MODE 6", rotate, test_dataloader, use_gpu=(device=="cuda"))

# MODE 7: train TE+GE (GE only) HKA
log("\n==================== MODE 7: train_GE_only_HKA ====================")
rotate = load_rotate_from_ckpt(ckpt).to(device)
text_encoder = fresh_te()
opt = make_opt_ge_only_entity(rotate, lr_ge=1e-4)
train_hka_joint(rotate, text_encoder, dataloader, opt, device=device,
                epochs_local=2, epochs_global=1, train_te=False, train_ge=True,
                grad_accum_steps=grad_accum_steps, doc_max_length=doc_max_length)
results["mode7_geonly_hka"] = eval_openke_linkpred("MODE 7", rotate, test_dataloader, use_gpu=(device=="cuda"))

log("\n==================== SUMMARY ====================")
for k, v in results.items():
    log(f"{k}: {v}")
