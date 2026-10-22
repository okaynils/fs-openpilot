"""Training loop (plain PyTorch; MPS / CUDA / CPU).

stage 1: trunk frozen, features from the cache -> trains policy (plan head) + BEV cone head. Fast.
stage 2: images in, last trunk stages unfrozen (L2-SP to pretrained weights), + image cone head,
         optionally mixed with FSOCO batches (image head only).
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from fsop.config import save_config
from fsop.data.sim_dataset import SimDataset, collate
from fsop.eval.bev_metrics import BevScorer, decode_bev, gt_from_targets
from fsop.geometry.op_frames import op_to_ego_plan
from fsop.model.build import SupercomboFS
from fsop.sim.expert import T_IDXS
from fsop.train import losses as L
from fsop.train.device import autocast_ctx, pick_device


@dataclass
class TrainCfg:
  name: str = "run"
  out_dir: str = "runs"
  sim_root: str = "data/sim/v1"
  cache_tag: str = "op_b361e952"
  fsoco_root: str = ""
  fsoco_every: int = 4  # one FSOCO batch every N sim batches (stage 2)
  stage: int = 1
  trunk: str = "frozen"  # frozen | last | all
  init_ckpt: str = ""
  batch_size: int = 64
  num_workers: int = 4
  steps: int = 4000
  lr: float = 3e-4
  lr_policy: float = 0.3  # multiplier for the pretrained policy
  lr_trunk: float = 0.1
  weight_decay: float = 1e-4
  warmup: int = 200
  grad_clip: float = 1.0
  l2sp: float = 1e-3
  w_plan: float = 1.0
  w_bev: float = 1.0
  w_img: float = 1.0
  val_every: int = 500
  val_batches: int = 40
  log_every: int = 25
  seed: int = 0
  device: str = "auto"
  max_episodes: int = 0  # >0: subsample train episodes (smoke tests)
  viz_samples: int = 6
  extra: dict = field(default_factory=dict)


class Trainer:
  def __init__(self, cfg: TrainCfg):
    self.cfg = cfg
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    self.device = pick_device(cfg.device)
    self.run_dir = Path(cfg.out_dir) / cfg.name
    (self.run_dir / "ckpt").mkdir(parents=True, exist_ok=True)
    save_config(cfg, self.run_dir / "config.yaml")
    mode = "cached" if cfg.stage == 1 else "images"
    self.train_ds = SimDataset(cfg.sim_root, "train", mode, cfg.cache_tag)
    if cfg.max_episodes:
      self.train_ds = SimDataset(cfg.sim_root, "train", mode, cfg.cache_tag,
                                 episodes=[e.path.name for e in self.train_ds.eps[: cfg.max_episodes]])
    self.val_ds = SimDataset(cfg.sim_root, "val", mode, cfg.cache_tag)
    self.model = SupercomboFS(image_head=cfg.stage >= 2)
    if cfg.init_ckpt:
      sd = torch.load(cfg.init_ckpt, map_location="cpu", weights_only=False)["model"]
      missing, unexpected = self.model.load_state_dict(sd, strict=False)
      print(f"init from {cfg.init_ckpt}: missing {len(missing)} unexpected {len(unexpected)}")
    self.model.set_trunk_trainable(cfg.trunk if cfg.stage >= 2 else "frozen")
    self.model.to(self.device)
    self.ref = {n: p.detach().clone() for n, p in self.model.base.vision.named_parameters() if p.requires_grad}
    groups = [
      {"params": [p for p in self.model.base.policy.parameters() if p.requires_grad], "lr": cfg.lr * cfg.lr_policy},
      {"params": list(self.model.bev.parameters()), "lr": cfg.lr},
    ]
    if self.model.img is not None:
      groups.append({"params": list(self.model.img.parameters()), "lr": cfg.lr})
    trunk = [p for p in self.model.base.vision.parameters() if p.requires_grad]
    if trunk:
      groups.append({"params": trunk, "lr": cfg.lr * cfg.lr_trunk, "weight_decay": 0.0})
    for g in groups:
      g["base_lr"] = g["lr"]
    self.opt = torch.optim.AdamW(groups, weight_decay=cfg.weight_decay)
    self.fsoco = None
    if cfg.fsoco_root and cfg.stage >= 2:
      from fsop.data.fsoco import FsocoDataset

      self.fsoco = FsocoDataset(cfg.fsoco_root, "train")
    from torch.utils.tensorboard import SummaryWriter

    self.tb = SummaryWriter(str(self.run_dir / "tb"))
    self.step = 0
    self.best = math.inf
    n_tr = sum(p.numel() for g in groups for p in g["params"])
    print(f"device {self.device} | train {len(self.train_ds)} samples ({len(self.train_ds.eps)} eps) | "
          f"val {len(self.val_ds)} | trainable params {n_tr / 1e6:.1f}M")

  # ------------------------------------------------------------------------------------------
  def _loader(self, ds, shuffle: bool, bs: int | None = None):
    return DataLoader(ds, batch_size=bs or self.cfg.batch_size, shuffle=shuffle, num_workers=self.cfg.num_workers,
                      collate_fn=collate, drop_last=shuffle, persistent_workers=self.cfg.num_workers > 0,
                      prefetch_factor=4 if self.cfg.num_workers > 0 else None)

  def _to(self, b: dict) -> dict:
    return {k: (v.to(self.device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in b.items()}

  def forward(self, b: dict) -> dict:
    if self.cfg.stage == 1:
      return self.model.forward_cached(b)
    return self.model.forward_images(b)

  def losses(self, out: dict, b: dict) -> dict[str, torch.Tensor]:
    ls = {"plan": self.cfg.w_plan * L.plan_nll(out["plan_mu"], out["plan_logstd"], b["plan"])}
    for k, v in L.bev_loss(out["bev"], b).items():
      ls[k] = self.cfg.w_bev * v
    if "img" in out:
      for k, v in L.img_loss(out["img"], b).items():
        ls[k] = self.cfg.w_img * v
    return ls

  def l2sp(self) -> torch.Tensor:
    if not self.ref or self.cfg.l2sp <= 0:
      return torch.zeros((), device=self.device)
    tot = sum(((p - self.ref[n]) ** 2).sum() for n, p in self.model.base.vision.named_parameters() if n in self.ref)
    return self.cfg.l2sp * tot

  def lr_at(self, step: int) -> float:
    c = self.cfg
    if step < c.warmup:
      return (step + 1) / c.warmup
    t = (step - c.warmup) / max(c.steps - c.warmup, 1)
    return 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(t, 1.0)))

  # ------------------------------------------------------------------------------------------
  def fit(self) -> None:
    c = self.cfg
    self.validate(tag="baseline")  # untouched model / init checkpoint as reference
    loader = self._loader(self.train_ds, True)
    fs_iter = iter(self._loader(self.fsoco, True, max(c.batch_size // 2, 2))) if self.fsoco else None
    it = iter(loader)
    t0 = time.time()
    self.model.train()
    while self.step < c.steps:
      try:
        b = next(it)
      except StopIteration:
        it = iter(loader)
        b = next(it)
      b = self._to(b)
      scale = self.lr_at(self.step)
      for g in self.opt.param_groups:
        g["lr"] = g["base_lr"] * scale
      with autocast_ctx(self.device):
        out = self.forward(b)
        ls = self.losses(out, b)
        if fs_iter is not None and self.step % c.fsoco_every == 0:
          try:
            fb = next(fs_iter)
          except StopIteration:
            fs_iter = iter(self._loader(self.fsoco, True, max(c.batch_size // 2, 2)))
            fb = next(fs_iter)
          fb = self._to(fb)
          fo = self.model.forward_images(fb, heads=("img",))
          for k, v in L.img_loss(fo["img"], fb).items():
            ls["fsoco_" + k] = c.w_img * v
        ls["l2sp"] = self.l2sp()
        loss = sum(ls.values())
      self.opt.zero_grad(set_to_none=True)
      loss.backward()
      gn = torch.nn.utils.clip_grad_norm_([p for g in self.opt.param_groups for p in g["params"]], c.grad_clip)
      self.opt.step()
      self.step += 1
      if self.step % c.log_every == 0:
        sps = c.log_every * c.batch_size / (time.time() - t0)
        t0 = time.time()
        for k, v in ls.items():
          self.tb.add_scalar(f"train/{k}", float(v.detach()), self.step)
        self.tb.add_scalar("train/loss", float(loss.detach()), self.step)
        self.tb.add_scalar("train/grad_norm", float(gn), self.step)
        self.tb.add_scalar("train/lr_scale", scale, self.step)
        print(f"step {self.step:6d} loss {float(loss):.4f} " + " ".join(f"{k} {float(v):.3f}" for k, v in ls.items())
              + f" | {sps:.0f} samples/s")
      if self.step % c.val_every == 0 or self.step == c.steps:
        self.validate()
        self.model.train()
    self.save("last")

  def save(self, name: str) -> Path:
    p = self.run_dir / "ckpt" / f"{name}.pt"
    torch.save({"model": self.model.state_dict(), "step": self.step, "cfg": asdict(self.cfg)}, p)
    return p

  # ------------------------------------------------------------------------------------------
  @torch.no_grad()
  def validate(self, tag: str | None = None) -> dict:
    c = self.cfg
    self.model.eval()
    if not hasattr(self, "_val_loader"):
      # fixed random subset of the validation set (same samples every time -> comparable curves)
      rng = np.random.default_rng(0)
      n = min(c.val_batches * c.batch_size, len(self.val_ds))
      sub = torch.utils.data.Subset(self.val_ds, sorted(rng.choice(len(self.val_ds), n, replace=False).tolist()))
      self._val_loader = DataLoader(sub, batch_size=c.batch_size, shuffle=False, num_workers=c.num_workers,
                                    collate_fn=collate, persistent_workers=c.num_workers > 0)
    agg: dict[str, list] = {}
    scorer = BevScorer()
    errs = {t: [] for t in (1.0, 2.0, 4.0)}
    viz = []
    for b in self._val_loader:
      b = self._to(b)
      out = self.forward(b)
      for k, v in self.losses(out, b).items():
        agg.setdefault(k, []).append(float(v))
      mu = op_to_ego_plan(out["plan_mu"].cpu().numpy())
      gt = op_to_ego_plan(b["plan"].cpu().numpy())
      for t in errs:
        i = int(np.argmin(np.abs(T_IDXS - t)))
        errs[t].append(np.c_[np.abs(mu[:, i, 0] - gt[:, i, 0]), np.abs(mu[:, i, 1] - gt[:, i, 1])])
      preds = decode_bev(out["bev"]["heat"], out["bev"]["off"])
      gts = gt_from_targets(b["bev_heat"], b["bev_off"], b["bev_pos"])
      for (pxy, pc, _), (gxy, gc) in zip(preds, gts, strict=True):
        scorer.add(pxy, pc, gxy, gc)
      if len(viz) < c.viz_samples:
        viz.append((b, out, mu, gt, preds))
    res = {k: float(np.mean(v)) for k, v in agg.items()}
    res["loss"] = float(sum(res.values()))
    for t, e in errs.items():
      e = np.concatenate(e)
      res[f"long_err_{t:.0f}s"] = float(e[:, 0].mean())
      res[f"lat_err_{t:.0f}s"] = float(e[:, 1].mean())
    bev = scorer.summary()
    res.update({"bev_f1": bev["f1"], "bev_precision": bev["precision"], "bev_recall": bev["recall"],
                "bev_pos_err": bev["pos_err_m"] or 0.0, "bev_swap": bev["blue_yellow_swap_rate"]})
    step_tag = tag or f"step{self.step}"
    for k, v in res.items():
      self.tb.add_scalar(f"val/{k}", v, self.step)
    print(f"[val {step_tag}] " + " ".join(f"{k} {v:.3f}" for k, v in res.items()))
    with open(self.run_dir / "val.jsonl", "a") as f:
      f.write(json.dumps({"step": self.step, "tag": step_tag, **res, "bev": bev}) + "\n")
    self._viz(viz, step_tag)
    if tag is None and res["loss"] < self.best:
      self.best = res["loss"]
      self.save("best")
    return res

  def _viz(self, viz, tag: str) -> None:
    """Per-validation Rerun recording of fixed samples: GT vs predicted plan + BEV cones."""
    try:
      import rerun as rr

      from fsop.viz import rr_log
    except ImportError:
      return
    (self.run_dir / "viz").mkdir(exist_ok=True)
    rec = rr.RecordingStream("fsop_val", recording_id=f"{self.cfg.name}_{tag}")
    rec.save(str(self.run_dir / "viz" / f"{tag}.rrd"))
    with rec:
      for i, (b, out, mu, gt, preds) in enumerate(viz):
        rr.set_time("sample", sequence=i)
        j = 0
        rr_log.log_plan("gt", gt[j], (40, 220, 80))
        std = np.exp(out["plan_logstd"][j].cpu().numpy())
        rr_log.log_plan("pred", mu[j], (230, 60, 200), std)
        pxy, pc, ps = preds[j]
        rr_log.log_cones_ego("cones_pred", pxy, pc, ps)
        gts = gt_from_targets(b["bev_heat"][j:j + 1], b["bev_off"][j:j + 1], b["bev_pos"][j:j + 1])[0]
        rr_log.log_cones_ego("cones_gt", gts[0], gts[1])
        rr.log("bev/gt", rr.Image(rr_log.bev_image(b["bev_heat"][j].cpu().numpy())))
        rr.log("bev/pred", rr.Image(rr_log.bev_image(torch.sigmoid(out["bev"]["heat"][j]).cpu().numpy())))
        if "img" in b:
          from fsop.geometry.yuv import model_to_rgb

          rr.log("world/ego/cam_wide/image", rr.Image(model_to_rgb(b["big_img"][j, 6:].cpu().numpy())))
