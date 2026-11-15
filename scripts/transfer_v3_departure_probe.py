"""Bounded departure trace and identical-prefix oracle-plan intervention; diagnostic only."""
import hashlib
import json
import tarfile
from pathlib import Path

import cv2
import numpy as np
import yaml
from transfer_diagnostics import environment

from fsop.data.targets import visible_cones, world_to_ego
from fsop.eval.closed_loop import ModelPolicy, mission_brake
from fsop.eval.controller import PlanPurePursuit
from fsop.geometry.camera import CAMERA_MODELS, CameraModel, project_world
from fsop.geometry.mounting import body_occluded
from fsop.geometry.warp import _cam_rot
from fsop.sim.expert import T_IDXS

root=Path('runs/transfer_departure_probe_v3')
if root.exists():
  raise ValueError('Preserve existing diagnostic')
root.mkdir(parents=True)
configuration=yaml.safe_load(Path('configs/eval/transfer_v2_val.yaml').read_text())['configurations'][2]
checkpoint='runs/transfer_guarded_v3/ckpt/step_000500.pt'
results=[]
for mode in ('policy','oracle_after_12s'):
  env=environment(configuration,True,98011,5101,False,60.)
  try:
    policy=ModelPolicy.from_checkpoint(checkpoint)
    policy.bev=False
    follower=PlanPurePursuit(env.vp)
    obs=env.observe()
    rows, panels, predictions, targets, plan_times=[],[],[],[],[]
    target=None
    while not env.done:
      steer,accel,info=policy(obs,env)
      intervened=mode=='oracle_after_12s' and obs.frame>=240
      sampled=obs.frame%4==0
      if sampled or intervened:
        target=env.label()
      oracle_steer,oracle_accel=follower(target,obs.state.vx)
      if intervened:
        steer,accel=oracle_steer,mission_brake(oracle_accel,obs,env)
      reference=env.mission._at(obs.s,env.mission.racing_line.offset)
      cones=world_to_ego(env.cones_xy,obs.state.x,obs.state.y,obs.state.yaw)
      points=np.c_[cones,np.zeros(len(cones))]
      visible=visible_cones(cones,env.T_ego_cam()) & ~env.referee.down
      physical=np.zeros(len(cones),bool)
      for camera in env.profile.cameras:
        offset=np.eye(4)
        offset[:3,:3]=_cam_rot(*np.deg2rad([camera.roll_deg,camera.pitch_deg,camera.yaw_deg]))
        physical |= project_world(points,np.eye(4),env.T_ego_cam()@offset,
                                  CameraModel(camera.name,camera.K,camera.width,camera.height))[2]
      visible &= physical & ~body_occluded(points,env.T_ego_cam(),env.vp)
      rows.append(dict(t=obs.t,x=obs.state.x,y=obs.state.y,yaw=obs.state.yaw,speed=obs.state.vx,
                       lateral=obs.lat,reference_error=obs.lat-reference,progress=obs.s,
                       steer=steer,accel=accel,oracle_steer=oracle_steer,
                       visible_cones=int(visible.sum()),intervened=intervened,
                       target_fresh=sampled or intervened,
                       target_feasible=env.expert.last_rollout.get('contact_samples',0)==0,
                       events=[e.kind for e in obs.events]))
      if sampled:
        predictions.append(info['plan'])
        targets.append(target)
        plan_times.append(obs.t)
      if obs.frame%80==0:
        image=obs.images['wide'].copy()
        for plan,color in ((target,(30,230,60)),(info['plan'],(255,50,70))):
          pts=np.c_[plan[:,:2],np.zeros(len(plan))]
          uv,_,ok=project_world(pts,np.eye(4),env.T_ego_cam(),CAMERA_MODELS['wide'])
          ok &= ~body_occluded(pts,env.T_ego_cam(),env.vp)
          for p in uv[ok]:
            cv2.circle(image,tuple(p.astype(int)),3,color,-1)
        label=f"{mode} t={obs.t:.0f} v={obs.state.vx:.1f} err={obs.lat-reference:.2f} visible={visible.sum()}"
        cv2.putText(image,label,(5,20),cv2.FONT_HERSHEY_SIMPLEX,.4,(255,255,255),1)
        panels.append(image)
      obs=env.step(steer,accel,add_noise=False)
    for k,plan in enumerate(predictions):
      index=4*k
      if index+20>=len(rows):
        break
      row,future=rows[index],rows[index+20]
      xy=np.array([np.interp(1.,T_IDXS,plan[:,axis]) for axis in (0,1)])
      c,s=np.cos(row['yaw']),np.sin(row['yaw'])
      predicted_world=np.array([row['x'],row['y']])+np.array([[c,-s],[s,c]])@xy
      row['one_second_tracking_error']=float(np.linalg.norm(predicted_world-[future['x'],future['y']]))
    record=dict(mode=mode,rows=rows,summary=env.referee.summary(),final_progress=env.s,
                renderer=env.renderer.base.win.getGsg().getDriverRenderer())
    results.append(record)
    (root/f'{mode}.json').write_text(json.dumps(record,indent=2))
    np.savez_compressed(root/f'{mode}_plans.npz',times=plan_times,predicted=predictions,target=targets)
    cv2.imwrite(str(root/f'{mode}.jpg'),np.vstack(panels)[...,::-1])
    print(mode,record['summary'], 'final reference error',rows[-1]['reference_error'],flush=True)
  finally:
    env.renderer.close()
keys=('x','y','yaw','speed','steer','accel','progress','reference_error')
prefix=[np.array([[r[k] for k in keys] for r in result['rows'] if r['t']<12.-1e-7]) for result in results]
assert prefix[0].shape==prefix[1].shape
prefix_error=float(np.max(np.abs(prefix[0]-prefix[1])))
assert prefix_error<1e-6, f'Counterfactual prefix mismatch: {prefix_error}'
report=dict(configuration=configuration,checkpoint=checkpoint,
            checkpoint_sha256=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
            track_seed=98011,scene_seed=5101,seconds=60,switch_frame=240,
            prefix_max_error=prefix_error,results=results,
            privileged='Both: simulated wheel speed, route desire and mission launch/stop assistance. '
                       'Oracle intervention additionally uses true state/dynamics, full racing line and cone map.')
(root/'report.json').write_text(json.dumps(report,indent=2))
with tarfile.open(root/'source.tar.gz','w:gz') as archive:
  for folder in ('src','scripts','configs','tests','pyproject.toml','uv.lock'):
    archive.add(folder,filter=lambda item:None if '__pycache__' in item.name else item)
print('prefix_max_error',prefix_error,flush=True)
