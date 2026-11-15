"""Changing vision weights must immediately change temporal features, without inventing history."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from fsop.data.sim_dataset import Episode
from fsop.model.history import ACTIVE_HISTORY, encode_fresh_history


@pytest.mark.parametrize('k',[0,1,8,12])
def test_recorded_history_pairs_match_runtime_spacing_and_start_padding(k):
  frames=np.arange(14*6*2*3,dtype=np.uint8).reshape(14,6,2,3)
  episode=SimpleNamespace(frames={'road':frames})
  window,valid=Episode.history_frames(episode,'road',k)
  assert window.shape==(ACTIVE_HISTORY+1,6,2,3)
  for slot,j in enumerate(range(k-ACTIVE_HISTORY,k)):
    assert valid[slot]==(j>=0)
    if j>=0:
      np.testing.assert_array_equal(np.concatenate([window[slot],window[slot+1]],axis=0),
                                    Episode.frame_pair(episode,'road',j))


def test_history_uses_current_weights_and_detaches_runtime_queue():
  road=torch.arange(2*9*6*2*3,dtype=torch.float32).reshape(2,9,6,2,3)
  wide=road+3
  valid=torch.tensor([[False]*7+[True],[True]*8])
  scale=torch.nn.Parameter(torch.tensor(2.))
  def encode(a,b):
    return SimpleNamespace(hidden=((a.mean((1,2,3))+b.mean((1,2,3)))*scale)[:,None].expand(-1,512))
  first=encode_fresh_history(encode,road,wide,valid,chunk_size=2)
  assert first.shape==(2,24,512)
  assert not first.requires_grad
  assert torch.count_nonzero(first[0,:23])==0
  assert torch.count_nonzero(first[1,:16])==0
  with torch.no_grad():
    scale.fill_(3.)
  second=encode_fresh_history(encode,road,wide,valid,chunk_size=3)
  torch.testing.assert_close(second,first*1.5)
  expected=encode(torch.cat([road[0,7],road[0,8]],0)[None],
                  torch.cat([wide[0,7],wide[0,8]],0)[None]).hidden
  torch.testing.assert_close(second[0,-1],expected[0])


def test_prepared_policy_ignores_older_sixteen_slots():
  from fsop.model.build import PREP, SupercomboFS
  from fsop.model.history import validate_policy_history
  validate_policy_history(PREP / 'policy_fp32.onnx')
  model=SupercomboFS(image_head=False).eval()
  torch.manual_seed(12)
  history=torch.randn(1,24,512)
  hidden=torch.randn(1,512)
  desire=torch.randn(1,25,8)
  traffic=torch.tensor([[1.,0.]])
  with torch.no_grad():
    first=model.plan(history,hidden,desire,traffic)
    history[:,:-ACTIVE_HISTORY]=0
    second=model.plan(history,hidden,desire,traffic)
  for key in first:
    torch.testing.assert_close(first[key],second[key],rtol=0,atol=0)


def test_fresh_dataset_keeps_invalid_frames_and_never_reads_cached_features(tmp_path):
  import json

  from fsop.data.sim_dataset import SimDataset
  ep=tmp_path/'ep'
  ep.mkdir()
  (ep/'meta.json').write_text(json.dumps({'frames':4,'mount':{},'vehicle':{}}))
  for name,data in [('states',np.zeros((4,12))),('plan',np.zeros((4,33,15))),
                    ('desire',np.zeros((4,8))),('cones',np.empty((0,3))),
                    ('label_valid',np.array([True,False,True,True]))]:
    np.save(ep/f'{name}.npy',data)
  ds=SimDataset(tmp_path,episodes=['ep'],mode='images',history_mode='fresh',image_targets_on=False)
  frames=np.broadcast_to(np.arange(4,dtype=np.uint8)[:,None,None,None],(4,6,2,3))
  ds.eps[0].frames={'road':frames,'wide':frames}
  def reject_cache(*args):
    raise AssertionError('Fresh image training must not read a stale feature cache')
  ds._feats=reject_cache
  sample=ds[0]
  assert sample['k']==2
  assert sample['history_valid'].tolist()==[False]*6+[True,True]
  assert np.all(sample['history_frames_road'][-1]==1)  # invalid label, real past image
  assert np.count_nonzero(sample['history'])==0


def test_fresh_features_match_actual_runtime_queue():
  from fsop.model.build import build_base
  from fsop.model.temporal import TemporalState
  base=build_base().eval()
  rng=np.random.default_rng(71)
  road=rng.integers(0,256,(8,6,128,256),dtype=np.uint8)
  wide=rng.integers(0,256,(8,6,128,256),dtype=np.uint8)
  state=TemporalState()
  with torch.no_grad():
    for frame in range(32):
      pair=state.push_frames({'road':road[frame//4],'wide':wide[frame//4]})
      if frame%4==0:
        hidden=base.encode(torch.from_numpy(pair['road'][None]).float(),
                           torch.from_numpy(pair['wide'][None]).float()).hidden[0].numpy()
      else:
        hidden=np.zeros(512,np.float32)  # unsampled slots never reach policy history
      state.push_hidden(hidden)
    r=torch.from_numpy(np.concatenate([road[:1],road],axis=0)[None])
    w=torch.from_numpy(np.concatenate([wide[:1],wide],axis=0)[None])
    actual=encode_fresh_history(base.encode,r,w,torch.ones(1,8,dtype=torch.bool),chunk_size=1)
    batched=encode_fresh_history(base.encode,r,w,torch.ones(1,8,dtype=torch.bool))
  torch.testing.assert_close(actual[0],torch.from_numpy(state.features()),rtol=0,atol=0)
  torch.testing.assert_close(batched,actual,rtol=2e-4,atol=2e-4)


def test_fresh_history_rejects_cached_stage_and_preserves_run(tmp_path):
  from fsop.train.trainer import TrainCfg, Trainer
  cfg=TrainCfg(name='invalid',out_dir=str(tmp_path),history_mode='fresh',stage=1,device='cpu')
  with pytest.raises(ValueError,match='Fresh history requires image stage'):
    Trainer(cfg)
  assert not (tmp_path/'invalid').exists()


def test_fresh_history_refuses_a_policy_with_more_active_slots(tmp_path):
  import onnx
  from onnx import TensorProto, helper, numpy_helper

  from fsop.model.history import validate_policy_history
  nodes=[helper.make_node('Transpose',['fb_full'],['transposed'],perm=[1,0,2]),
         helper.make_node('GatherND',['transposed','indices'],['selected'])]
  graph=helper.make_graph(nodes,'unsupported',
                         [helper.make_tensor_value_info('fb_full',TensorProto.FLOAT,[1,25,512])],
                         [helper.make_tensor_value_info('selected',TensorProto.FLOAT,[10,1,512])],
                         [numpy_helper.from_array(np.arange(-10,0)[:,None],'indices')])
  path=tmp_path/'policy.onnx'
  onnx.save(helper.make_model(graph),path)
  with pytest.raises(ValueError,match='eight previous'):
    validate_policy_history(path)
