"""Reference reuse must fail closed when paired dynamics or inference code changes."""
import json
import runpy
from pathlib import Path

import pytest

merge = runpy.run_path(str(Path(__file__).parents[1]/'scripts/transfer_v3_merge.py'))['merge']


def fixture(tmp_path, changed):
  paths=[]
  for number in (0,1):
    root=tmp_path/str(number)
    root.mkdir()
    identity=dict(configuration={'name':'c'},track_seed=1,scene_seed=2,recovery=False,max_seconds=100,
                  source_sha256={'src/fsop/eval/closed_loop.py':'same',
                                 'src/fsop/sim/transfer_collection.py':'old'})
    if number:
      identity['source_sha256'][changed]='new'
    (root/'c_1_2.json').write_text(json.dumps({'identity':identity}))
    row=dict(configuration='c',track_seed=1,scene_seed=2,recovery=False,
             **{'pass':True,'complete':True,'doo':0,'oc':0,'n_laps':1,'dq':False})
    report=dict(suite={'configurations':[{'name':'c'}],'track_seeds':[1],'scene_seeds':[2]},
                recovery_rule='same',time_limits_seconds={'rover':100},checkpoints=[str(number)],
                results=[dict(row,checkpoint='expert'),dict(row,checkpoint=str(number))])
    path=root/'report.json'
    path.write_text(json.dumps(report))
    paths.extend((path,root))
  return paths


def test_collection_only_change_permits_paired_reference_reuse(tmp_path):
  paths=fixture(tmp_path,'src/fsop/sim/transfer_collection.py')
  report=merge(*paths,tmp_path/'merged.json')
  assert len(report['results'])==3
  assert report['source_audit'][0]['collection_only_differences']==['src/fsop/sim/transfer_collection.py']


@pytest.mark.parametrize('changed',['src/fsop/eval/closed_loop.py','src/fsop/model/new_module.py'])
def test_changed_or_new_inference_source_rejects_reuse(tmp_path,changed):
  paths=fixture(tmp_path,changed)
  with pytest.raises(ValueError,match='Inference/simulation source changed'):
    merge(*paths,tmp_path/'merged.json')


def test_expert_outcome_disagreement_rejects_reuse(tmp_path):
  paths=fixture(tmp_path,'src/fsop/sim/transfer_collection.py')
  report=json.loads(paths[2].read_text())
  report['results'][0]['doo']=1
  paths[2].write_text(json.dumps(report))
  with pytest.raises(ValueError,match='Expert repeat disagrees'):
    merge(*paths,tmp_path/'merged.json')
