"""Seeded scenes using the Apache-2.0 Menagerie SO-101 collision model."""
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np

ASSET = Path(__file__).parent / 'assets' / 'so101'
DEFAULTS = dict(cube_size=.05, container_size=.15, rim_height=.012, floor_z=.004,
                carry_height=.085, grasp_height=.028, release_height=.065,
                randomize_positions=True, randomize_colors=True, randomize_lighting=True,
                randomize_camera=True, width=448, height=448, timestep=.002,
                action_seconds=.65, settle_seconds=.3)
COLORS={'red':[.85,.08,.07,1], 'blue':[.08,.27,.88,1], 'green':[.1,.65,.22,1], 'purple':[.58,.18,.72,1], 'orange':[.95,.45,.06,1]}

SIDE_CAMERA_DEFAULTS=dict(name='fixed_side_v1',position=(.54,-.34,.25),
                          target=(.2375,0.,.055),fovy=46.,width=448,height=448)

def configured_side_camera(config):
    """Return a validated fixed-camera spec, or None when the option is off."""
    value=config.get('side_camera',False)
    if value is False or value is None:return None
    if value is True:value={}
    if not isinstance(value,dict):raise ValueError('side_camera must be false, true, or a configuration mapping')
    spec={**SIDE_CAMERA_DEFAULTS,**value}
    if not isinstance(spec['name'],str) or not spec['name']:raise ValueError('side camera name is required')
    position=np.asarray(spec['position'],dtype=float);target=np.asarray(spec['target'],dtype=float)
    if position.shape!=(3,) or target.shape!=(3,) or not np.isfinite(np.r_[position,target]).all():
        raise ValueError('side camera position and target must be finite XYZ triples')
    if np.linalg.norm(target-position)<1e-6:raise ValueError('side camera position and target must differ')
    if not np.isfinite(spec['fovy']) or not 1<float(spec['fovy'])<179:raise ValueError('side camera fovy is invalid')
    if int(spec['width'])!=config['width'] or int(spec['height'])!=config['height']:
        raise ValueError('side camera dimensions must match the shared renderer dimensions')
    forward=(target-position)/np.linalg.norm(target-position);up=np.array([0.,0.,1.])
    right=np.cross(forward,up)
    if np.linalg.norm(right)<1e-6:raise ValueError('side camera look direction is parallel to world up')
    right/=np.linalg.norm(right);camera_up=np.cross(right,forward);camera_up/=np.linalg.norm(camera_up)
    return {**spec,'position':position.tolist(),'target':target.tolist(),
            'xyaxes':np.r_[right,camera_up].tolist(),'fovy':float(spec['fovy']),
            'width':int(spec['width']),'height':int(spec['height'])}

def build_scene(config, seed, task):
    rng=np.random.default_rng(seed); root=ET.parse(ASSET/'so101.xml').getroot()
    root.find('compiler').set('meshdir',str(ASSET/'assets'))
    opt=root.find('option');opt.set('timestep',str(config['timestep']));opt.set('iterations','50')
    root.find('visual').append(ET.Element('global',offwidth=str(config['width']),offheight=str(config['height'])))
    # Preserve actual mechanism and collision geometry; use high-friction rubber pads.
    for g in root.findall(".//default[@class='collision_gripper']/geom")+root.findall(".//default[@class='collision_gripper_mesh']/geom"):
        g.set('friction','2.5 .02 .002');g.set('solref','.006 1')
    site=root.find(".//site[@name='gripperframe']");site.set('pos',f'{config.get("tcp_x",.027)} 0 -.095')
    w=root.find('worldbody')
    ET.SubElement(w,'geom',name='table',type='plane',size='1 1 .02',rgba='.72 .76 .79 1',friction='1 .01 .001')
    light=.85+rng.uniform(-.12,.12) if config['randomize_lighting'] else .85
    ET.SubElement(w,'light',pos='.1 -.2 .8',dir='0 0 -1',diffuse=f'{light} {light} {light}')
    jitter=rng.uniform(-.004,.004,2) if config['randomize_camera'] else np.zeros(2)
    ET.SubElement(w,'camera',name='overhead',pos=f'{.20+jitter[0]} {jitter[1]} .68',xyaxes='1 0 0 0 1 0',fovy='48')
    side=configured_side_camera(config)
    if side is not None:
        if root.find(f".//camera[@name='{side['name']}']") is not None:raise ValueError('side camera name already exists')
        ET.SubElement(w,'camera',name=side['name'],pos=' '.join(f'{v:.12g}' for v in side['position']),
                      xyaxes=' '.join(f'{v:.12g}' for v in side['xyaxes']),fovy=f"{side['fovy']:.12g}",
                      resolution=f"{side['width']} {side['height']}")
    names=list(COLORS);rng.shuffle(names) if config['randomize_colors'] else None
    c=config['cube_size']; inner=config['container_size'];floor=config['floor_z'];rim=floor+config['rim_height']
    if task=='C':
        positions=np.asarray(config.get('c_cube_xy',[[.23,-.07],[.23,.07]]),dtype=float)
        if positions.shape!=(2,2) or not np.isfinite(positions).all():raise ValueError('c_cube_xy must contain two finite XY pairs')
        if config['randomize_positions']:positions=positions+rng.uniform(-.005,.005,(2,2))
        if any(not .19<=xy[0]<=.285 or not -.14<=xy[1]<=.14 for xy in positions):raise ValueError('C cubes outside comfortable workspace')
        if np.max(np.abs(positions[1]-positions[0]))<c+.01:raise ValueError('C cubes must have at least 1cm initial clearance')
        colors={}
        for index,xy in enumerate(positions):
            name='cube' if index==0 else 'cube_2';colors[name]=names[index]
            body=ET.SubElement(w,'body',name=name,pos=f'{xy[0]} {xy[1]} {c/2+.001}')
            ET.SubElement(body,'freejoint',name=name+'_free')
            ET.SubElement(body,'geom',name=name,type='box',size=f'{c/2} {c/2} {c/2}',mass='.035',friction='2.5 .02 .002',condim='6',solref='.006 1',rgba=' '.join(map(str,COLORS[names[index]])))
        return ET.tostring(root,encoding='unicode'),[],colors
    positions=[np.asarray(config.get('source_xy',[.23,-.09]),dtype=float),np.asarray(config.get('target_xy',[.23,.09]),dtype=float)]
    explicit_positions='source_xy' in config or 'target_xy' in config
    if explicit_positions:
        for name,xy in zip(('source_xy','target_xy'),positions):
            if xy.shape!=(2,) or not np.isfinite(xy).all():raise ValueError(f'{name} must be two finite world coordinates')
            if not .19<=xy[0]<=.285 or not -.14<=xy[1]<=.14:raise ValueError(f'{name} must lie within the comfortable XY workspace')
    if config['randomize_positions']:
        positions=[p+rng.uniform(-.008,.008,2) for p in positions]
    if explicit_positions:
        # Axis-aligned reset footprints must have at least 2mm separation.
        # A separates one cube from the destination rim; B separates two rims.
        half_extent_source=(inner/2+.008) if task=='B' else c/2
        required_separation=half_extent_source+(inner/2+.008)+.002
        if np.max(np.abs(positions[1]-positions[0]))<required_separation-1e-12:
            raise ValueError(f'{task} reset object/container footprints overlap or lack 2mm clearance')
    containers=[]
    for k,xy in enumerate(positions if task=='B' else positions[1:]):
        name='container_1' if task=='B' and k==0 else 'container_2'; color=names[(k+1)%len(names)]
        b=ET.SubElement(w,'body',name=name,pos=f'{xy[0]} {xy[1]} 0')
        ET.SubElement(b,'geom',name=name+'_floor',type='box',size=f'{inner/2+.004} {inner/2+.004} {floor/2}',pos=f'0 0 {floor/2}',rgba=' '.join(map(str,COLORS[color])))
        for j,(sx,sy,px,py) in enumerate([(.004,inner/2+.008,-inner/2-.004,0),(.004,inner/2+.008,inner/2+.004,0),(inner/2,.004,0,-inner/2-.004),(inner/2,.004,0,inner/2+.004)]):
            ET.SubElement(b,'geom',name=f'{name}_rim{j}',type='box',size=f'{sx} {sy} {rim/2}',pos=f'{px} {py} {rim/2}',rgba=' '.join(map(str,COLORS[color])))
        containers.append(dict(name=name,pos=[*xy,0.],inner_size=[inner,inner],floor_z=floor,rim_z=rim,color=color))
    xy=positions[0]; z=(floor if task=='B' else 0)+c/2+.001
    b=ET.SubElement(w,'body',name='cube',pos=f'{xy[0]} {xy[1]} {z}')
    yaw=float(config.get('cube_yaw',0.))
    if config.get('randomize_yaw',False):yaw+=rng.uniform(0,np.pi/2)
    if yaw:b.set('quat',f'{np.cos(yaw/2)} 0 0 {np.sin(yaw/2)}')
    ET.SubElement(b,'freejoint',name='cube_free')
    ET.SubElement(b,'geom',name='cube',type='box',size=f'{c/2} {c/2} {c/2}',mass='.035',friction='2.5 .02 .002',condim='6',solref='.006 1',rgba=' '.join(map(str,COLORS[names[0]])))
    return ET.tostring(root,encoding='unicode'),containers,names[0]
