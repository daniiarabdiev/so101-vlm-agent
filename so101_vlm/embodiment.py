"""Contact-only MuJoCo embodiment. No welds, attachments, or object relocation."""
import copy
import math
import re
import numpy as np
import mujoco
from scipy.optimize import least_squares
from .scene import DEFAULTS, build_scene, configured_side_camera

class Embodiment:
    def __init__(self, config=None):
        self.config={**DEFAULTS,**(config or {})};self.renderer=None;self.model=None
    def reset(self, seed, task='A'):
        if task not in ('A','B','C'):raise ValueError('Unknown task')
        self.close();self.seed=int(seed);self.task=task;self.step_count=0;self.last_action=None;self.last_feedback={}
        xml,self.containers,self.cube_color=build_scene(self.config,seed,task)
        self.model=mujoco.MjModel.from_xml_string(xml);self.data=mujoco.MjData(self.model);self._scratch=mujoco.MjData(self.model)
        self.site=self.model.site('gripperframe').id;self.gripper_body=self.model.body('gripper').id
        self.workspace={'min':[.19,-.14,.027],'max':[.285,.14,.09]}
        self.grip_target=1.0
        home=self.config.get('home_pos',[.23,0,self.config['carry_height']])
        if len(home)!=3 or not np.isfinite(home).all():raise ValueError('home_pos must contain three finite world coordinates')
        result=self.ik(home);self.data.qpos[:5]=result['joint_pos'];self.data.qpos[5]=self.grip_target
        self.data.ctrl[:]=self.data.qpos[:6];mujoco.mj_forward(self.model,self.data);self._advance(.5)
        return self.observe()
    def ik(self,target):
        """Return joint_pos(5), achieved_pos, error metres, success (<3mm). No mutation."""
        target=np.asarray(target,float);d=self._scratch;m=self.model
        d.qpos[:]=self.data.qpos
        def fun(q):
            d.qpos[:5]=q;mujoco.mj_kinematics(m,d)
            return np.r_[d.site_xpos[self.site]-target,.08*(d.xmat[self.gripper_body].reshape(3,3)-np.eye(3)).ravel()]
        bounds=(m.jnt_range[:5,0]+1e-5,m.jnt_range[:5,1]-1e-5)
        x=np.clip(self.data.qpos[:5],*bounds)
        r=least_squares(fun,x,bounds=bounds,max_nfev=60,ftol=1e-8,gtol=1e-8)
        if np.linalg.norm(r.fun)>.003:
            r2=least_squares(fun,[0,-.2,.5,1.3,0],bounds=bounds,max_nfev=60)
            if np.linalg.norm(r2.fun)<np.linalg.norm(r.fun):r=r2
        fun(r.x);error=float(np.linalg.norm(d.site_xpos[self.site]-target))
        return dict(joint_pos=r.x.tolist(),achieved_pos=d.site_xpos[self.site].tolist(),error=error,success=error<.003)
    def _advance(self,seconds):
        for _ in range(round(seconds/self.model.opt.timestep)):mujoco.mj_step(self.model,self.data)
        mujoco.mj_forward(self.model,self.data)
    def _move(self,target):
        target=np.clip(target,self.workspace['min'],self.workspace['max']);sol=self.ik(target)
        start=self.data.ctrl[:5].copy();end=np.array(sol['joint_pos']);n=round(self.config['action_seconds']/self.model.opt.timestep)
        for k in range(n):
            t=(k+1)/n;t=t*t*(3-2*t)
            self.data.ctrl[:5]=start+(end-start)*t;self.data.ctrl[5]=self.grip_target;mujoco.mj_step(self.model,self.data)
        self._advance(self.config['settle_seconds']);return sol
    def _contacts(self,object_name='cube'):
        fixed=[];moving=[];cube=self.model.geom(object_name).id
        for index,c in enumerate(self.data.contact[:self.data.ncon]):
            if cube not in (c.geom1,c.geom2):continue
            other=c.geom2 if c.geom1==cube else c.geom1
            body=self.model.geom_bodyid[other]
            force=np.zeros(6);mujoco.mj_contactForce(self.model,self.data,index,force)
            if force[0]<=1e-5:continue
            normal=c.frame[:3].copy()*(1 if c.geom2==cube else -1)*force[0]
            if body==self.gripper_body:fixed.append(normal)
            if body==self.model.body('moving_jaw_so101_v1').id:moving.append(normal)
        if not fixed or not moving:return False
        a=np.mean(fixed,axis=0);b=np.mean(moving,axis=0)
        return bool(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12)<-.25)
    def observe(self):
        if self.task=='C':return self._observe_c()
        d=self.data;m=self.model;hold=self._contacts();state='holding' if hold else ('open' if self.grip_target>.5 else 'closed')
        fixed=d.geom_xpos[m.geom('fixed_jaw_sph_tip1').id];moving=d.geom_xpos[m.geom('moving_jaw_sph_tip1').id]
        v=np.zeros(6);mujoco.mj_objectVelocity(m,d,mujoco.mjtObj.mjOBJ_BODY,m.body('cube').id,v,0)
        return dict(seed=self.seed,task=self.task,step_count=self.step_count,ee_pos=d.site_xpos[self.site].tolist(),joint_pos=d.qpos[:6].tolist(),
          gripper_opening=float(np.linalg.norm(fixed-moving)),gripper_state=state,last_action=self.last_action,last_feedback=copy.deepcopy(self.last_feedback),
          objects=[dict(name='cube',pos=d.xpos[m.body('cube').id].tolist(),size=self.config['cube_size'],color=self.cube_color,velocity=v[3:].tolist(),angular_velocity=v[:3].tolist(),rotation_matrix=d.xmat[m.body('cube').id].reshape(3,3).tolist())],
          containers=copy.deepcopy(self.containers),workspace=self.workspace,goal=f'Put the {self.cube_color} cube into the {self.containers[-1]["color"]} container.' + (f' It starts in the {self.containers[0]["color"]} container.' if self.task=='B' else ''),holding='cube' if hold else None,
          carry_height=self.config['carry_height'],grasp_height=self.config['grasp_height'],release_height=self.config['release_height'])
    def _observe_c(self):
        d=self.data;m=self.model;objects=[];held=[]
        for name,color in self.cube_color.items():
            body=m.body(name).id;v=np.zeros(6)
            mujoco.mj_objectVelocity(m,d,mujoco.mjtObj.mjOBJ_BODY,body,v,0)
            holding=self._contacts(name)
            if holding:held.append(name)
            objects.append(dict(name=name,pos=d.xpos[body].tolist(),size=self.config['cube_size'],color=color,velocity=v[3:].tolist(),angular_velocity=v[:3].tolist(),rotation_matrix=d.xmat[body].reshape(3,3).tolist(),held=holding))
        colors=list(self.cube_color.values())
        order=self.config.get('c_color_order',colors)
        if len(order)!=2 or set(order)!=set(colors):raise ValueError('c_color_order must list the two actual cube colors once')
        state='holding' if held else ('open' if self.grip_target>.5 else 'closed')
        fixed=d.geom_xpos[m.geom('fixed_jaw_sph_tip1').id];moving=d.geom_xpos[m.geom('moving_jaw_sph_tip1').id]
        return dict(seed=self.seed,task='C',step_count=self.step_count,ee_pos=d.site_xpos[self.site].tolist(),joint_pos=d.qpos[:6].tolist(),
            gripper_opening=float(np.linalg.norm(fixed-moving)),gripper_state=state,last_action=self.last_action,last_feedback=copy.deepcopy(self.last_feedback),
            objects=objects,containers=[],workspace=self.workspace,holding=held[0] if held else None,holding_objects=held,
            goal=f'Arrange the two cubes left to right in this colour order: {order[0]}, {order[1]}. Release both cubes.',
            color_order=list(order),table_z=0.,order_margin=float(self.config.get('order_margin',.02)),ordering_center=self.config.get('ordering_center',[.235,0.]),
            carry_height=self.config['carry_height'],grasp_height=self.config['grasp_height'],release_height=self.config['release_height'])
    def step_target_xy(self,target_xy):
        """Move to an adapter-owned absolute XY target at the current height."""
        requested=np.asarray(target_xy,dtype=float)
        if requested.shape!=(2,) or not np.isfinite(requested).all():
            raise ValueError('target_xy must contain one finite XY pair')
        before=np.asarray(self.observe()['ee_pos'],dtype=float)
        target=np.clip(requested,np.asarray(self.workspace['min'][:2]),np.asarray(self.workspace['max'][:2]))
        sol=self._move([*target,before[2]])
        self.step_count+=1;self.last_action='grid_move';obs=self.observe()
        displacement=np.asarray(obs['ee_pos'])-before
        robot_contact=any(1 <= self.model.geom_bodyid[c.geom1] <= self.model.body('moving_jaw_so101_v1').id or 1 <= self.model.geom_bodyid[c.geom2] <= self.model.body('moving_jaw_so101_v1').id for c in self.data.contact[:self.data.ncon])
        stalled=bool(not sol['success'] or np.linalg.norm(np.asarray(sol['achieved_pos'])-obs['ee_pos'])>.004)
        self.last_feedback=dict(actual_displacement=displacement.tolist(),requested_displacement=[float(target[0]-before[0]),float(target[1]-before[1]),0.],target_xy=target.tolist(),contact=robot_contact,stalled=stalled,clamped=bool(np.any(target!=requested)),gripper_state=obs['gripper_state'],holding=obs['holding'],ik_error=sol['error'])
        return copy.deepcopy(self.last_feedback)
    def step(self,action):
        from .actions import action_delta
        before=np.array(self.observe()['ee_pos']);sol=None;joint_stalled=False
        joint_match=re.fullmatch(r'joint_([0-5])_(plus|minus)',action)
        if joint_match:
            index=int(joint_match.group(1));sign=1 if joint_match.group(2)=='plus' else -1
            start=self.data.ctrl.copy();end=start.copy()
            limits=(max(self.model.actuator_ctrlrange[index,0],self.model.jnt_range[index,0]),min(self.model.actuator_ctrlrange[index,1],self.model.jnt_range[index,1]))
            end[index]=np.clip(end[index]+sign*float(self.config.get('joint_step',.10)),*limits)
            self.grip_target=float(end[5]);n=round(self.config['action_seconds']/self.model.opt.timestep)
            for k in range(n):
                t=(k+1)/n;t=t*t*(3-2*t);self.data.ctrl[:]=start+(end-start)*t;mujoco.mj_step(self.model,self.data)
            self._advance(self.config['settle_seconds'])
            joint_stalled=abs(self.data.qpos[index]-end[index])>.02
        elif action in ('open','close'):
            self.grip_target=1.0 if action=='open' else -.12;self.data.ctrl[5]=self.grip_target;self._advance(.6)
        elif action=='lift':sol=self._move([*before[:2],self.config['carry_height']])
        elif action in ('grasp','descend_close'):
            sol=self._move([*before[:2],self.config['grasp_height']]);self.grip_target=-.12;self.data.ctrl[5]=self.grip_target;self._advance(.6)
        elif action in ('release','lower_open'):
            sol=self._move([*before[:2],self.config['release_height']]);self.grip_target=1.;self.data.ctrl[5]=1.;self._advance(.8)
        elif action=='done':self._advance(.4)
        else:
            delta=action_delta(action)
            if delta is None:raise ValueError(f'Unknown action: {action}')
            sol=self._move(before+delta)
        self.step_count+=1;self.last_action=action;obs=self.observe();displacement=np.array(obs['ee_pos'])-before
        robot_contact=any(1 <= self.model.geom_bodyid[c.geom1] <= self.model.body('moving_jaw_so101_v1').id or 1 <= self.model.geom_bodyid[c.geom2] <= self.model.body('moving_jaw_so101_v1').id for c in self.data.contact[:self.data.ncon])
        stalled=bool(joint_stalled or (sol and (not sol['success'] or np.linalg.norm(np.array(sol['achieved_pos'])-obs['ee_pos'])>.004)) or (action in ('close','grasp','descend_close') and abs(self.data.qpos[5]-self.grip_target)>.05))
        self.last_feedback=dict(actual_displacement=displacement.tolist(),contact=robot_contact,stalled=stalled,gripper_state=obs['gripper_state'],holding=obs['holding'],ik_error=sol['error'] if sol else None)
        return copy.deepcopy(self.last_feedback)
    def render(self):
        if self.renderer is None:
            self.renderer=mujoco.Renderer(self.model,height=self.config['height'],width=self.config['width'])
        # CGL can change the first frame after idle by one channel level.
        # Warm every capture, including initialization, before returning pixels.
        self.renderer.update_scene(self.data,camera='overhead');self.renderer.render()
        out={}
        cameras=[('overhead','overhead'),('wrist','wrist_cam')]
        side=configured_side_camera(self.config)
        if side is not None:cameras.append((side['name'],side['name']))
        for key,camera in cameras:
            self.renderer.update_scene(self.data,camera=camera);out[key]=self.renderer.render().copy()
        return out
    def camera_calibration(self,name):
        """Return explicit pinhole calibration for a configured rendered camera."""
        camera={'wrist':'wrist_cam'}.get(name,name)
        cid=mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_CAMERA,camera)
        if cid<0:raise ValueError(f'Camera {name} not found')
        side=configured_side_camera(self.config)
        if side is not None and camera==side['name']:width,height=side['width'],side['height']
        else:width,height=int(self.config['width']),int(self.config['height'])
        fovy=float(self.model.cam_fovy[cid]);focal=height/(2*math.tan(math.radians(fovy)/2))
        return dict(name=name,image_size=[width,height],fovy_degrees=fovy,
                    focal_length_pixels=[float(focal),float(focal)],
                    principal_point_pixels=[width/2,height/2],
                    camera_to_world_rotation=self.data.cam_xmat[cid].reshape(3,3).tolist(),
                    center_world_m=self.data.cam_xpos[cid].tolist(),
                    pixel_axes='u right; v down; camera looks along local negative Z')
    def get_state(self):
        spec=mujoco.mjtState.mjSTATE_INTEGRATION;state=np.empty(mujoco.mj_stateSize(self.model,spec));mujoco.mj_getState(self.model,self.data,state,spec)
        return dict(config=self.config,seed=self.seed,task=self.task,integration=state.tolist(),step_count=self.step_count,last_action=self.last_action,last_feedback=self.last_feedback,grip_target=self.grip_target)
    def set_state(self,snapshot):
        if self.model is None or self.seed!=snapshot['seed'] or self.task!=snapshot['task'] or self.config!=snapshot['config']:
            self.config=copy.deepcopy(snapshot['config']);self.reset(snapshot['seed'],snapshot['task'])
        mujoco.mj_setState(self.model,self.data,np.array(snapshot['integration']),mujoco.mjtState.mjSTATE_INTEGRATION);mujoco.mj_forward(self.model,self.data)
        for key in ('step_count','last_action','last_feedback','grip_target'):setattr(self,key,copy.deepcopy(snapshot[key]))
    def close(self):
        if self.renderer is not None:self.renderer.close();self.renderer=None
