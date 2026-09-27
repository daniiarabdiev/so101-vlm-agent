"""Task-agnostic image transforms; privileged maps are explicitly diagnostic."""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


ROBOT_FIELDS = ('joint_pos', 'ee_pos', 'gripper_opening', 'gripper_state',
                'last_action', 'step_count', 'workspace')
FEEDBACK_FIELDS = ('actual_displacement', 'contact', 'stalled', 'gripper_state',
                   'requested_displacement', 'clamped', 'ik_error')


def robot_state(observation):
    """Allowlist, never a blacklist: simulator object state cannot slip through."""
    state = {k: observation[k] for k in ROBOT_FIELDS if k in observation}
    feedback = observation.get('last_feedback') or {}
    state['last_feedback'] = {k: feedback[k] for k in FEEDBACK_FIELDS if k in feedback}
    return state


def project_point(sim, point, camera, width, height):
    import mujoco
    camera = {'wrist': 'wrist_cam'}.get(camera, camera)
    cid = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
    if cid < 0:
        raise ValueError(f'Camera {camera} not found')
    rotation = sim.data.cam_xmat[cid].reshape(3, 3)
    local = rotation.T @ (np.asarray(point) - sim.data.cam_xpos[cid])
    if local[2] >= -1e-5:
        return None
    focal = height / (2 * math.tan(math.radians(sim.model.cam_fovy[cid]) / 2))
    return (float(width/2 + focal*local[0]/-local[2]),
            float(height/2 - focal*local[1]/-local[2]))


def draw_arrow(draw, start, end, label, color):
    if start is None or end is None:
        return
    draw.line([start, end], fill=color, width=3)
    dx, dy = end[0] - start[0], end[1] - start[1]
    norm = max(math.hypot(dx, dy), 1)
    ux, uy = dx/norm, dy/norm
    draw.polygon([end, (end[0]-8*ux+4*uy, end[1]-8*uy-4*ux),
                  (end[0]-8*ux-4*uy, end[1]-8*uy+4*ux)], fill=color)
    x, y = end
    draw.rectangle([x-8, y-8, x+8, y+8], fill='white', outline=color)
    draw.text((x-3, y-6), str(label), fill='black')


def draw_world_grid(sim, image, observation):
    """Known camera calibration + robot XY only. Never consumes object state."""
    draw = ImageDraw.Draw(image)
    def point(x,y):
        return project_point(sim,[float(x),float(y),0.0],'overhead',*image.size)
    for x in np.arange(.10,.361,.02):
        a,b=point(x,-.20),point(x,.20)
        if a and b:
            draw.line([a,b],fill='#a4acb6',width=1)
            if round((x-.10)/.02)%3 == 0:
                draw.text((b[0]+2,b[1]+2),f'x{x:.2f}',fill='#243953',stroke_width=1,stroke_fill='white')
    for y in np.arange(-.20,.201,.02):
        a,b=point(.10,y),point(.36,y)
        if a and b:
            draw.line([a,b],fill='#a4acb6',width=1)
            draw.text((b[0]+2,b[1]-5),f'y{y:+.2f}',fill='#243953',stroke_width=1,stroke_fill='white')
    x,y=observation['ee_pos'][:2]
    xy=point(x,y)
    if xy:
        u,v=xy
        draw.line([(u-10,v),(u+10,v)],fill='red',width=3)
        draw.line([(u,v-10),(u,v+10)],fill='red',width=3)
        draw.text((u+12,v+4),'TCP XY',fill='red',stroke_width=1,stroke_fill='white')


def candidate_images(sim, images, observation, options):
    from so101_vlm.actions import action_delta
    result = []
    for camera, array in images.items():
        image = Image.fromarray(array).convert('RGB')
        draw = ImageDraw.Draw(image)
        ee = np.asarray(observation['ee_pos'])
        origin = project_point(sim, ee, camera, *image.size)
        for index, (label, action) in enumerate(options.items()):
            delta = action_delta(action)
            if delta is None:
                continue
            target = project_point(sim, ee + delta, camera, *image.size)
            draw_arrow(draw, origin, target, label, ('#e32b2b', '#1d56d1', '#159348')[index % 3])
        result.append(image)
    return result


def parse_grounding(text):
    """Accept only image-space points; malformed perception stays empty."""
    candidates = re.findall(r'\{[\s\S]*\}', text)
    if not candidates:
        return []
    try:
        parsed = json.loads(candidates[-1])
    except (ValueError, TypeError):
        return []
    objects = []
    for obj in parsed.get('objects', [])[:20]:
        try:
            x, y = float(obj['x']), float(obj['y'])
            if math.isfinite(x+y) and 0 <= x <= 1 and 0 <= y <= 1:
                objects.append({'name': str(obj.get('name', 'object'))[:60], 'x': x, 'y': y})
        except (KeyError, TypeError, ValueError):
            continue
    return objects


def ground_objects(backend, overhead):
    prompt = ('Locate visible loose objects and containers in this image. Report each '
              'visible object with a short color/shape name and its center in image '
              'coordinates normalized 0..1: x from left to right, y from top to bottom. '
              'Do not plan an action. Return JSON only: '
              '{"objects":[{"name":"red cube","x":0.5,"y":0.5}]}.')
    if hasattr(backend, 'generate_text'):
        response = backend.generate_text(prompt, [overhead], mode='no_thinking', max_tokens=384)
    else:
        response = backend.decide(prompt, [overhead], ['A', 'B'], max_tokens=384, constrain_answer=False)
    return parse_grounding(response['output_text']), {'prompt': prompt, **response}


def object_map(sim, observation, objects, size, *, diagnostic=False):
    width, height = size
    image = Image.new('RGB', size, '#f4f3ee')
    draw = ImageDraw.Draw(image)
    for x in range(0, width, max(1, width//10)):
        draw.line([(x, 0), (x, height)], fill='#dfddd3')
    for y in range(0, height, max(1, height//10)):
        draw.line([(0, y), (width, y)], fill='#dfddd3')
    for index, obj in enumerate(objects, 1):
        x, y = obj['x']*width, obj['y']*height
        stale=obj.get('stale_age',0)
        if 'stale_age' in obj:
            draw.ellipse((x-7, y-7, x+7, y+7), fill=None if stale else '#587998',outline='#bd7400' if stale else '#587998',width=2)
        else:
            draw.ellipse((x-7, y-7, x+7, y+7), fill='#587998')
        suffix=f' STALE {stale} steps' if stale else ''
        draw.text((x+9, y-5), f'{index}: {obj["name"]}'+suffix, fill='#bd7400' if stale else '#18334b')
    ee = project_point(sim, observation['ee_pos'], 'overhead', width, height)
    if ee:
        x, y = ee
        draw.line([(x-9,y),(x+9,y)], fill='red', width=3)
        draw.line([(x,y-9),(x,y+9)], fill='red', width=3)
        draw.text((x+10,y+8), 'gripper', fill='red')
    draw.text((8,8), 'ORACLE-PERCEPTION DIAGNOSTIC' if diagnostic else
              'VLM grounding map (image coordinates)', fill='black')
    return image


LEGEND = '''The SO-101 is a 5-joint arm plus gripper. Code executes one choice at a time.
World coordinates are metres: left=-x, right=+x, forward=+y, back=-y, up=+z, down=-z.
Small movements are 1 cm, large movements 4 cm (unless the experiment config says otherwise).
The gripper tip remains oriented down. Move above an object with open fingers, lower,
close, lift, carry above the destination, lower, open, then lift away. Check the images
and actual-displacement/contact feedback after each choice. A closed gripper alone
does not establish a successful grasp. Choose done only after the goal is visibly met.
If offered, grasp descends and closes at the CURRENT XY; lift rises to a fixed carry
height; release lowers and opens at the CURRENT XY. Macros do not locate objects.'''


def separate_static_reference(inputs, *, legend_image_present=False):
    """Split an explicit P1/P1c/P1g reference without changing scene content.

The caller restricts this experimental layout to no history/planner/map. In
those pipelines the optional fixed legend is the final supplied image.
Ordinary build_input remains byte-for-byte unchanged.
"""
    prompt = inputs['prompt']
    if prompt.count(LEGEND) != 1:
        raise ValueError('Static layout requires exactly one ordinary robot legend')
    result = dict(inputs)
    result['prompt'] = prompt.replace(LEGEND, '', 1)
    result['images'] = list(inputs['images'])
    result['static_prompt'] = LEGEND
    result['static_images'] = []
    if legend_image_present:
        caption = 'An additional image strip illustrates actions from a fixed pose.'
        if caption not in result['prompt'] or len(result['images']) < 2:
            raise ValueError('Static legend image is missing or its image order is ambiguous')
        result['prompt'] = result['prompt'].replace(caption, '', 1)
        result['static_images'] = [result['images'].pop()]
    result['input_layout'] = 'static-first-v1'
    return result


def build_input(sim, observation, images, actions, pipeline='P0', history=(),
                history_window=4, legend_path=None, backend=None, subtask=None, camera_views='both', grounding_memory=None, grounding_image_links=None, visual_history_images=None, brief_reasoning=False,
                side_camera_view=False, side_height_question=False):
    if not isinstance(side_camera_view,bool) or not isinstance(side_height_question,bool):
        raise ValueError('side camera options must be booleans')
    if side_height_question and not side_camera_view:
        raise ValueError('side_height_question requires side_camera_view')
    labels = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')[:len(actions)]
    if len(actions) > len(labels):
        raise ValueError('Single-token letter vocabulary exceeds 26 actions')
    options = dict(zip(labels, actions))
    image_list = [Image.fromarray(images[k]).convert('RGB') for k in ('overhead','wrist')]
    image_names=['current overhead','current wrist']
    auxiliary = []
    details = []
    if pipeline not in ('P0', 'P1', 'P1c', 'P1g', 'P2', 'P2m', 'P3', 'P4', 'P2_oracle'):
        raise ValueError(f'Unknown pipeline {pipeline}')
    joint_vocabulary=any(action.startswith('joint_') for action in actions)
    if joint_vocabulary:
        joint_names=[sim.model.joint(i).name for i in range(6)]
        joint_limits=[[max(float(sim.model.jnt_range[i,0]),float(sim.model.actuator_ctrlrange[i,0])),min(float(sim.model.jnt_range[i,1]),float(sim.model.actuator_ctrlrange[i,1]))] for i in range(6)]
        details.append('JOINT-SPACE ABLATION: each joint_N_plus/minus increases/decreases only actuator N by '+str(sim.config.get('joint_step',.1))+' radians, interpolated and limit-clamped. Joint 5 is the gripper: plus opens, minus closes. There are no Cartesian moves or grasp/lift/release macros in this vocabulary. The end-effector orientation can change during joint steps. Joint names: '+json.dumps(joint_names)+'. Current per-joint limits (radians): '+json.dumps(joint_limits)+'. Use the actual readings and images after every choice. Done terminates the episode.')
    if pipeline != 'P0':
        if not joint_vocabulary:details.append(LEGEND)
        if legend_path and Path(legend_path).exists():
            image_list.append(Image.open(legend_path).convert('RGB'))
            image_names.append('fixed-pose action legend strip')
            details.append('An additional image strip illustrates actions from a fixed pose.')
    if pipeline == 'P3' and joint_vocabulary:
        raise ValueError('P3 Cartesian candidate arrows are unavailable for joint vocabulary; use P0/P1 or a grounding map')
    if pipeline == 'P3':
        image_list[:2] = candidate_images(sim, images, observation, options)
        details.append('Arrows project each labeled candidate move from the current gripper tip. '
                       'Overlapping or invisible arrows may occur; use the label legend too.')
    if pipeline == 'P1c':
        details.append('In the overhead image, forward (+y) is UP, back (-y) is DOWN, '
                       'right (+x) is RIGHT, left (-x) is LEFT. The red cross marks the '
                       'gripper tool reference from joint kinematics, not an object detector.')
        for camera, image in zip(('overhead', 'wrist'), image_list[:2]):
            xy = project_point(sim, observation['ee_pos'], camera, *image.size)
            if xy:
                x,y = xy
                draw = ImageDraw.Draw(image)
                draw.line([(x-12,y),(x+12,y)], fill='#ff2222', width=3)
                draw.line([(x,y-12),(x,y+12)], fill='#ff2222', width=3)
                draw.text((x+12,y+5), 'TCP', fill='#ff2222', stroke_width=1, stroke_fill='white')
    if pipeline == 'P1g':
        draw_world_grid(sim,image_list[0],observation)
        details.append(('Joint ablation: the 2cm table-plane grid is a position reference only. A joint step can move the TCP along several Cartesian axes and change orientation. The red cross is the joint-derived TCP projection, not object perception. ' if joint_vocabulary else 'Overhead image: up=forward(+y), down=back(-y), right=+x, left=-x. '
                       'The grid is a fixed TABLE-PLANE coordinate grid with 2cm spacing. '
                       'A small planar move is HALF a grid square; a large move is TWO squares. '
                       'The red TCP XY cross is the gripper footprint projected onto this fixed '
                       'table plane from robot joints, independent of all objects. Its height '
                       'is given separately in robot readings. Use the grid to judge planar '
                       'alignment before grasping; being high above the table does not imply '
                       'being aligned over a cube. The yellow articulated structure is the robot.'))
    if pipeline in ('P2', 'P2m', 'P4', 'P2_oracle'):
        width, height = image_list[0].size
        if pipeline == 'P2m':
            if backend is None or grounding_memory is None:raise ValueError('P2m requires backend and episode GroundingMemory')
            objects,response=grounding_memory.ground(backend,image_list[0],observation['step_count'],grounding_image_links)
            auxiliary.append(response)
            details.append('Grounding memory is model perception only. Orange hollow marks are STALE last-seen points, not current object positions. Occluded objects may have moved; consult raw current images and stale age.')
        elif pipeline == 'P2_oracle':
            objects = []
            for obj in observation.get('objects', []) + observation.get('containers', []):
                xy = project_point(sim, obj['pos'], 'overhead', width, height)
                if xy:
                    objects.append({'name': obj['name'], 'x': xy[0]/width, 'y': xy[1]/height})
            auxiliary.append({'source': 'simulator_ground_truth', 'diagnostic': True})
        else:
            if backend is None:
                raise ValueError('P2/P4 require an explicit task-agnostic VLM grounder')
            objects, response = ground_objects(backend, image_list[0])
            auxiliary.append({'source': 'vlm_grounding', **response})
        if pipeline.startswith('P2'):
            image_list.append(object_map(sim, observation, objects, (width,height),
                                         diagnostic=pipeline == 'P2_oracle'))
            image_names.append('model grounding memory map with stale ages' if pipeline=='P2m' else 'object map')
        else:
            draw = ImageDraw.Draw(image_list[0])
            for index, obj in enumerate(objects, 1):
                x,y = obj['x']*width, obj['y']*height
                draw.ellipse((x-10,y-10,x+10,y+10), fill='white', outline='black', width=2)
                draw.text((x-3,y-6), str(index), fill='black')
        details.append('Object marks from ' + ('PRIVILEGED ORACLE PERCEPTION (diagnostic only)' if
                        pipeline == 'P2_oracle' else 'the VLM grounding pass') + ': ' + json.dumps(objects))
    if camera_views not in ('both', 'overhead'):
        raise ValueError('camera_views must be both or overhead')
    if camera_views == 'overhead':
        del image_list[1]
        del image_names[1]
    height_question=None
    if side_camera_view:
        from .scene import configured_side_camera
        side=configured_side_camera(sim.config)
        if side is None or side['name'] not in images:
            raise ValueError('side_camera_view requires configured rendered side-camera pixels')
        image_list.append(Image.fromarray(images[side['name']]).convert('RGB'))
        image_names.append('fixed task-independent side RGB view')
        if side_height_question:
            if not observation.get('containers'):
                raise ValueError('side height question requires a destination container')
            from .side_input import side_height_question as build_side_height_question
            height_question=build_side_height_question(side['name'])
    if visual_history_images:
        for name,array in visual_history_images.items():
            image_list.append(Image.fromarray(np.asarray(array)).convert('RGB'))
            image_names.append(name+' RAW reference (not current)')
        details.append('Historical raw overhead frames are references only; objects may have moved. No object tracking or action inference was applied to these frames.')
    if pipeline=='P2m' or visual_history_images:
        details.append('Image order: '+'; '.join(f'{i+1}: {name}' for i,name in enumerate(image_names)))
    state = robot_state(observation)
    camera_text = ('First image is overhead, second is wrist.' if camera_views == 'both' else
                   'The camera image is overhead. No wrist image is included in this ablation.')
    if side_camera_view:
        camera_text += ' The final image is a fixed calibrated side-camera view; image x increases right and image y increases down.'
    prompt = ('You control the pictured robot. ' + camera_text + '\n'
              f'Goal: {observation.get("goal", "Put the cube inside the container.")}\n'
              + '\n'.join(details) + '\nRobot readings and adapter feedback: ' +
              json.dumps(state, separators=(',', ':')) + '\nRecent choices: ' +
              json.dumps(list(history)[-history_window:] if history_window else []) + '\n')
    if subtask:
        prompt += 'Current planner subtask: ' + str(subtask) + '\n'
    prompt += ('Choose exactly ONE next action. Options:\n' +
               '\n'.join(f'{label}: {action}' for label, action in options.items()) +
               '\nAfter any internal reasoning, output exactly ANSWER: followed by ONE '
               'option letter. Do not repeat the options or explain the answer.\nANSWER:')
    if brief_reasoning:
        prompt='Reason briefly about only the next action. Do not restate the scene, joint values, or option list. Limit reasoning to at most three short sentences, then select the next choice.\n'+prompt
    result={'prompt': prompt, 'images': image_list, 'options': options,
            'auxiliary': auxiliary, 'pipeline': pipeline, 'robot_state': state}
    if height_question is not None:result['height_question']=height_question
    return result
