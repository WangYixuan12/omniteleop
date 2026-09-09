"""Live checks used by teleop_joystick.py --smoke-test; creates no demonstration."""
from pathlib import Path
import json
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image
import yaml

from robot_model import follower_urdf


def verify_session(session, output):
    from joystick_service import GROUPS
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    env = session.env
    nominal_q = yaml.safe_load(session.args.config.read_text())['nominal_posture']
    nominal = {g: [float(nominal_q.get(n, 0)) for n in names] for g, names in GROUPS.items()}
    urdf = follower_urdf()
    session.handle(dict(cmd='initialize', nominal=nominal, urdf=str(urdf)))
    report = dict(dof_names=env.names, urdf=str(urdf), grippers={}, cameras={})
    for value, label in [(0., 'open'), (1., 'closed'), (0., 'reopened')]:
        for _ in range(round(1.5 / env.dt)):
            session.step(nominal, [0, 0, 0], [value, value])
        actual = {side: float(env.finger_position_normalized(side)) for side in ('left', 'right')}
        report['grippers'][label] = actual
        print(f'[verify] grippers {label}: {actual}', flush=True)
    q = env.robot.get_joint_positions().detach().cpu().numpy()
    joint_error = max(abs(float(q[env.name2idx[n]]) - v) for n, v in nominal_q.items() if n in env.name2idx)
    report['max_nominal_joint_error_rad'] = joint_error
    report['joints'] = {n: dict(measured=float(q[env.name2idx[n]]), target=float(v))
                        for n, v in nominal_q.items() if n in env.name2idx}
    report['drive_targets'] = env.robot.get_joint_position_targets().detach().cpu().tolist()
    from omnigibson.utils.usd_utils import RigidContactAPI
    contacts = RigidContactAPI.get_contact_pairs(env.env.scene.idx, set(env.robot.links.values()),
                                                set(env.robot.links.values()), current_only=True)
    report['self_contacts'] = sorted({tuple(sorted(p.rsplit('/', 1)[-1] for p in pair)) for pair in contacts})
    for side in session.recorder.camera_arms:
        rgb = session.recorder._wrist_rgb(side)
        Image.fromarray(rgb).save(output / f'{side}_wrist.png')
        report['cameras'][side] = dict(shape=list(rgb.shape), std=float(rgb.std()))
    rgb, depth = session.recorder._head_images()
    Image.fromarray(rgb).save(output / 'head.png')
    head_info = dict(shape=list(rgb.shape), std=float(rgb.std()),
                     head_depth=bool(session.recorder.head_depth))
    if depth is not None:
        Image.fromarray(depth).save(output / 'head_depth.png')
        head_info['nonzero_depth_fraction'] = float(np.mean(depth > 0))
    report['cameras']['head'] = head_info
    report['intrinsic'] = session.recorder.intrinsic().tolist()
    # Evaluate FK against the source URDF at the measured pose, independently of Isaac.
    from scipy.spatial.transform import Rotation
    from robot_model import origin_matrix
    transforms = {'base': np.eye(4)}
    remaining = list(ET.parse(urdf).getroot().findall('joint'))
    while remaining:
        progress = False
        for joint in remaining[:]:
            parent = joint.find('parent').get('link')
            if parent not in transforms:
                continue
            local = origin_matrix(joint)
            if joint.get('type') != 'fixed':
                n = joint.get('name')
                angle = float(q[env.name2idx[n]]) if n in env.name2idx else 0.
                axis = np.fromstring(joint.find('axis').get('xyz'), sep=' ')
                motion = np.eye(4)
                if joint.get('type') == 'prismatic':
                    motion[:3, 3] = axis * angle
                else:
                    motion[:3, :3] = Rotation.from_rotvec(axis * angle).as_matrix()
                local = local @ motion
            transforms[joint.find('child').get('link')] = transforms[parent] @ local
            remaining.remove(joint)
            progress = True
        if not progress:
            raise AssertionError('Follower URDF is not a connected tree rooted at base')
    base_inv = np.linalg.inv(env.link_pose('base'))
    report['fk_errors'] = {}
    for link in ('L_ee', 'R_ee', 'zed_depth_frame', 'L_wrist_camera', 'R_wrist_camera'):
        actual = base_inv @ env.link_pose(link)
        delta = np.linalg.inv(transforms[link]) @ actual
        report['fk_errors'][link] = dict(position_m=float(np.linalg.norm(delta[:3, 3])),
                                          rotation_rad=float(Rotation.from_matrix(delta[:3, :3]).magnitude()))
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(f'[verify] report and rendered images: {output}', flush=True)
    for side in ('left', 'right'):
        assert report['grippers']['open'][side] < .15, report['grippers']
        assert report['grippers']['closed'][side] > .85, report['grippers']
        assert report['grippers']['reopened'][side] < .15, report['grippers']
    assert joint_error < .08, report
    for camera in report['cameras'].values():
        assert camera['std'] > 2., report['cameras']
    for error in report['fk_errors'].values():
        assert error['position_m'] < .005 and error['rotation_rad'] < .02, report['fk_errors']
    print('[verify] PASS: joint tracking, both grippers, enabled cameras, and follower FK', flush=True)
    import time
    action = env._hold_action()
    timings = {}
    for label, step in [('original_env_step', lambda: env.env.step(action)),
                        ('teleop_step', lambda: session.step(nominal, [0, 0, 0], [0, 0]))]:
        start = time.perf_counter()
        for _ in range(100):
            step()
        timings[label] = (time.perf_counter() - start) * 10
    report['step_ms'] = timings
    from omnigibson.sensors.vision_sensor import VisionSensor
    report['camera_mode'] = session.args.camera_mode
    report['sensor_paths'] = list(VisionSensor.SENSORS)
    if env.head_only:
        assert len(VisionSensor.SENSORS) == 1, report['sensor_paths']
        viewport = session.recorder.head_cam._viewport
        if viewport is not None:
            report['head_viewport_visible'] = bool(viewport.visible)
            assert report['head_viewport_visible']
    capture = dict(cmd='capture', grippers=[0., 0.],
                   effective_targets={k: np.eye(4) for k in ('left', 'right', 'head')},
                   sent_joints={n: v for g, names in GROUPS.items() for n, v in zip(names, nominal[g])},
                   sent_base_twist=[0., 0., 0.], base_pose=[0., 0., 0.])
    start = time.perf_counter()
    for _ in range(20):
        frame = session.handle(capture)['frame']
    report['capture_ms'] = (time.perf_counter() - start) * 50
    if env.head_only:
        assert set(frame['obs']['images']) == {'head_left_rgb'}
        assert 'depth_linear' not in session.recorder.head_cam.modalities
    print(f'[verify] capture: {report["capture_ms"]:.1f} ms; sensors: {report["sensor_paths"]}', flush=True)
    report['scene_objects'] = len(env.env.scene.objects)
    if getattr(session.task, 'UNANNOTATED', False):
        report['props'] = {name: dict(bounds=[v.detach().cpu().tolist() for v in obj.aabb])
                           for name, obj in session.task.props.items()}
        for name in ('pillow', 'basket', 'towel'):
            assert report['props'][name]['bounds'][0][2] > .65, report['props']
        result = session.handle(dict(cmd='episode_result'))
        assert not result['success_evaluated'], result
    if getattr(session.task, 'LAYOUT_NAME', None) == 'episode7_room_v1' and not env.head_only:
        assert not any('chair' in obj.category for obj in env.env.scene.objects)
        report['furniture'] = {obj.name: dict(category=obj.category,
            bounds=[v.detach().cpu().tolist() for v in obj.aabb])
            for obj in env.env.scene.objects
            if obj.name in ('front_bookshelf', 'right_table', 'left_table') or obj.name.startswith('right_rack_')}
        # Look down inside the room, below its ceiling, for a layout inspection image.
        camera = env.og.sim.viewer_camera
        camera.focal_length = 8.0
        camera.set_position_orientation(position=[0., -1.50, 2.35], orientation=[0., 0., 0., 1.])
        for _ in range(4):
            env.og.sim.render()
        rgb = camera.get_obs()[0]['rgb'].detach().cpu().numpy()[..., :3]
        Image.fromarray(rgb).save(output / 'room_overview.png')
    print(f'[verify] step milliseconds: {timings}; objects: {report["scene_objects"]}', flush=True)
    env.og.sim.stop()
    try:
        session.snapshot()
    except RuntimeError as exc:
        assert 'timeline' in str(exc), str(exc)
        report['stopped_timeline_guard'] = 'passed'
    else:
        raise AssertionError('Stopped timeline was not rejected')
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
