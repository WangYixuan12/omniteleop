"""Human demonstration scene approximating lh_data/episode_0.hdf5.

The pillow is a rigid proxy; this is not a cloth simulation.
Stages and success require human annotation after collection.
"""
import numpy as np

from .episode7_room import Episode7PickPlaceTask


class LHDemoRoom(Episode7PickPlaceTask):
    name = "lh_demo"
    LAYOUT_NAME = "lh_episode0_v1"
    UNANNOTATED = True
    ROBOT_POS = (-.55, -2.65, .03)

    def object_configs(self):
        objects = [o for o in super().object_configs()
                   if o['name'] not in ('apple', 'bowl')
                   and not o['name'].startswith('right_rack_')]
        for obj in objects:
            if obj['name'] == 'front_bookshelf':
                # The inherited njwsoa packs 7 shelves into 2.00 m, so at any height that
                # fits the room its openings stay ~0.25 m -- too tight to reach into, and
                # stretching it only scales the whole unit. xgvyyv is the same kind of tall
                # bookcase (1.96 m native) with 3 bays instead of 7: measured off its USD,
                # at 1.90 m the shelf surfaces land at 0.10 / 0.74 / 1.25 m with 0.60 m and
                # 0.48 m of clear space between them, and 0.74 m matches the table height.
                obj['model'] = 'xgvyyv'
                obj['bounding_box'] = [.38, .80, 1.90]
                obj['position'][2] = 1.90 / 2 + .02
        objects.extend([
            dict(type="DatasetObject", name="pillow", category="pillow", model="iyjelw",
                 bounding_box=[.34, .44, .12], position=[1.02, -2.38, 1.0]),
            dict(type="DatasetObject", name="basket", category="wicker_basket", model="unasxd",
                 bounding_box=[.34, .36, .16], position=[1.02, -2.93, 1.0]),
        ])
        return objects

    def bind(self, env):
        registry = env.env.scene.object_registry
        self.table = registry('name', 'right_table')
        self.props = {name: registry('name', name) for name in ('pillow', 'basket')}

    def reset(self, env):
        self.reset_furniture(env)
        table_z = float(self.table.aabb[1][2])
        for name, xy, bottom in (
            ('pillow', (1.02, -2.38), table_z + .01),
            ('basket', (1.02, -2.93), table_z + .01),
        ):
            obj = self.props[name]
            lo, hi = (v.detach().cpu().numpy() for v in obj.aabb)
            pos = env.obj_pos(obj).copy()
            pos[:2] += np.asarray(xy) - (lo[:2] + hi[:2]) / 2
            pos[2] += bottom - lo[2]
            obj.set_position_orientation(position=pos)

    def objects_of_interest(self, env):
        return np.stack([env.obj_pos(self.props[n]) for n in ('pillow', 'basket')])

    def success(self, env):
        raise NotImplementedError('Human long-horizon demonstrations require manual outcome annotation')

    def expert_reset(self, env):
        raise RuntimeError('lh_demo is for human teleoperation; scripted collection is not supported')
