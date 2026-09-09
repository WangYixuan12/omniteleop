"""Human demonstration scene approximating lh_data/episode_0.hdf5.

Pillow and folded towel are rigid proxies; this is not a cloth simulation.
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
        objects = [o for o in super().object_configs() if o['name'] not in ('apple', 'bowl')]
        for obj in objects:
            if obj['name'].startswith('right_rack_post_'):
                obj['scale'][2] = 1.10
                obj['position'][2] = .565
            elif obj['name'].startswith('right_rack_shelf_'):
                obj['position'][2] = (.12, .43, .75, 1.07)[int(obj['name'][-1])]
        objects.extend([
            dict(type="DatasetObject", name="pillow", category="pillow", model="iyjelw",
                 bounding_box=[.34, .44, .12], position=[1.02, -2.38, 1.0]),
            dict(type="DatasetObject", name="basket", category="wicker_basket", model="unasxd",
                 bounding_box=[.34, .36, .16], position=[1.02, -2.93, 1.0]),
            dict(type="DatasetObject", name="towel", category="dishtowel", model="ltydgg",
                 bounding_box=[.30, .42, .035], position=[-.65, -3.55, 1.15]),
            dict(type="DatasetObject", name="table_chair", category="straight_chair", model="amgwaw",
                 bounding_box=[.46, .46, .82], position=[.35, -2.65, .45]),
        ])
        return objects

    def bind(self, env):
        registry = env.env.scene.object_registry
        self.table = registry('name', 'right_table')
        self.props = {name: registry('name', name) for name in ('pillow', 'basket', 'towel', 'table_chair')}

    def reset(self, env):
        self.reset_furniture(env)
        registry = env.env.scene.object_registry
        table_z = float(self.table.aabb[1][2])
        stand_z = float(registry('name', 'right_rack_shelf_3').aabb[1][2])
        for name, xy, bottom in (
            ('pillow', (1.02, -2.38), table_z + .01),
            ('basket', (1.02, -2.93), table_z + .01),
            ('towel', (-.65, -3.55), stand_z + .01),
            ('table_chair', (.35, -2.65), .015),
        ):
            obj = self.props[name]
            lo, hi = (v.detach().cpu().numpy() for v in obj.aabb)
            pos = env.obj_pos(obj).copy()
            pos[:2] += np.asarray(xy) - (lo[:2] + hi[:2]) / 2
            pos[2] += bottom - lo[2]
            obj.set_position_orientation(position=pos)

    def objects_of_interest(self, env):
        return np.stack([env.obj_pos(self.props[n]) for n in ('pillow', 'basket', 'towel', 'table_chair')])

    def success(self, env):
        raise NotImplementedError('Human long-horizon demonstrations require manual outcome annotation')

    def expert_reset(self, env):
        raise RuntimeError('lh_demo is for human teleoperation; scripted collection is not supported')
