"""Sparse furniture layout inspired by /data/Dexmate/wm_data/episode_7.hdf5.

Positions approximate the requested arrangement, not a metric reconstruction.
The robot faces +X; its left is +Y. Only the room shell comes from Rs_int.
"""
from .pick_place_task import PickPlaceTask


class Episode7PickPlaceTask(PickPlaceTask):
    LAYOUT_NAME = "episode7_room_v1"
    ROBOT_POS = (-0.60, -1.40, 0.03)
    APPLE_XY = (0.95, -2.35)
    BOWL_XY = (0.95, -2.60)
    FURNITURE = (
        # name, category, model, bounding box, XY center
        ("front_bookshelf", "bookcase", "njwsoa", (.38, 1.0, 1.70), (1.40, -1.25)),
        ("right_table", "breakfast_table", "skczfi", (1.05, 1.10, .76), (1.25, -2.65)),
        ("left_table", "breakfast_table", "skczfi", (1.30, .70, .76), (-.30, .25)),
    )

    def object_configs(self):
        furniture = [dict(type="DatasetObject", name=name, category=category, model=model,
                          bounding_box=list(size), position=[*xy, size[2] / 2 + .02],
                          orientation=([0., 0., 1., 0.] if name == "front_bookshelf" else [0., 0., 0., 1.]),
                          fixed_base=True)
                     for name, category, model, size, xy in self.FURNITURE]
        # Build an open metal stand instead of stretching a wall shelf into a rack.
        for x in (-.90, -.40):
            for y in (-3.98, -3.12):
                furniture.append(dict(type="PrimitiveObject", primitive_type="Cube",
                    name=f"right_rack_post_{len(furniture)}", category="stand", fixed_base=True,
                    scale=[.035, .035, 1.80], position=[x, y, .915],
                    rgba=[.25, .28, .30, 1.]))
        for i, z in enumerate((.12, .65, 1.18, 1.75)):
            furniture.append(dict(type="PrimitiveObject", primitive_type="Cube",
                name=f"right_rack_shelf_{i}", category="shelf", fixed_base=True,
                scale=[.55, .95, .035], position=[-.65, -3.55, z],
                rgba=[.45, .48, .50, 1.]))
        return furniture + super().object_configs()

    def bind(self, env):
        registry = env.env.scene.object_registry
        self.table = registry("name", "right_table")
        self.block_chair = None
        self.apple = registry("name", "apple")
        self.bowl = registry("name", "bowl")

    def reset_furniture(self, env):
        # Dataset origins need not coincide with bbox centers. Place measured bounds
        # on the floor and at the requested XY centers after each environment reset.
        registry = env.env.scene.object_registry
        for name, _, _, _, xy in self.FURNITURE:
            obj = registry("name", name)
            lo, hi = (v.detach().cpu().numpy() for v in obj.aabb)
            pos = env.obj_pos(obj).copy()
            pos[:2] += xy - (lo[:2] + hi[:2]) / 2
            pos[2] += .015 - lo[2]
            obj.set_position_orientation(position=pos)

    def reset(self, env):
        self.reset_furniture(env)
        super().reset(env)
