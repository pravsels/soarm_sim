
#!/usr/bin/env python3

# soarm_sim.py
from __future__ import annotations
from pathlib import Path
import os

from dm_control import mjcf  # wrapper around MuJoCo's XML & physics
from dm_control.mjcf import parser as mjcf_parser 
import mujoco as _mj
import mujoco.viewer as _mjv

import tempfile
import re 

_BASE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = Path(os.environ.get("SOARM_ASSETS_DIR", _BASE_DIR / "robot_models" / "so100")).resolve()
SO100_XML = ASSETS_DIR / "so100.xml"

def load_so100_mjcf():
    """Prefer URDF: prefix bare mesh filenames with 'assets/', load, round-trip to MJCF, then parse."""
    urdf_path = ASSETS_DIR / "so100.urdf"
    if urdf_path.is_file():
        txt = urdf_path.read_text(encoding="utf-8")

        # Keep this: make bare STL names explicit (assets/...), so URDF compiles reliably.
        txt = re.sub(
            r'filename="([^/"][^"]*\.stl)"',
            r'filename="assets/\1"',
            txt,
            flags=re.IGNORECASE,
        )

        # Write the temp URDF in ASSETS_DIR (URDF expects assets/* relative to here)
        with tempfile.NamedTemporaryFile(suffix=".urdf", dir=str(ASSETS_DIR), delete=False) as tf:
            tf.write(txt.encode("utf-8"))
            tmp_urdf = tf.name

        # Compile URDF -> MjModel
        m = _mj.MjModel.from_xml_path(tmp_urdf)

        # ✅ Save the round-tripped MJCF in the *assets* directory
        assets_dir = ASSETS_DIR / "assets"
        assets_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(suffix=".xml", dir=str(assets_dir), delete=False) as tmp:
            tmp_xml = tmp.name
        _mj.mj_saveLastXML(tmp_xml, m)

        # Now dm_control will resolve <mesh file="Base.stl"> as ./assets/Base.stl
        return mjcf_parser.from_path(tmp_xml, escape_separators=True)

    if not SO100_XML.is_file():
        raise FileNotFoundError(
            f"Expected {urdf_path} or {SO100_XML}. Place your URDF in robot_models/so100/so100.urdf."
        )
    return mjcf_parser.from_path(str(SO100_XML), escape_separators=True)

# ---------- Pick & Place model augmentation (no env yet) ----------
TABLE_Z = 0.0
CUBE_EDGE = 0.03          # 3 cm cube
TARGET_XY = (0.35, 0.0)   # reachable spot in front of the arm
TARGET_RADIUS = 0.03      # 3 cm success radius

def build_pick_place_model():
    """
    Load SO100 MJCF and augment it with:
      - a plane 'table'
      - a free cube body 'cube_body' with geom 'cube'
      - a visual target site 'target_site'
    Returns an mjcf.RootElement ready for Physics().
    """
    model = load_so100_mjcf()
    world = model.worldbody

    # Table: single plane at z=TABLE_Z
    world.add(
        "geom", name="table", type="plane",
        size=[0.6, 0.6, 0.02], pos=[0.0, 0.0, TABLE_Z],
        rgba=[0.9, 0.9, 0.9, 1.0], friction=[1.0, 0.005, 0.0001]
    )

    # Cube: free body with a box geom
    cube_z = TABLE_Z + (CUBE_EDGE / 2.0)
    cube_body = world.add("body", name="cube_body", pos=[0.28, 0.0, cube_z])
    cube_body.add("freejoint", name="cube_free")
    cube_body.add(
        "geom", name="cube", type="box",
        size=[CUBE_EDGE / 2.0, CUBE_EDGE / 2.0, CUBE_EDGE / 2.0],
        rgba=[0.2, 0.4, 0.8, 1.0],
        friction=[1.0, 0.005, 0.0001],
        mass="0.05",
    )

    # Target: visual site near the table surface
    world.add(
        "site", name="target_site", type="sphere",
        size=[TARGET_RADIUS], pos=[TARGET_XY[0], TARGET_XY[1], TABLE_Z + 0.001],
        rgba=[0.1, 0.8, 0.1, 0.8], group="4",
    )

    return model


def __view_interactive__():
    model = build_pick_place_model()

    # Export XML + in-memory asset blobs (dm_control rewrites file names to hashed ones)
    xml_str, assets = mjcf.export_with_assets(model)

    # Build MuJoCo model using the asset dict (no filesystem lookups needed)
    m = _mj.MjModel.from_xml_string(xml_str, assets=assets)
    d = _mj.MjData(m)

    if m.nu > 0:
        ctrl_mid = 0.5 * (m.actuator_ctrlrange[:, 0] + m.actuator_ctrlrange[:, 1])
        d.ctrl[:] = ctrl_mid

    for _ in range(10):
        _mj.mj_step(m, d)

    with _mjv.launch_passive(m, d) as viewer:
        print("Viewer running — drag to orbit, scroll to zoom, ESC to quit.")
        while viewer.is_running():
            _mj.mj_step(m, d)
            viewer.sync()

if __name__ == "__main__":

    __view_interactive__()  

