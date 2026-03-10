"""
Every wrapper has a sync_node() method. 
The workflow is always:
XML → parse into wrapper objects → do Python math/structural changes → sync_node() → 
write XML → MuJoCo loads it

Why do we need a wrapper instead of just editing the XML directly?

1. To make it more manageable and to encapsulate the logic for how 
changes to the robot's structure should be reflected in the XML.
2. The wrapper can maintain an internal representation of the robot's 
structure and provide methods to modify it.

The wrappers are for these things:
- Body: represents a body in the robot, which can have child bodies and geoms. (Like bones)
- Geom: represents a geometric shape attached to a body, which can be used for collision and visualization. (Like the physical shape of a limb)
- Joint: represents a joint connecting two bodies, which defines how they can move relative to each other. (Like the hinge of a knee or the ball-and-socket of a shoulder)
- Actuator: represents a motor that can apply forces to a joint, allowing the robot to move. (Like the muscles that control the joints)

Finally the RobotWrapper class will contain the whole robot structure and provide methods to modify it, such as adding limbs, changing joint types, etc. 
It will also have a sync_node() method that updates the XML tree based on the current state of the wrapper objects.
"""

import numpy as np
from copy import deepcopy
from io import BytesIO
from lxml.etree import XMLParser, parse
from lxml import etree
from utils.xml_utils import *


class Joint:
    """Body
        ├── joints: [Joint, ...]
        │     └── actuator: Actuator   ← Joint owns its motor
        └── geoms:  [Geom, ...]
        
    Member variables:
    - node
    - body
    - local_coord
    - name
    - type
    - range
    - pos
    - actuator
    
    Methods:
    - __init__(node, body)
    - __repr__()
    - sync_node(new_pos)
    """
    def __init__(self, node, body):
        self.node=node        # the raw lxml <joint> XML element — needed so sync_node() can write back
        self.body=body        # parent Body object — joints don't exist independently, they belong to a body
        self.local_coord = body.local_coord # bool: are positions local or world-space? (Hopper vs Ant)
        self.name = node.attrib['name']      # e.g. "body_1_joint" — used to link the actuator to this joint in XML
        
        self.type = node.attrib['type']        # "hinge" or "free" — determines what other attributes are valid      
        if self.type == 'hinge': # numpy array [min, max] in radians — the joint's movement limits
            # Convert joint range from degrees (MuJoCo convention) to radians for internal use.
            self.range = np.deg2rad(parse_vec(node.attrib.get('range', "-360 360")))
        
        self.pos = parse_vec(node.attrib['pos'])         # numpy array [x, y, z] — position of the joint in the body
        if self.local_coord:
            # Hopper uses local coordinates, so joint position must be offset by body position
            # to get world-space coordinates for internal geometry calculations.
            self.pos += body.pos
            
        # Look up the motor actuator that drives this joint in the <actuator> section.
        actu_node = body.tree.getroot().find("actuator").find(f'motor[@joint="{self.name}"]')
        if actu_node is not None:
            self.actuator = Actuator(actu_node, self)
        else:
            self.actuator = None
        

    def __repr__(self):
        return 'joint_' + self.name
    
    def sync_node(self, new_pos):
        """
        When syncing, the following things are updated:
            1. Position of the joint is updated
            2. Joint's name is updated
            3. Joint's node's name is updated
            4. The new position is written back to the XML node's 'pos' attribute (Eg. pos = [1.5, 0.0, -3.14159265, 2.100000] → '1.5 0 -3.141593 2.1')
            5. If the joint has an actuator, the actuator is also synced
        """
        if new_pos is not None:
            pos = new_pos
        else:
            pos = self.pos
        self.name = self.body.name + '_joint'
        self.node.attrib['name'] = self.name
        self.node.attrib['pos'] = ' '.join([f'{x:.6f}'.rstrip('0').rstrip('.') for x in pos])
        if self.actuator is not None:
            self.actuator.sync_node()
            
class Actuator:
    """Actuator
    Member variables:
    - node
    - joint
    - joint_name
    - name
    - gear
    
    Methods:
    - __init__(node, joint)
    - __repr__()
    - sync_node()
    """
    def __init__(self, node, joint):
        self.node = node  # the raw lxml <motor> XML element — needed so sync_node() can write back
        self.joint = joint  # parent Joint object — actuators are linked to joints in the XML by the 'joint' attribute
        self.joint_name = node.attrib['joint']
        self.name = self.joint.name  
        
        self.gear = float(node.attrib['gear']) # gear is the main tunable parameter: higher gear = more torque on this joint.
    
    def __repr__(self):
        return 'actuator_for_' + self.joint.name
    
    def sync_node(self):
        """
        When syncing, the following things are updated:
            1. The actuator's gear is updates based on gear
            2. The new joint name is written back to the XML node's 'joint' attribute
        """
        self.node.attrib['gear'] = f'{self.gear:.6f}'.rstrip('0').rstrip('.')
        self.name = self.joint.name
        self.node.attrib['name'] = self.name
        self.node.attrib['joint'] = self.joint.name
        
class Geom:
    """
    Body
        ├── joints: [Joint, ...]
        │     └── actuator: Actuator   ← Joint owns its motor
        └── geoms:  [Geom, ...]
        
    Member variables:
    - node
    - body
    - local_coord
    - name
    - type
    - pos
    - size
    - start
    - end
    - bone_start
    - ext_start
    
    Methods:
    - __init__(node, body)
    - __repr__()
    - sync_node(new_pos)
    """

    def __init__(self, node, body):
        self.node = node
        self.body = body # the geom belongs to a body
        self.local_coord = body.local_coord
        self.name = node.attrib.get('name', '')
        self.type = node.attrib['type']

        self.size = parse_vec(node.attrib['size'])
        
        
        if self.type == 'capsule': 
            """capsule means a cylindrical limb segment, defined by its start and end points in 3D space.
            It looks like a capsule because it has hemispherical ends, and the size attribute's first value gives the radius of the capsule.
            The start and end points are defined in the XML by the 'fromto' attribute, which is a string of 6 numbers: "x_start y_start z_start x_end y_end z_end".
            For example, 'fromto="0 0 0 0 1 0"' means the capsule starts at the origin (0, 0, 0) and ends at (0, 1, 0), so it's a vertical capsule of length 1 along the y-axis.
            """
            if 'fromto' in node.attrib: 
                self.start, self.end = parse_fromto(node.attrib['fromto'])

            if self.local_coord:
                self.start = body.pos 
                self.end = body.pos
            
            # bone_start is the attachment point of this geom to its parent body.
            # It's used in rebuild() to recompute absolute positions after structural edits.
            if body.bone_start is None:
                self.bone_start = self.start.copy()
                body.bone_start = self.start.copy()
            else:
                self.bone_start = body.bone_start.copy()
            
            # ext_start captures how far the geom start extends beyond the bone_start.
            self.ext_start = np.linalg.norm(self.bone_start - self.start)

    def __repr__(self):
        return 'geom_' + self.name

    def update_start(self):
        # Recomputes geom.start so it stays at ext_start distance from bone_start
        # along the bone direction. Called during rebuild() when bone geometry changes.
        if self.type == 'capsule':
            vec = self.bone_start - self.end
            norm = np.linalg.norm(vec)
            if norm > 0:
                self.start = self.bone_start + vec * (self.ext_start / norm)

    def sync_node(self, body_start=None, body_end=None):
        self.node.attrib.pop('name', None)
        self.node.attrib['size'] = ' '.join([f'{x:.6f}'.rstrip('0').rstrip('.') for x in self.size])
        if self.type == 'capsule':
            # Convert to local coords for Hopper; Ant/Swimmer use world coords.
            start = self.start - self.body.pos if self.local_coord else self.start
            end = self.end - self.body.pos if self.local_coord else self.end
            if not self.local_coord:
                if body_start is not None:
                    # print("body start is not none, i.e. received new body_start and body_end")
                    self.node.attrib['fromto'] = ' '.join([f'{x:.6f}'.rstrip('0').rstrip('.') for x in np.concatenate([body_start, body_end])])
                else:
                    # print("body start is none, i.e. didnot receive new body_start and body_end")
                    self.node.attrib['fromto'] = ' '.join([f'{x:.6f}'.rstrip('0').rstrip('.') for x in np.concatenate([start, end])])


    
class Body:
    """
    Represents a body (bone) in the MuJoCo robot hierarchy.

    Member variables:
    - node:        raw lxml <body> XML element
    - tree:        full XML tree (from robot)
    - parent:      parent Body object (None for root)
    - child:       list of child Body objects
    - depth:       tree depth (0 for root)
    - name:        body name string
    - pos:         world-space position, numpy [x, y, z]
    - local_coord: True for Hopper-style (local frames), False for Ant/Swimmer (world)
    - bone_start:  attachment point to parent, world coords — initialised to pos
    - bone_end:    distal end of this segment, computed by init()
    - bone_offset: vector bone_start→bone_end, preserved across rebuilds
    - joints:      list of Joint objects owned by this body
    - geoms:       list of Geom objects owned by this body

    Methods:
    - __init__(node, parent_body, robot)
    - __repr__()
    - reindex()
    - init()
    - sync_geom()
    - sync_joint()
    - rebuild()
    - sync_node(body_start, body_end)
    """

    def __init__(self, node, parent_body, robot):
        self.node = node
        self.tree = robot.tree
        self.local_coord = robot.local_coord

        self.parent = parent_body
        self.child = []
        if parent_body is not None:
            parent_body.child.append(self)
            self.depth = parent_body.depth + 1
        else:
            self.depth = 0

        # Derive name from attribute; fall back to parent-based name for cloned nodes.
        self.name = node.attrib['name'] if 'name' in node.attrib else self.parent.name + f'_child{len(self.parent.child)}'
        self.pos = parse_vec(node.attrib['pos'])

        # bone_start: attachment point to parent body in world coords.
        # Initialised to this body's pos; updated by rebuild() as the skeleton changes.
        self.bone_start = self.pos.copy()
        self.bone_end = None    # distal end; computed by init()
        self.bone_offset = None # bone_start→bone_end vector; preserved across rebuilds

        # Parse joints (hinge and free) and geoms (capsule and sphere).
        self.joints = (
            [Joint(x, self) for x in node.findall('joint[@type="hinge"]')] +
            [Joint(x, self) for x in node.findall('joint[@type="free"]')]
        )
        self.geoms = (
            [Geom(x, self) for x in node.findall('geom[@type="capsule"]')] +
            [Geom(x, self) for x in node.findall('geom[@type="sphere"]')]
        )

    def __repr__(self):
        return 'body_' + self.name

    def reindex(self):
        """Assign a hierarchical numeric name based on position in the tree.
        Root → '0'. Each child is named by its 1-based index prepended to the parent's name
        (e.g. child 2 of body '1' → '21'), so MuJoCo can consistently identify bodies after edits."""
        if self.parent is None:
            self.name = '0'
        else:
            ind = self.parent.child.index(self) + 1
            pname = '' if self.parent.name == '0' else self.parent.name
            self.name = str(ind) + pname

    def init(self):
        """  The question this answers: "Where does this body segment end?"
        A body's bone_end is its distal tip — the far end of the limb segment. But how you
        find it depends on whether the body is a branch or a leaf:

                hip (body)
                /          \
            thigh (body)   thigh (body)     ← non-leaf: has children
                |
            shin (body)                   ← leaf: no children, only geoms
                |
            [capsule geom]

        Non-leaf body (has children):
        bone_ends = [c.bone_start for c in self.child]
        Each child body starts at the end of this body. So bone_end = average of where the
        children attach. This handles bodies that branch (e.g. a pelvis with two thigh
        children).

        Leaf body (no children — end of a limb):
        bone_ends = [g.end for g in self.geoms if hasattr(g, 'end')]
        The tip of the limb is wherever its capsule geoms end. hasattr(g, 'end') guards
        against non-capsule geoms (spheres don't have end).

        Then:
        self.bone_end = np.mean(np.stack(bone_ends), axis=0)  # average if multiple
        self.bone_offset = self.bone_end - self.bone_start     # direction + length of this
        segment

        bone_offset is the key thing saved here — it's the vector from attachment point to
        tip. During rebuild(), when positions shift, this offset is replayed (bone_end =
        bone_start + bone_offset) to preserve the limb's length and direction."""
        if len(self.child) > 0:
            bone_ends = [c.bone_start for c in self.child]
        else:
            # geom fromto is in local body frame; add self.pos to get world coords.
            bone_ends = [self.pos + g.end for g in self.geoms if hasattr(g, 'end')]
        if bone_ends:
            self.bone_end = np.mean(np.stack(bone_ends), axis=0)
            self.bone_offset = self.bone_end - self.bone_start  # so the difference

    def sync_geom(self):
        """Push current bone geometry into all geoms in local body frame.
        bone_start/bone_end are world-frame; MuJoCo reads fromto as local body frame,
        so we convert: local_end = bone_end - bone_start = bone_offset."""
        for geom in self.geoms:
            geom.bone_start = self.bone_start.copy()
            geom.start = np.zeros(3)            # geom starts at body origin (local [0,0,0])
            geom.end = self.bone_offset.copy()  # direction vector in local body frame

    def sync_joint(self):
        """Place joint at body origin in local frame (non-root bodies only)."""
        if self.parent is not None:
            for joint in self.joints:
                joint.pos = np.zeros(3)

    def rebuild(self):
        """Recompute body positions after structural edits."""
        if self.parent is not None:
            self.bone_start = self.parent.bone_end.copy()
            self.pos = self.bone_start.copy()
        if self.bone_offset is not None:
            self.bone_end = self.bone_start + self.bone_offset
        self.sync_geom()
        self.sync_joint()

    def sync_node(self, body_start=None, body_end=None):
        """Write current Python-side state back to the lxml XML node."""
        self.node.attrib['name'] = self.name
        for joint in self.joints:
            joint.sync_node(body_start)
        for geom in self.geoms:
            geom.sync_node(body_start=body_start, body_end=body_end)


class Robot:
    """
    Top-level wrapper for a MuJoCo robot XML.

    Owns the lxml tree and the full list of Body wrappers.
    The standard lifecycle is:

        robot = Robot('ant.xml', 'Ant-v5')        # load + init
        robot.add_child_to_body(robot.bodies[1])  # structural edit
        robot.rebuild()                           # recompute positions
        robot.write_xml('ant_new.xml')            # save

    Member variables:
    - tree:        lxml ElementTree — the live XML document
    - local_coord: True for Hopper-v5 (local frames), None/False for Ant/Swimmer (world)
    - bodies:      list of Body objects in DFS order (index 0 = root/torso)

    Methods:
    - __init__(xml, env_id, is_xml_str)
    - load_from_xml(xml, env_id, is_xml_str)
    - add_body(body_node, parent_body)       — internal, called during load
    - init_bodies(body_start, body_end)      — compute bone geometry + first sync
    - sync_node(body_start, body_end)        — reindex all bodies and write to XML
    - add_child_to_body(parent_body)         — clone + attach a new limb
    - remove_body(body)                      — prune a limb and its actuators
    - rebuild()                              — recompute all positions after edits
    - write_xml(fname)                       — serialize tree to file
    - export_xml_string()                    — serialize tree to bytes (no disk I/O)
    - get_gnn_edges()                        — (2, E) bidirectional edge index for GNN
    """

    def __init__(self, xml, env_id, is_xml_str=False):
        self.bodies = []
        self.tree = None
        self.load_from_xml(xml, env_id, is_xml_str)
        self.init_bodies()

    def load_from_xml(self, xml, env_id, is_xml_str=False):
        # remove_blank_text=True is required for pretty_print to work correctly on write.
        parser = XMLParser(remove_blank_text=True)
        # Accept either a file path (normal) or raw XML bytes (e.g. from export_xml_string).
        self.tree = parse(BytesIO(xml) if is_xml_str else xml, parser=parser)
        # Hopper uses local coordinate frames; Ant and Swimmer use world-space coordinates.
        self.local_coord = True if env_id == 'Hopper-v5' else None
        # The robot is always rooted at the first <body> inside <worldbody>.
        root = self.tree.getroot().find('worldbody').find('body')
        self.add_body(root, None)

    def add_body(self, body_node, parent_body):
        """DFS traversal: wrap each XML <body> node and recurse into children."""
        body = Body(body_node, parent_body, self)
        self.bodies.append(body)
        for child_node in body_node.findall('body'):
            self.add_body(child_node, body)

    def init_bodies(self, body_start=None, body_end=None):
        """Compute bone geometry (bone_end, bone_offset) for every body, then sync to XML.
        Called once after loading; must be called again after structural changes."""
        for body in self.bodies:
            body.init()
        self.sync_node(body_start, body_end)

    def sync_node(self, body_start=None, body_end=None):
        """Reindex all bodies (assign consistent names) and write state back to XML."""
        for body in self.bodies:
            body.reindex()
            body.sync_node(body_start, body_end)

    def add_child_to_body(self, parent_body, angle=None, tilt=None):
        """Clone an existing limb segment and attach it as a new child of parent_body.

        The root body (torso) is special — its first child is used as the clone template
        because the root itself has a different structure. For all other bodies, the body
        itself is cloned. Nested sub-bodies are stripped from the clone so we get a single
        clean segment. A matching actuator node is also duplicated in <actuator>.

        angle: optional rotation in radians around the Z-axis applied to bone_offset before
               rebuild(). Without it the new limb continues straight in the parent's direction;
               with it the limb branches off at the given angle.
        tilt:  optional rotation in radians around the X-axis applied after angle rotation.
               Negative values tilt the limb downward (toward the ground)."""
        # Choose the template to clone.
        if parent_body is self.bodies[0]:
            body2clone = parent_body.child[0]
        else:
            body2clone = parent_body
        child_node = deepcopy(body2clone.node)
        # Remove any nested bodies — we want a single flat limb segment.
        for bnode in child_node.findall('body'):
            child_node.remove(bnode)

        child_body = Body(child_node, parent_body, self)

        # Duplicate each motor actuator for the new body's joints.
        actu_section = parent_body.tree.getroot().find('actuator')
        for joint in child_body.joints:
            new_actu_node = deepcopy(actu_section.find(f'motor[@joint="{joint.name}"]'))
            actu_section.append(new_actu_node)
            joint.actuator = Actuator(new_actu_node, joint)

        # Inherit parent's bone_offset so the new limb has the same default proportions.
        child_body.bone_offset = parent_body.bone_offset.copy()

        # Optionally rotate the limb direction around Z before rebuilding.
        if angle is not None:
            R = np.array([[np.cos(angle), -np.sin(angle), 0],
                          [np.sin(angle),  np.cos(angle), 0],
                          [0,              0,              1]])
            child_body.bone_offset = R @ child_body.bone_offset

        # Optionally tilt the limb around the X-axis (negative = downward).
        if tilt is not None:
            Rx = np.array([[1, 0,            0           ],
                           [0, np.cos(tilt), -np.sin(tilt)],
                           [0, np.sin(tilt),  np.cos(tilt)]])
            child_body.bone_offset = Rx @ child_body.bone_offset

        child_body.rebuild()
        child_body.sync_node()
        parent_body.node.append(child_node)
        self.bodies.append(child_body)
        self.sync_node()

    def remove_body(self, body):
        """Remove a body and its actuators from both the XML tree and the body list."""
        body.node.getparent().remove(body.node)
        body.parent.child.remove(body)
        self.bodies.remove(body)
        actu_section = body.tree.getroot().find('actuator')
        for joint in body.joints:
            actu_section.remove(joint.actuator.node)
        del body
        self.sync_node()

    def rebuild(self):
        """Recompute all absolute bone positions from stored bone_offsets.
        Call after any structural change (add/remove body) to propagate updates
        down the kinematic chain."""
        for body in self.bodies:
            body.rebuild()
            body.sync_node()

    def write_xml(self, fname):
        """Serialize the lxml tree to a file. pretty_print keeps XML human-readable."""
        self.tree.write(fname, pretty_print=True)

    def export_xml_string(self):
        """Return the XML as a byte string for in-memory use (e.g. mujoco.from_xml_string)."""
        return etree.tostring(self.tree, pretty_print=True)

    def get_gnn_edges(self):
        """Return a (2, E) edge index array with bidirectional parent↔child edges.
        Used to encode the robot's kinematic tree as input to a GNN policy."""
        edges = []
        for i, body in enumerate(self.bodies):
            if body.parent is not None:
                j = self.bodies.index(body.parent)
                edges.append([i, j])
                edges.append([j, i])
        return np.stack(edges, axis=1)


# Make a Robot from XML, add a limb, and write back to XML to test the wrapper.
if __name__ == '__main__':
    XML_PATH = 'assets/base_ant_flat.xml'
    OUT_PATH = 'assets/tmp/ant_modified.xml'

    # --- 1. Load ---
    robot = Robot(XML_PATH, 'Ant-v5')

    print('=== Initial bodies ===')
    for b in robot.bodies:
        print(f'  {b}  parent={b.parent}  pos={b.pos}  bone_start={b.bone_start}  bone_end={b.bone_end}')

    print(f'\nInitial GNN edges:\n{robot.get_gnn_edges()}\n')

    # --- 2. Add a limb to the main body (torso) at 90° ---
    torso = robot.bodies[0]
    print(f'Adding child to torso: {torso}')
    robot.add_child_to_body(torso, angle=np.pi / 2)

    # --- 3. Add a limb branching 90° sideways from body '1' ---
    leg = robot.bodies[1]
    print(f'Adding child to: {leg}')
    robot.add_child_to_body(leg, angle=np.pi / 2)

    print('\n=== Bodies after add_child_to_body ===')
    for b in robot.bodies:
        print(f'  {b}  parent={b.parent}  pos={b.pos}')

    print(f'\nNew GNN edges:\n{robot.get_gnn_edges()}\n')

    # --- 4. Write modified XML ---
    robot.write_xml(OUT_PATH)
    print(f'Written to {OUT_PATH}')


"""
When a new limb is added, it is added to the prev limb at an angle. 
"""
