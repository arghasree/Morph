"""
In this file, we are investigating the structure of the XML files that define the robot designs. 
We want to understand how the robot's body is structured in terms of its limbs and joints.
Things that are investigated:
1. How many bodies are there in the XML file? (Each body can be thought of as a limb or part of the robot)
2. How are the bodies connected to each other? (This will help us understand the robot's structure and how it moves)
3. What are the lengths of the limbs? (This will help us understand the size of the robot and how it interacts with the environment)
4. How are the joints defined? (This will help us understand how the robot can move and what kind of actions it can take)
5. How is the robot's starting position defined? (This will help us understand how the robot is initialized in the simulation and how it interacts with the ground)
6. The number of limbs
"""


import xml.etree.ElementTree as ET
import math

def investigate_xml(xml_path):
    # Load the XML file
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # Find the <worldbody> element
    worldbody = root.find('worldbody')
    print(f'Worldbody element: {worldbody}')
    
    # Find the root body (the one with name="0")
    body_zero = worldbody.find(".//body[@name='0']")
    print(f'Body with name "0": {body_zero}')
    
    if body_zero is None:
        print("No body with name '0' found in the XML.")
        return
    
def find_legs_and_joints(xml_path):
    # Load the XML file
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # Find the <worldbody> element
    worldbody = root.find('worldbody')
    
    # Find the root body (the one with name="0")
    body_zero = worldbody.find(".//body[@name='0']")
    
    # Build a child->parent map so we can look up each joint's parent body.
    # ET doesn't expose .getparent(), so we construct it by iterating the whole tree.
    parent_map = {}
    for parent in root.iter():      # walk every element in the XML tree
        for child in parent:        # for each direct child of that element
            print(f"Parent: {parent.tag}, Child: {child.tag}")  # Debug print to see the structure
            parent_map[child] = parent  # record: this child's parent is 'parent'

    # Find the joints in the XML
    joints = root.findall('.//joint')
    print(f'Number of joints in the XML: {len(joints)}')
    for joint in joints:
        parent_body = parent_map.get(joint)
        body_name = parent_body.get('name') if parent_body is not None else 'unknown'
        print(f"Joint name: {joint.get('name')}, body: {body_name}, type: {joint.get('type')}, axis: {joint.get('axis')}")
        
        
def number_of_bodies(xml_path):
    # Load the XML file
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # Find the <worldbody> element
    worldbody = root.find('worldbody')
    
    # Count all <body> elements in the worldbody
    bodies = worldbody.findall('.//body')
    print(f'Number of bodies in the XML: {len(bodies)}')
    for body in bodies:
        print(f"Body name: {body.get('name')}, pos: {body.get('pos')}, quat: {body.get('quat')}")
        
        
def legs(xml_path):
    """Finds all the legs and their lengths"""
    # Load the XML file
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # Find the <worldbody> element
    worldbody = root.find('worldbody')
    
    # Find the root body (the one with name="0")
    body_zero = worldbody.find(".//body[@name='0']")
    
    # Find the bodies connected to the body "0" (these are the limbs)
    limb_bodies = body_zero.findall('body') # Only direct children of body "0" are considered here, not deeper bodies in the hierarchy.
    print(f'Number of limbs (bodies only directly connected to body "0"): {len(limb_bodies)}')
    
    results = []
    recursive_limbs(body_zero, results)
    for body_name, total_length in results:
        print(f"Body: {body_name}, total length: {total_length:.4f}")
                
                
def recursive_limbs(body, results, depth=0):
    """Recursively find all limbs and their total lengths, starting from a given body.

    For each child body, sums the lengths of all its geom segments, then recurses
    into that child's children. Returns a list of (body_name, total_length) tuples
    for every body in the subtree.
    """
    children = body.findall('body')
    if not children:
        return results  # Base case: leaf node, no more limbs to explore

    for child in children:      # direct children of this body
        # Sum the lengths of all geom segments in this child body
        total_length = 0.0
        for geom in child.findall('geom'):
            fromto = geom.get('fromto')
            if fromto:                      # only capsules have fromto; skip spheres
                total_length += calculate_segment_length(fromto)

        indent = "  " * depth              # indent for readable printout
        print(f"{indent}Body: {child.get('name')}, segment length: {total_length:.4f}")
        results.append((child.get('name'), total_length))

        # Recurse into this child's children (sub-limbs)
        recursive_limbs(child, results, depth + 1)

    # return results
                

def calculate_segment_length(fromto):
    import numpy as np
    coords = [float(x) for x in fromto.split()]
    start = np.array(coords[:3])   # first 3 numbers = start point
    end   = np.array(coords[3:])   # last  3 numbers = end point
    length = np.linalg.norm(end - start)
    return length

# Recursive function to find the deepest branch and collect geoms along that branch
def find_deepest_branch(body, current_depth=1, bodies_path=None):
    """Finds the longest chain of bodies (deepest branch) in the kinematic tree.
    At each call, we try every child body and recurse. Whichever child leads to
    the greatest depth wins — that branch is returned as the deepest path.
    Returns:
        max_depth    : int,  depth of the deepest branch found
        deepest_path : list of body elements from root down to the deepest leaf
        total_length : float, sum of all geom segment lengths along that path
    """
    if bodies_path is None:
        bodies_path = []

    bodies_path = bodies_path + [body]  # add current body to the path (don't mutate the caller's list)

    child_bodies = body.findall('body')

    # Base case: leaf node — no children, so this is the end of a branch
    if not child_bodies:
        # Sum all geom lengths at this leaf body
        total_length = 0.0
        for geom in body.findall('geom'):
            if geom.get('fromto'):
                total_length+= calculate_segment_length(geom.get('fromto'))

        return current_depth, bodies_path, total_length

    # Recursive case: try each child and keep track of which gives the deepest branch
    max_depth = -1
    deepest_path = bodies_path
    deepest_length = 0.0

    for child in child_bodies:
        depth, path, length = find_deepest_branch(child, current_depth + 1, bodies_path)

        # Add this body's own geom lengths to the child's accumulated length
        own_length = sum(
            calculate_segment_length(geom.get('fromto'))
            for geom in body.findall('geom')
            if geom.get('fromto')
        )

        if depth > max_depth:
            max_depth = depth
            deepest_path = path
            deepest_length = length + own_length

    return max_depth, deepest_path, deepest_length

def get_deepest_branch_length(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    worldbody = root.find('worldbody')
    body_zero = worldbody.find(".//body[@name='0']")
    
    max_depth, deepest_path, total_length = find_deepest_branch(body_zero)
    print(f"Deepest branch depth: {max_depth}, total length: {total_length:.4f}")
    return total_length


xml_path = 'assets/base_ant_flat.xml'
investigate_xml(xml_path)
find_legs_and_joints(xml_path)
number_of_bodies(xml_path)
legs(xml_path)
get_deepest_branch_length(xml_path)

"""
Output: 
------------------------------------------------------------------
Worldbody element: <Element 'worldbody' at 0x149c9f05be70>
Body with name "0": <Element 'body' at 0x149c9f05bf60>

Parent: mujoco, Child: compiler
Parent: mujoco, Child: option
Parent: mujoco, Child: custom
Parent: mujoco, Child: default
Parent: mujoco, Child: asset
Parent: mujoco, Child: worldbody
Parent: mujoco, Child: actuator
Parent: custom, Child: numeric
Parent: default, Child: joint
Parent: default, Child: geom
Parent: asset, Child: texture
Parent: asset, Child: texture
Parent: asset, Child: texture
Parent: asset, Child: material
Parent: asset, Child: material
Parent: worldbody, Child: light
Parent: worldbody, Child: geom
Parent: worldbody, Child: body
Parent: body, Child: camera
Parent: body, Child: geom
Parent: body, Child: joint
Parent: body, Child: body
Parent: body, Child: geom
Parent: body, Child: joint
Parent: actuator, Child: motor

Number of joints in the XML: 3
Joint name: None, body: None, type: None, axis: None
Joint name: root, body: 0, type: free, axis: None
Joint name: body_1, body: 1, type: hinge, axis: 0.707 0.707 0

Number of bodies in the XML: 2
Body name: 0, pos: 0 0 0.75, quat: None
Body name: 1, pos: 0.2 0.2 0, quat: None

Number of limbs (bodies only directly connected to body "0"): 1
Body: 1, segment length: 0.2828
Body: 1, total length: 0.2828

Deepest branch depth: 2, total length: 0.5657
------------------------------------------------------------------

The actual form of this ant is very simple and can be found in: /home/arghasre/scratch/CMPUT 605/C605_Cogent/arghasre/ant_random_video.mp4
"""