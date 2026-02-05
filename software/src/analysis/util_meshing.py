"""
generate rigid block model for limit analysis
"""
import numpy as np
import matplotlib.pyplot as plt
import glob
import trimesh
import gmsh
from tqdm import tqdm
import json
import sys
import os
import pymeshlab
from pymeshlab import PureValue
from scipy.spatial import KDTree
import csv
import pandas as pd

def generate_rigid_block_model(material_json_path, mortar_ply_path, stones_dir, mortar_msh_path, output_dir, boundary_string="double_bending"):
    """
    Generate rigid block model for limit analysis
    
    Args:
        material_json_path: Path to material.json file
        mortar_ply_path: Path to mortar.ply file
        stones_dir: Directory containing stone mesh files (*stone_*.ply)
        mortar_msh_path: Path to mortar .msh file
        output_dir: Output directory for CSV files
        boundary_string: Boundary condition type (default: "double_bending")
    
    Returns:
        None (outputs point_mortar.csv and element.csv to output_dir)
    """
    print("Boundary condition: ", boundary_string)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # read material_json.json
    with open(material_json_path) as json_file:
        material_json = json.load(json_file)
    
    Cohesion_scale = 1/int(material_json['nb_points_per_interface'])
    # Strengths (MPa)
    fc_from_test = float(material_json["fc_from_test"])
    fc_stone = float(material_json["fc_stone"])
    fc_beam = float(material_json["fc_beam"])

    # Friction coefficients (dimensionless)
    mu_interface_stone = float(material_json["mu_interface_stone"])
    mu_interface_mortar = float(material_json["mu_interface_mortar"])
    mu_interface_beam = float(material_json["mu_interface_beam"])

    # Elastic moduli (MPa)
    E_stone = float(material_json["Emodulus_stone"])
    E_mortar = float(material_json["Emodulus_mortar"])
    E_beam = float(material_json["Emodulus_beam"])

    lambda_stone = float(material_json["lambda_stone"])
    lambda_mortar = float(material_json["lambda_mortar"])
    lambda_beam = float(material_json["lambda_beam"])



    # Interface ratios
    ratio_strength_interface = float(material_json["beta"])

    # Mortar interface strength params (MPa)
    cohesion_m_m_interface = float(material_json["m_m_cohesion"])
    tensile_m_m_interface = float(material_json["m_m_tensile"])

    # ---------------------------
    # Beam–mortar interface strength (MPa)
    # ---------------------------
    m_b_tensile = float(material_json["m_b_tensile"])
    m_b_cohesion = float(material_json["m_b_cohesion"])

    # ---------------------------
    # Fracture energies (N/mm)
    # ---------------------------
    # Stone
    G_f1_stone = float(material_json["G_f1_stone"])
    G_f2_stone = float(material_json["G_f2_stone"])
    G_c_stone  = float(material_json["G_c_stone"])

    # Mortar
    G_f1_mortar = float(material_json["G_f1_mortar"])
    G_f2_mortar = float(material_json["G_f2_mortar"])
    G_c_mortar  = float(material_json["G_c_mortar"])

    # Beam
    G_f1_beam = float(material_json["G_f1_beam"])
    G_f2_beam = float(material_json["G_f2_beam"])
    G_c_beam  = float(material_json["G_c_beam"])
    
    beam_ground_element_center_to_interface = float(material_json['beam_ground_element_center_to_interface'])
    FAKE_thickness = 0.001#m
    
    mortar_to_mortar_property = {"contact_type":"friction_fc_cohesion","cohesion":cohesion_m_m_interface*1e6,\
                                "mu":mu_interface_mortar,"fc":fc_from_test*1e6,"ft":tensile_m_m_interface*1e6,\
                                "E":E_mortar*1e6,"Gf1":G_f1_mortar*1e3,"Gf2":G_f2_mortar*1e3,\
                                "Gc":G_c_mortar*1e3,"lambda":lambda_mortar}
    interface_stone_property = {"contact_type":"friction_fc_cohesion","cohesion":ratio_strength_interface*cohesion_m_m_interface*1e6,\
                                "mu":mu_interface_stone,"fc":fc_stone*1e6,"ft":ratio_strength_interface*tensile_m_m_interface*1e6/1,\
                                "E":E_stone*1e6,"Gf1":G_f1_stone*1e3,"Gf2":G_f2_stone*1e3,\
                                "Gc":G_c_stone*1e3,"lambda":lambda_stone}
    interface_mortar_property = {"contact_type":"friction_fc_cohesion","cohesion":ratio_strength_interface*cohesion_m_m_interface*1e6,\
                                "mu":mu_interface_stone,"fc":fc_from_test*1e6,"ft":ratio_strength_interface*tensile_m_m_interface*1e6/1,\
                                "E":E_mortar*1e6,"Gf1":G_f1_stone*1e3,"Gf2":G_f2_stone*1e3,\
                                "Gc":G_c_stone*1e3,"lambda":lambda_stone}
    contact_to_beam_property = {"contact_type":"friction_fc_cohesion","cohesion":m_b_cohesion*1e6,\
                                "mu":mu_interface_beam,"fc":fc_beam*1e6,"ft":m_b_tensile*1e6,"E":E_beam*1e6,\
                                "Gf1":G_f1_beam*1e3,"Gf2":G_f2_beam*1e3,\
                                "Gc":G_c_beam*1e3,"lambda":lambda_beam}
    contact_to_ground_property = {"contact_type":"friction_fc_cohesion","cohesion":m_b_cohesion*1e6,\
                                "mu":mu_interface_beam,"fc":fc_beam*1e6,"ft":m_b_tensile*1e6,"E":E_beam*1e6,\
                                "Gf1":G_f1_beam*1e3,"Gf2":G_f2_beam*1e3,\
                                "Gc":G_c_beam*1e3,"lambda":lambda_beam}
    
    #save properties as json
    with open(os.path.join(output_dir, "properties.json"), "w+") as f:
        json.dump({"mortar_to_mortar_property":mortar_to_mortar_property,\
                   "interface_stone_property":interface_stone_property,\
                   "interface_mortar_property":interface_mortar_property,\
                       "contact_to_beam_property":contact_to_beam_property,\
                           "contact_to_ground_property":contact_to_ground_property,\
                            "Cohesion_scale":Cohesion_scale},f,indent=4)
    
    Axial_force = 0#-0.11*700*400#N
    Boundary_condition = boundary_string
    Wall_height = float(material_json['Wall_height'])
    Wall_diagonal = float(material_json['Wall_diagonal'])
    Sample_points_radius_to_D = float(material_json['Sample_points_radius_to_D'])
    Force_scaling = False#True
    Force_ground_beam_by_x = bool(material_json['Force_ground_beam_by_x'])
    
    # Load stone files
    stone_files = glob.glob(os.path.join(stones_dir, "*stone_*.ply"))
    
    wall_mesh = trimesh.load(mortar_ply_path)
    wall_center = wall_mesh.centroid
    max_x = np.max(wall_mesh.vertices[:,0])
    min_x = np.min(wall_mesh.vertices[:,0])
    height_in_mesh = max_x - min_x
    if Force_scaling:
        scale_factor = Wall_height/height_in_mesh
    else:
        scale_factor = 1
    
    Density = {"stone":float(material_json['Density_stone'])*9.81,"mortar":float(material_json['Density_mortar'])*9.81}#N/m^3
    Sample_points_radius = Wall_diagonal*Sample_points_radius_to_D/scale_factor#m
    
    wall_plane_zs = material_json['wall_plane_zs']
    wall_plane_ys = material_json['wall_plane_ys']
    mortar_bound_xs = material_json['wall_plane_xs']
    
    def move_point1_toward_point2(point1, point2,distance = None):
        """move point1 toward point2 by distance
        """
        direction = point2 - point1
        direction = direction/np.linalg.norm(direction)
        return point1 + direction*distance
    
    def get_sample_points_from_surface(mesh, nsamples,unique_point_radius = Sample_points_radius):
        """return a list of point coordinates sampled from the surface of the mesh
        """
        pymeshlab_mesh_set =  pymeshlab.MeshSet()
        pymeshlab_mesh_set.add_mesh(pymeshlab.Mesh(mesh.vertices,mesh.faces))
        pymeshlab_mesh_set.meshing_isotropic_explicit_remeshing(targetlen = PureValue(unique_point_radius))
        sample_points = pymeshlab_mesh_set.mesh(0).vertex_matrix().tolist()
        return sample_points
    
    def get_volume_tetrahedron(node_coords):
        """return the volume of a tetrahedron
        """
        a = node_coords[0]
        b = node_coords[1]
        c = node_coords[2]
        d = node_coords[3]
        return np.abs(np.dot(a-d,np.cross(b-d,c-d)))/6
    
    # read stone meshes
    elems = dict()
    sample_points = []
    sample_point_to_element_id_map = dict()
    #iteration_id_to_element_id_map = dict()
    element_id = 0
    pc_element_centers = []
    
    if len(stone_files) == 0:
        print("!!!!!!!!!!!!no stone files found!!!!!!!!!!!!!!!!!!!!!")
    else:
        for stone_file in stone_files:
            stone_id = int(stone_file.split("stone_")[1].split(".ply")[0])
            stone_mesh = trimesh.load(stone_file)
            stone_center = stone_mesh.centroid
            stone_volume = abs(stone_mesh.volume)
            surface_area = stone_mesh.area
            elems[element_id] = {"id":stone_id, "mesh":stone_mesh, "center":stone_center, "volume":stone_volume,"element_id":element_id,"type":f"stone_{stone_id}"}
            #iteration_id_to_element_id_map[iteration] = element_id
            if len(sample_points) == 0:
                sample_points = get_sample_points_from_surface(stone_mesh, int(surface_area/(Sample_points_radius**2)),unique_point_radius = Sample_points_radius)
                sample_point_to_element_id_map = dict.fromkeys(list(range(len(sample_points))),element_id)
            else:
                prev_sample_points_length = len(sample_points)
                sample_points.extend(get_sample_points_from_surface(stone_mesh, int(surface_area/(Sample_points_radius**2)),unique_point_radius = Sample_points_radius))
                sample_point_to_element_id_map.update(dict.fromkeys(list(range(prev_sample_points_length,len(sample_points))),element_id))
            element_id += 1
    
    # #write iteration_id_to_element_id_map to json
    # with open(os.path.join(output_dir, "iteration_id_to_element_id_map.json"), "w+") as f:
    #     json.dump(iteration_id_to_element_id_map,f,indent=4)
    
    # read wall mesh
    wall_mesh = trimesh.load(mortar_ply_path)
    wall_center = wall_mesh.centroid
    max_x = np.max(wall_mesh.vertices[:,0])
    min_x = np.min(wall_mesh.vertices[:,0])
    
    #generate beam
    beam_center = np.array([max_x,wall_center[1],wall_center[2]])
    elems[element_id] = {"element_id":element_id, "center":beam_center,"type":"beam","volume":0}
    beam_nodes = np.array([[max_x,wall_mesh.bounds[0][1],wall_mesh.bounds[0][2]],\
                            [max_x,wall_mesh.bounds[0][1],wall_mesh.bounds[1][2]],\
                            [max_x,wall_mesh.bounds[1][1],wall_mesh.bounds[1][2]],\
                            [max_x,wall_mesh.bounds[1][1],wall_mesh.bounds[0][2]]])
    beam_mesh = trimesh.Trimesh(vertices=beam_nodes,faces=[[0,1,2],[0,2,3]])
    surface_area = beam_mesh.area
    prev_sample_points_length = len(sample_points)
    sample_points.extend(get_sample_points_from_surface(beam_mesh, int(surface_area/(Sample_points_radius**2))))
    sample_point_to_element_id_map.update(dict.fromkeys(list(range(prev_sample_points_length,len(sample_points))),element_id))
    element_id += 1
    
    # generate ground
    ground_center = np.array([min_x,wall_center[1],wall_center[2]])
    elems[element_id] = {"element_id":element_id, "center":ground_center,"type":"ground","volume":0}
    ground_nodes = np.array([[min_x,wall_mesh.bounds[0][1],wall_mesh.bounds[0][2]],\
                            [min_x,wall_mesh.bounds[0][1],wall_mesh.bounds[1][2]],\
                            [min_x,wall_mesh.bounds[1][1],wall_mesh.bounds[1][2]],\
                            [min_x,wall_mesh.bounds[1][1],wall_mesh.bounds[0][2]]])
    ground_mesh = trimesh.Trimesh(vertices=ground_nodes,faces=[[0,1,2],[0,2,3]])
    surface_area = ground_mesh.area
    prev_sample_points_length = len(sample_points)
    sample_points.extend(get_sample_points_from_surface(ground_mesh, int(surface_area/(Sample_points_radius**2))))
    sample_point_to_element_id_map.update(dict.fromkeys(list(range(prev_sample_points_length,len(sample_points))),element_id))
    element_id += 1
    
    #generate point tree for each sample points
    sample_points_tree = KDTree(sample_points)
    
    #write sample points to ply
    sample_points_ply = trimesh.points.PointCloud(sample_points)
    sample_points_ply.export(os.path.join(output_dir, "sample_points.ply"))
    
    # generate contact points
    contact_points = dict()
    contact_point_id = 0
    mortar_elements = dict()
    face_centers = []
    face_center_to_element_map = dict()
    mort_tret_to_elem_id_map = dict()
    
    # read mortar gmsh
    try:
        gmsh.initialize()
    except ValueError as e:
        # Signal handling error in non-main thread - try alternative initialization
        if "signal only works in main thread" in str(e):
            print("Warning: Running in non-main thread, gmsh signal handling disabled")
            # Initialize gmsh without signal handlers
            import sys
            # Store original argv
            original_argv = sys.argv.copy()
            # Initialize with -noenv to avoid signal issues
            sys.argv = ['', '-noenv']
            try:
                gmsh.initialize()
            finally:
                sys.argv = original_argv
        else:
            raise
    
    gmsh.open(mortar_msh_path)
    
    # Initialize a dictionary to hold faces
    face_dict = {}
    
    # Get all elements
    element_types, element_tags, element_node_tags = gmsh.model.mesh.getElements()
    
    #iterate over elements
    for element_tag_by_type, element_type, element_node_tag in zip(element_tags, element_types, element_node_tags):
        if element_type == 4:
            #iterate over each element
            for element_tag in tqdm(element_tag_by_type,desc="Generate mortar elements"):
                elementType, nodetags, dim, tag = gmsh.model.mesh.getElement(element_tag)
                if len(nodetags) != 4:
                    print(f"number of nodes is not 4: {len(nodetags)}")
                    print("elementType",elementType)
                    print("dim",dim)
                
                #get element center
                node_coords = [gmsh.model.mesh.getNode(nodetag)[0] for nodetag in nodetags]
                element_center = np.mean(node_coords,axis=0)
                #get element volume
                element_volume = get_volume_tetrahedron(node_coords)
                element_type = "mortar"
                mortar_elements[element_tag] = {"element_id":element_id, "center":element_center,"type":element_type,"volume":element_volume}
                element_id += 1
                
                #iterate over faces of the element
                for except_index in range(4):
                    sorted_face = tuple(sorted([nodetags[i] for i in range(4) if i != except_index]))
                    if sorted_face not in face_dict.keys():
                        face_dict[sorted_face] = [element_tag]
                    else:
                        face_dict[sorted_face].append(element_tag)
    
    face_id = 0
    contact_points_with_ground_pc = []
    potential_ground_contact_point = []
    
    for face, tetras in tqdm(face_dict.items(),desc="Generate contact points face by face"):
        cand_element_tag = tetras[0]
        face_id += 1
        face_node_coords = [gmsh.model.mesh.getNode(i)[0] for i in face]
        face_center = np.mean(face_node_coords,axis=0)
        
        if len(tetras) == 2:
            # internal face
            property_this_interface = mortar_to_mortar_property
            anta_id = mortar_elements[tetras[1]]["element_id"]
            anta_center = mortar_elements[tetras[1]]["center"]
            property_anta_interface = mortar_to_mortar_property
        elif len(tetras) == 1:
            # boundary face
            max_nearest_point_distance,nearest_point_index = sample_points_tree.query(face_center)
            for node_on_face in face_node_coords:
                distance,p_index = sample_points_tree.query(node_on_face)
                if distance > max_nearest_point_distance:
                    max_nearest_point_distance = distance
                    nearest_point_index = p_index
            if max_nearest_point_distance > Sample_points_radius:# a free face without contact
                property_this_interface = None
                continue
            anta_id = sample_point_to_element_id_map[nearest_point_index]
            if elems[anta_id]["type"] == "beam":
                property_this_interface = contact_to_beam_property
                anta_center = elems[anta_id]["center"]
                property_anta_interface = contact_to_beam_property
            elif elems[anta_id]["type"] == "ground":
                property_this_interface = contact_to_ground_property
                anta_center = elems[anta_id]["center"]
                property_anta_interface = contact_to_ground_property
            else:
                property_this_interface = interface_mortar_property
                anta_center = elems[anta_id]["center"]
                property_anta_interface = interface_stone_property
        else:
            print(f"ERROR: {len(tetras)} tetras found")
            property_this_interface = None    
        
        if property_this_interface is not None:
            # apply mortar-ground/mortar-beam contact according to x coordinate
            if Force_ground_beam_by_x:
                if face_center[0]<=mortar_bound_xs[0]:
                    property_this_interface = contact_to_ground_property
                    property_anta_interface = contact_to_ground_property
                elif face_center[0]>=mortar_bound_xs[1]:
                    property_this_interface = contact_to_beam_property
                    property_anta_interface = contact_to_beam_property
            
            # add contact points
            face_node_coords = [gmsh.model.mesh.getNode(i)[0] for i in face]
            face_center = np.mean(face_node_coords,axis=0)
            
            # find normal direction of the face by cross product of two edges
            edge1 = face_node_coords[1]-face_node_coords[0]
            edge2 = face_node_coords[2]-face_node_coords[0]
            normal = np.cross(edge1,edge2)
            normal = normal/np.linalg.norm(normal)
            
            # orient normal to the center of the element
            center_to_face = face_center - mortar_elements[cand_element_tag]["center"]
            if np.dot(normal,center_to_face) > 0:
                normal = -normal
            
            #skip boundary faces
            if face_center[1]*scale_factor<wall_plane_ys[0]+0.001 or \
                face_center[1]*scale_factor>wall_plane_ys[1]-0.001:
                if abs(normal[1])>0.999:
                    continue
            if face_center[2]*scale_factor<wall_plane_zs[0]+0.001 or \
                        face_center[2]*scale_factor>wall_plane_zs[1]-0.001:
                if abs(normal[2])>0.999:
                    continue
            
            # define tangent1 direction
            if np.linalg.norm(np.cross(normal,np.array([1,0,0]))) < 1e-6:
                helper_vector = np.array([0,1,0])
                tangent1 = np.cross(normal,helper_vector)
            else:
                helper_vector = np.array([1,0,0])
                tangent1 = np.cross(normal,helper_vector)
            tangent1 = tangent1/np.linalg.norm(tangent1)
            
            # define tangent2 direction
            tangent2 = np.cross(normal,tangent1)
            tangent2 = tangent2/np.linalg.norm(tangent2)
            
            # find area of the triangle face
            section_area = np.linalg.norm(np.cross(edge1,edge2))/2
            
            for node_of_the_face in face_node_coords:
                if len(node_of_the_face) != 3:
                    print(node_of_the_face)
                    print("----------------")
                distance_from_point_to_face_center = np.linalg.norm(node_of_the_face-face_center)
                node_of_the_face = move_point1_toward_point2(node_of_the_face,face_center,distance=(distance_from_point_to_face_center/2))
                
                # find distance from the node to element center
                vector_to_element_center = mortar_elements[cand_element_tag]["center"]-node_of_the_face
                proj_dist_to_element_center = np.dot(vector_to_element_center,-normal)
                thickness = 2*abs(proj_dist_to_element_center)
                if thickness==0:
                    thickness = FAKE_thickness
                
                contact_points[contact_point_id] = {"id":contact_point_id, "coordinate":node_of_the_face.tolist(),\
                                                    "normal":normal.tolist(),"tangent1":tangent1.tolist(),"tangent2":tangent2.tolist(),\
                                                        "candidate_id":mortar_elements[cand_element_tag]["element_id"],"antagonist_id":anta_id,\
                                                            "section_area":section_area,"contact_type":property_this_interface["contact_type"],\
                                                                "cohesion":property_this_interface["cohesion"],"mu":property_this_interface["mu"],\
                                                                    "fc":property_this_interface["fc"],"ft":property_this_interface["ft"],\
                                                                        "face_id":face_id,"E":property_this_interface["E"],"thickness":thickness,\
                                                                            "counter_point":contact_point_id+1,"Gf1":property_this_interface["Gf1"],\
                                                                            "Gf2":property_this_interface["Gf2"],"Gc":property_this_interface["Gc"]\
                                                                            ,"lambda":property_this_interface["lambda"]}
                contact_point_id += 1
                
                # create counterpoint on the anta element
                vector_to_element_center = anta_center-node_of_the_face
                proj_dist_to_element_center = np.dot(vector_to_element_center,normal)
                thickness = 2*abs(proj_dist_to_element_center)
                if thickness==0:
                    thickness = FAKE_thickness
                
                contact_points[contact_point_id] = {"id":contact_point_id, "coordinate":node_of_the_face.tolist(),\
                                                    "normal":(-normal).tolist(),"tangent1":tangent1.tolist(),"tangent2":tangent2.tolist(),\
                                                        "candidate_id":anta_id,"antagonist_id":mortar_elements[cand_element_tag]["element_id"],\
                                                            "section_area":section_area,"contact_type":property_anta_interface["contact_type"],\
                                                                "cohesion":property_anta_interface["cohesion"],"mu":property_anta_interface["mu"],\
                                                                    "fc":property_anta_interface["fc"],"ft":property_anta_interface["ft"],\
                                                                        "face_id":face_id,"E":property_anta_interface["E"],"thickness":thickness,\
                                                                            "counter_point":contact_point_id-1,"Gf1":property_anta_interface["Gf1"],\
                                                                            "Gf2":property_anta_interface["Gf2"],"Gc":property_anta_interface["Gc"]\
                                                                            ,"lambda":property_anta_interface["lambda"]}
                contact_point_id += 1
                
                if anta_id in elems.keys() and elems[anta_id]["type"] == "ground":
                    contact_points_with_ground_pc.append(node_of_the_face)
                elif anta_id in elems.keys():
                    potential_ground_contact_point.append(node_of_the_face)
    
    contact_points_with_ground_pc=np.array(contact_points_with_ground_pc)
    contact_points_with_ground_ply = trimesh.points.PointCloud(contact_points_with_ground_pc)
    contact_points_with_ground_ply.export(os.path.join(output_dir, "contact_points_with_ground.ply"))
    
    potential_ground_contact_point=np.array(potential_ground_contact_point)
    potential_ground_contact_ply = trimesh.points.PointCloud(potential_ground_contact_point)
    potential_ground_contact_ply.export(os.path.join(output_dir, "potential_ground_contact_point.ply"))
    
    gmsh.finalize()
    
    # write the maximal contact point id
    with open(os.path.join(output_dir, "parameters.json"), "w+") as f:
        json.dump({"max_contact_point_id":contact_point_id},f,indent=4)
    
    # write contact points
    contact_points_file = os.path.join(output_dir, "point_mortar.csv")
    with open(contact_points_file, mode='w') as contact_points_file:
        contact_points_writer = csv.writer(contact_points_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        contact_points_writer.writerow(["id","x","y","z","nx","ny","nz","t1x","t1y","t1z","t2x","t2y","t2z",\
            "candidate_id","antagonist_id","section_area","contact_type","cohesion","mu","fc","ft","face_id",\
                "E","thickness","counter_point","Gf1","Gf2","Gc","lamda"])
        for contact_point_id in contact_points.keys():
            contact_point = contact_points[contact_point_id]
            section_area = contact_point["section_area"]*scale_factor**2
            thickness = contact_point["thickness"]*scale_factor
            contact_points_writer.writerow([contact_point["id"],contact_point["coordinate"][0]*scale_factor,contact_point["coordinate"][1]*scale_factor,contact_point["coordinate"][2]*scale_factor,\
                                            contact_point["normal"][0],contact_point["normal"][1],contact_point["normal"][2],\
                                                contact_point["tangent1"][0],contact_point["tangent1"][1],contact_point["tangent1"][2],\
                                                    contact_point["tangent2"][0],contact_point["tangent2"][1],contact_point["tangent2"][2],\
                                                        contact_point["candidate_id"],contact_point["antagonist_id"],section_area*Cohesion_scale,\
                                                            contact_point["contact_type"],contact_point["cohesion"],contact_point["mu"],\
                                                                contact_point["fc"],contact_point["ft"],\
                                                                    contact_point["face_id"],contact_point["E"],contact_point["thickness"],\
                                                                        contact_point["counter_point"],contact_point["Gf1"],contact_point["Gf2"],\
                                                                        contact_point["Gc"],contact_point["lambda"]])
    
    # write elements
    elements_file = os.path.join(output_dir, "element.csv")
    with open(elements_file, mode='w') as elements_file:
        elements_writer = csv.writer(elements_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        elements_writer.writerow(["id","type","cx","cy","cz","mass","shape","dl_fx","dl_fy","dl_fz","dl_mx","dl_my","dl_mz","ll_fx","ll_fy","ll_fz","ll_mx","ll_my","ll_mz"])
        for element_id in elems.keys():
            element = elems[element_id]
            if element["type"]=="ground":
                elements_writer.writerow([element["element_id"],element["type"],element["center"][0]*scale_factor,element["center"][1]*scale_factor,element["center"][2]*scale_factor,0,\
                                         None,0,0,0,0,0,0,0,0,0,0,0,0])
            elif element["type"]=="beam":
                if Boundary_condition == "double_bending":
                    verticle_height = element["center"][0]*scale_factor/2
                else:
                    verticle_height = element["center"][0]*scale_factor
                elements_writer.writerow([element["element_id"],element["type"],verticle_height,element["center"][1]*scale_factor,element["center"][2]*scale_factor,0,\
                                         None,Axial_force,0,0,0,0,0,0,1,0,0,0,0])
            else:
                gravity_load = -element["volume"]*Density["stone"]*scale_factor**3
                elements_writer.writerow([element["element_id"],element["type"],element["center"][0]*scale_factor,element["center"][1]*scale_factor,element["center"][2]*scale_factor,element["volume"]*Density["stone"]*scale_factor**3,\
                                         None,gravity_load,0,0,0,0,0,0,0,0,0,0,0])
        for element_tag in mortar_elements.keys():
            element = mortar_elements[element_tag]
            gravity_load = -element["volume"]*Density["mortar"]*scale_factor**3
            elements_writer.writerow([element["element_id"],element["type"],element["center"][0]*scale_factor,element["center"][1]*scale_factor,element["center"][2]*scale_factor,element["volume"]*Density["mortar"]*scale_factor**3,\
                                         None,gravity_load,0,0,0,0,0,0,0,0,0,0,0])
    
    print(f"Successfully generated point_mortar.csv and element.csv in {output_dir}")


# Command line interface - keep original behavior if run as script
if __name__ == "__main__":
    if len(sys.argv) < 6:
        print("Usage: python script.py <material_json_path> <mortar_ply_path> <stones_dir> <mortar_msh_path> <output_dir> [boundary_string]")
        sys.exit(1)
    
    material_json_path = sys.argv[1]
    mortar_ply_path = sys.argv[2]
    stones_dir = sys.argv[3]
    mortar_msh_path = sys.argv[4]
    output_dir = sys.argv[5]
    boundary_string = sys.argv[6] if len(sys.argv) > 6 else "double_bending"
    stone_stone_contact = False if sys.argv[7]=="false" else True
    
    generate_rigid_block_model(material_json_path, mortar_ply_path, stones_dir, mortar_msh_path, output_dir, boundary_string)
    if not stone_stone_contact:
        #rename stone_stone_contact+"point_mortar.csv" to stone_stone_contact+mortar.csv
        points_combined = pd.read_csv(os.path.join(output_dir, "point_mortar.csv"))
        points_combined.to_csv(os.path.join(output_dir, "point.csv"))
