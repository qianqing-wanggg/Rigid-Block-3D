"""
generate rigid block model for limit analysis
"""
import numpy as np
import matplotlib.pyplot as plt
import glob
import trimesh
from trimesh.voxel import creation
import gmsh
from tqdm import tqdm
import json
import sys
import os
import pymeshlab
from pymeshlab import PureValue
import scipy
from scipy.spatial import KDTree
from sklearn.decomposition import PCA
import csv
import pandas as pd
from plyfile import PlyData, PlyElement
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
    
    #beam_ground_element_center_to_interface = float(material_json['beam_ground_element_center_to_interface'])
    FAKE_thickness = 0.001#m
    
    mortar_to_mortar_property = {"contact_type":"friction_fc_cohesion","cohesion":cohesion_m_m_interface*1e6,\
                                "mu":mu_interface_mortar,"fc":fc_from_test*1e6,"ft":tensile_m_m_interface*1e6,\
                                "E":E_mortar*1e6,"Gf1":G_f1_mortar*1e3,"Gf2":G_f2_mortar*1e3,\
                                "Gc":G_c_mortar*1e3,"lambda":lambda_mortar}
    interface_stone_property = {"contact_type":"friction_fc_cohesion","cohesion":ratio_strength_interface*cohesion_m_m_interface*1e6,\
                                "mu":mu_interface_stone,"fc":fc_from_test*1e6,"ft":ratio_strength_interface*tensile_m_m_interface*1e6/1,\
                                "E":E_stone*1e6,"Gf1":G_f1_stone*1e3,"Gf2":G_f2_stone*1e3,\
                                "Gc":G_c_mortar*1e3,"lambda":lambda_stone}
    interface_mortar_property = {"contact_type":"friction_fc_cohesion","cohesion":ratio_strength_interface*cohesion_m_m_interface*1e6,\
                                "mu":mu_interface_stone,"fc":fc_from_test*1e6,"ft":ratio_strength_interface*tensile_m_m_interface*1e6/1,\
                                "E":E_mortar*1e6,"Gf1":G_f1_stone*1e3,"Gf2":G_f2_stone*1e3,\
                                "Gc":G_c_mortar*1e3,"lambda":lambda_stone}
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


def generate_ss_contact(material_json_path, stones_dir, output_dir, voxelize_pitch = 0.002):
    np.random.seed(315)
    scale_factor=1
    # read material_json.json
    with open(material_json_path) as json_file:
        material_json = json.load(json_file)

    stone_neiborhood_radius = 0.2#!need to be dynamic
    existing_points = pd.read_csv(os.path.join(output_dir, "point_mortar.csv"))
    contact_point_id = int(existing_points['id'].max())+1
    face_id = int(existing_points['face_id'].max())+1
    nb_points_per_intersection = int(material_json['nb_points_per_interface'])

    # Stone
    fc_stone = float(material_json['fc_stone'])#MPa
    mu_ss = float(material_json['mu_stone_stone'])#MPa
    cohesion_ss = float(material_json['cohesion_stone_stone'])#MPa
    E_stone = float(material_json['Emodulus_stone'])#MPa
    lambda_stone = float(material_json["lambda_stone"])
    G_f1_stone = 0
    G_f2_stone = 0
    G_c_stone  = float(material_json["G_c_stone"])

    stone_to_stone_property = {"contact_type":"friction_fc_cohesion","cohesion":cohesion_ss*1e6,"mu":mu_ss,"fc":fc_stone*1e6,\
                            "ft":0,"E":E_stone*1e6,"Gf1":G_f1_stone*1e3,"Gf2":G_f2_stone*1e3,\
                                "Gc":G_c_stone*1e3,"lambda":lambda_stone}


    def voxelize_mesh(mesh,radius,voxel_center,pitch):
        #print("Generating voxelization with radius: ",radius," pitch: ",pitch," voxel_center: ",voxel_center)
        stone_voxel = creation.local_voxelize(mesh,voxel_center,pitch,radius,fill=True)
        return stone_voxel.matrix

    def occ2points(coordinates):
        points  = []
        len = coordinates.shape[0]
        for i in range(len):
            points.append(np.array([round(coordinates[i,0]),round(coordinates[i,1]),round(coordinates[i,2])]))

        return np.array(points)


    def generate_faces(points):
        corners = np.zeros((8*len(points),3))
        faces = np.zeros((6*len(points),4))
        for index in range(len(points)):
            corners[index*8]= np.array([points[index,0]-0.5, points[index,1]-0.5, points[index,2]-0.5])
            corners[index*8+1]= np.array([points[index,0]+0.5, points[index,1]-0.5, points[index,2]-0.5])
            corners[index*8+2]= np.array([points[index,0]-0.5, points[index,1]+0.5, points[index,2]-0.5])
            corners[index*8+3]= np.array([points[index,0]+0.5, points[index,1]+0.5, points[index,2]-0.5])
            corners[index*8+4]= np.array([points[index,0]-0.5, points[index,1]-0.5, points[index,2]+0.5])
            corners[index*8+5]= np.array([points[index,0]+0.5, points[index,1]-0.5, points[index,2]+0.5])
            corners[index*8+6]= np.array([points[index,0]-0.5, points[index,1]+0.5, points[index,2]+0.5])
            corners[index*8+7]= np.array([points[index,0]+0.5, points[index,1]+0.5, points[index,2]+0.5])
            base=len(points)+8*index
            faces[index*6]= np.array([base+2, base+3,base+1,base+0])
            faces[index*6+1]= np.array([base+4, base+5, base+7,base+6])
            faces[index*6+2]= np.array([base+3, base+2, base+6,base+7])
            faces[index*6+3]= np.array([base+0, base+1, base+5,base+4])
            faces[index*6+4]= np.array([base+2, base+0,base+4,base+6])
            faces[index*6+5]= np.array([base+1, base+3,base+7,base+5])
        
        return corners, faces
    
    def write_ply(points, face_data, filename, text=True):

        points = [(points[i,0], points[i,1], points[i,2]) for i in range(points.shape[0])]

        vertex = np.array(points, dtype=[('x', 'f4'), ('y', 'f4'),('z', 'f4')])

        face = np.empty(len(face_data),dtype=[('vertex_indices', 'i4', (4,))])
        face['vertex_indices'] = face_data

        ply_faces = PlyElement.describe(face, 'face')
        ply_vertexs = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
        PlyData([ply_vertexs, ply_faces], text=text).write(filename)

    
    def writeocc(coordinates,save_path,filename,pitch = 1):
        points = occ2points(coordinates/pitch)
        #remove duplicate points
        points = np.unique(points,axis=0)

        #print(points.shape)
        corners, faces = generate_faces(points)
        if points.shape[0] == 0:
            print('the predicted mesh has zero point!')
        else:
            points = np.concatenate((points,corners),axis=0)
            write_ply(points, faces, os.path.join(save_path,filename))
    def get_overlap_points(mesh1,mesh2,plot_voxel=False,voxelize_pitch = 0.001):
        """
        Get the overlap points between two meshes
        """
        #get the origin of voxelization
        bbox_min = np.min(np.vstack((mesh1.bounds[0],mesh2.bounds[0])),axis=0)
        bbox_max = np.max(np.vstack((mesh1.bounds[1],mesh2.bounds[1])),axis=0)
        voxel_center = (bbox_min+bbox_max)/2
        pitch = voxelize_pitch
        radius = int(np.max(bbox_max-bbox_min)/2/pitch)
        #voxelize the two meshes
        stone_matrix_1 = voxelize_mesh(mesh1,radius,voxel_center,pitch)
        stone_matrix_2 = voxelize_mesh(mesh2,radius,voxel_center,pitch)
        #get the overlap points
        overlap_points = np.argwhere(np.logical_and(stone_matrix_1,stone_matrix_2))
        #convert the overlap points to the coordinates
        overlap_points = voxel_center + (overlap_points - np.array([stone_matrix_1.shape[0]/2,stone_matrix_1.shape[1]/2,stone_matrix_1.shape[2]/2]))*pitch
        #plot
        if plot_voxel:
            # save voxel
            coordinates = np.argwhere(stone_matrix_1!=0)
            coordinates=coordinates- np.array([stone_matrix_1.shape[0]/2,stone_matrix_1.shape[1]/2,stone_matrix_1.shape[2]/2])
            coordinates = coordinates*pitch+voxel_center
            result_dir="."
            writeocc(coordinates,result_dir,f'voxelization_stone_matrix_1.ply',\
                        pitch = pitch)
            # save voxel
            coordinates = np.argwhere(stone_matrix_2!=0)
            coordinates=coordinates- np.array([stone_matrix_1.shape[0]/2,stone_matrix_1.shape[1]/2,stone_matrix_1.shape[2]/2])
            coordinates = coordinates*pitch+voxel_center
            result_dir="."
            writeocc(coordinates,result_dir,f'voxelization_stone_matrix_2.ply',\
                        pitch = pitch)
            # save voxel
            coordinates = overlap_points
            result_dir="."
            writeocc(coordinates,result_dir,f'voxelization_ovelap.ply',\
                        pitch = pitch)
        
        return overlap_points

    def write_points_as_ply(points, filename,normal=None):
        """
        Write points to a ply file
        """
        with open(filename, 'w+') as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write("element vertex "+str(points.shape[0])+"\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            if normal is not None:
                f.write("property float nx\n")
                f.write("property float ny\n")
                f.write("property float nz\n")
            f.write("end_header\n")
            for point in points:
                if normal is not None:
                    f.write(str(point[0])+" "+str(point[1])+" "+str(point[2])+" "+str(normal[0])+" "+str(normal[1])+" "+str(normal[2])+"\n")
                else:
                    f.write(str(point[0])+" "+str(point[1])+" "+str(point[2])+"\n")
    def get_largest_triangle_points(points,nb_iterations = 100):
        """select three points that form the largest triangle
        """
        iteration = 0
        max_area = 0
        max_triangle = []
        max_points = []
        while iteration < nb_iterations:
            random_indices = np.random.choice(points.shape[0],3,replace=False)
            triangle = points[random_indices]
            area = np.linalg.norm(np.cross(triangle[1]-triangle[0],triangle[2]-triangle[0]))/2
            if area > max_area:
                max_area = area
                max_triangle = triangle
                max_points = random_indices
            iteration += 1
        return points[max_points]


    def get_mesh_center(mesh):
        return np.mean(mesh.vertices, axis=0)

    def are_collinear(coords, tol=None):
        coords = np.array(coords, dtype=float)
        coords -= coords[0] # offset for collinear points to intersect the origin
        return np.linalg.matrix_rank(coords, tol=tol)==1

    def get_envelop_area(points, normal):
        """
        Project 3d points to 2d plane and calculate the envelop area
        """
        #origin_plane = np.mean(points, axis=0)
        #find two directions perpendicular to the normal
        x = np.array([1,0,0])
        #print(abs(np.dot(x, normal)))
        if abs(np.dot(x, normal)) >= 0.999:
            x = np.array([1,1,0])
        #print("x: ",x)
        x = x - np.dot(x, normal) * normal
        print("normal:",normal)
        print("x: ",x)
        x /= np.sqrt((x**2).sum())
        y = np.cross(normal, x)
        #project the points to the plane
        projected_points = np.dot(points, np.array([x,y]).T)
        print("origional points:", points)
        print("projected points:",projected_points)
        #check if projected_points are colinear
        projected_points = np.unique(projected_points, axis=0)
        if are_collinear(projected_points):
            return 0,0
        if len(projected_points) < 3:
            return 0, 0
        #get the envelop area
        try:
            envelop_ch = scipy.spatial.ConvexHull(projected_points)
        except scipy.spatial.qhull.QhullError:
            return 0, 0
        envelop_area = envelop_ch.volume
        return envelop_area, envelop_ch

    #read iteration_id_to_element_id_map json to dict
    #iteration_id_to_element_id_map = dict()
    #with open(f"../result/result_{folder_id}/iteration_id_to_element_id_map.json") as f:
    #    iteration_id_to_element_id_map = json.load(f)


    # read stone meshes
    #stones_dir = "../data/data_"+folder_id+"/stones/"
    stone_files = glob.glob(os.path.join(stones_dir, "*stone_*.ply"))
    pc_element_centers = []
    element_index_to_stone_id_map = dict()
    stone_id_to_mesh_map = dict()
    if len(stone_files) == 0:
        print("no stone files found")
    else:
        for stone_file in stone_files:
            stone_id = int(stone_file.split("stone_")[1].split(".ply")[0])
            #iteration = int(stone_file.split("Stone")[1].split("_")[0])
            stone_mesh = trimesh.load(stone_file)
            #stone_mesh = pymesh.load_mesh(stone_file)
            stone_id_to_mesh_map[stone_id] = stone_mesh
            stone_center = stone_mesh.centroid
            pc_element_centers.append(stone_center)
            element_index_to_stone_id_map[len(pc_element_centers)-1] = stone_id
    #kd tree for element centers
    pc_element_centers = np.array(pc_element_centers)
    from scipy.spatial import KDTree
    element_centers_kd_tree = KDTree(pc_element_centers)

    #iterate stone meshes
    cand_anta_pairs = []
    contact_points = dict()
    for element_index in range(len(element_index_to_stone_id_map.keys())):
        stone_id = element_index_to_stone_id_map[element_index]
        element_center = pc_element_centers[element_index]
        cand_id = stone_id
        #find the stones in the neightborhood
        neighbor_indices = element_centers_kd_tree.query_ball_point(element_center, stone_neiborhood_radius)
        #print("Number of neighbors: ",len(neighbor_indices))
        neighbor_stone_ids = [element_index_to_stone_id_map[neighbor_index] for neighbor_index in neighbor_indices]
        #check overlap with all neighbors
        stone_mesh = stone_id_to_mesh_map[stone_id]
        for neighbor_k,neighbor_stone_id in enumerate(neighbor_stone_ids):
            # exclude the stone itself
            if neighbor_stone_id==stone_id:
                continue
            # exclude checked pairs
            if (stone_id,neighbor_stone_id) in cand_anta_pairs or (neighbor_stone_id,stone_id) in cand_anta_pairs:
                continue
            cand_anta_pairs.append((stone_id,neighbor_stone_id))
            # check overlap
            neighbor_stone_mesh = stone_id_to_mesh_map[neighbor_stone_id]
            #find intersection
            #expanded_stone_mesh = stone_mesh.copy()
            #expanded_stone_mesh.apply_scale(expand_mesh_scale)
            #expanded_neighbor_stone_mesh = neighbor_stone_mesh.copy()
            #expanded_neighbor_stone_mesh.apply_scale(expand_mesh_scale)
            #intersection = trimesh.boolean.intersection([expanded_stone_mesh,expanded_neighbor_stone_mesh])
            #print("Checking overlap between ",iteration_id," and ",neighbor_iteration_id)
            intersection = get_overlap_points(stone_mesh,neighbor_stone_mesh,voxelize_pitch=voxelize_pitch)
            if len(intersection) == 0 or len(intersection) < 0.5*nb_points_per_intersection:
                continue
            #_= get_overlap_points(stone_mesh,neighbor_stone_mesh,plot_voxel=True)
            
            #if intersection exists
            #print("Intersection found")
            anta_id = neighbor_stone_id
            neighbor_index = neighbor_indices[neighbor_k]
            neighbor_center = pc_element_centers[neighbor_index]
            # export the intersection
            #intersection.export(f"../result/result_{folder_id}/intersection_{iteration_id}_{neighbor_iteration_id}.ply")
            #sample points on the intersection
            #area_intersection = intersection.area
            #nb_points = int(intersection.area/(sample_intersection_points_radius**2))
            #intersection_sample_points = trimesh.sample.sample_surface_even(intersection, nb_points,radius = sample_intersection_points_radius)[0]
            #print(intersection_sample_points)
            #if len(intersection_sample_points) == 0:
            #    continue
            center_of_sample_points = np.mean(intersection, axis=0)
            #pca on the intersection points using sklearn
            pca = PCA(n_components=3)
            pca.fit(intersection)
            #get three principal components
            principal_components = pca.components_
            normal_dir = principal_components[-1]
            tangent1 = principal_components[0]
            tangent2 = principal_components[1]
            intersection_visual = intersection.copy()
            intersection_visual/=voxelize_pitch
            #write_points_as_ply(intersection_visual, f"../result/result_{folder_id}/intersection_{iteration_id}_{neighbor_iteration_id}.ply",normal = normal_dir)
            # #project the intersection to the plane
            # projected_polygon = intersection.projected(normal = normal_dir,origin = center_of_sample_points)
            # area_projected_polygon = projected_polygon.area
            #project the points to the plane
            envelop_area, envelop_ch = get_envelop_area(intersection,normal_dir)
            if envelop_area == 0:
                continue
            #randomly sample 3 point from intersection_sample_points
            #choose 3 points that form the largest triangle
            random_sample_points = get_largest_triangle_points(intersection,1000)
            random_sample_points_visual = random_sample_points.copy()
            random_sample_points_visual/=voxelize_pitch
            #write_points_as_ply(random_sample_points_visual, f"../result/result_{folder_id}/intersection_{iteration_id}_{neighbor_iteration_id}_contps.ply",normal = normal_dir)
            #a = input(f"overlap detected between {iteration_id} and {neighbor_iteration_id}, continue?")
            # random_sample_indices = np.random.choice(intersection.shape[0],nb_points_per_intersection, replace=False)
            # random_sample_points = intersection[random_sample_indices]
            for sample_point in random_sample_points:
                section_area = envelop_area*(scale_factor**2)/nb_points_per_intersection
                # find distance from the node to element center
                vector_to_element_center = element_center-sample_point
                proj_dist_to_element_center = np.dot(vector_to_element_center,-normal_dir)
                thickness = 2*abs(proj_dist_to_element_center)
                contact_points[contact_point_id] = {"id":contact_point_id, "coordinate":(sample_point*scale_factor).tolist(),\
                                                        "normal":normal_dir.tolist(),"tangent1":tangent1.tolist(),"tangent2":tangent2.tolist(),\
                                                            "candidate_id":cand_id,"antagonist_id":anta_id,\
                                                                "section_area":section_area,"contact_type":stone_to_stone_property["contact_type"],\
                                                                    "cohesion":stone_to_stone_property["cohesion"],"mu":stone_to_stone_property["mu"],\
                                                                        "fc":stone_to_stone_property["fc"],"ft":stone_to_stone_property["ft"],\
                                                                            "face_id":face_id,"E":stone_to_stone_property["E"],"thickness":thickness,\
                                                                            "counter_point":contact_point_id+1,"Gf1":stone_to_stone_property["Gf1"],\
                                                                            "Gf2":stone_to_stone_property["Gf2"],"Gc":stone_to_stone_property["Gc"]\
                                                                            ,"lambda":stone_to_stone_property["lambda"]}

                contact_point_id += 1
                #add counter point
                # find distance from the node to element center
                vector_to_element_center = neighbor_center-sample_point
                proj_dist_to_element_center = np.dot(vector_to_element_center,normal_dir)
                thickness = 2*abs(proj_dist_to_element_center)
                contact_points[contact_point_id] = {"id":contact_point_id, "coordinate":(sample_point*scale_factor).tolist(),\
                                                        "normal":(-normal_dir).tolist(),"tangent1":tangent1.tolist(),"tangent2":tangent2.tolist(),\
                                                            "candidate_id":anta_id,"antagonist_id":cand_id,\
                                                                "section_area":section_area,"contact_type":stone_to_stone_property["contact_type"],\
                                                                    "cohesion":stone_to_stone_property["cohesion"],"mu":stone_to_stone_property["mu"],\
                                                                        "fc":stone_to_stone_property["fc"],"ft":stone_to_stone_property["ft"],\
                                                                            "face_id":face_id,"E":stone_to_stone_property["E"],"thickness":thickness,\
                                                                            "counter_point":contact_point_id-1,"Gf1":stone_to_stone_property["Gf1"],\
                                                                            "Gf2":stone_to_stone_property["Gf2"],"Gc":stone_to_stone_property["Gc"]\
                                                                            ,"lambda":stone_to_stone_property["lambda"]}
                contact_point_id += 1
            face_id +=1
                    

    # write contact points
    contact_points_file = os.path.join(output_dir, "points_ss.csv")
    with open(contact_points_file, mode='w') as contact_points_file:
        contact_points_writer = csv.writer(contact_points_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        contact_points_writer.writerow(["id","x","y","z","nx","ny","nz","t1x","t1y","t1z","t2x","t2y","t2z",\
            "candidate_id","antagonist_id","section_area","contact_type","cohesion","mu","fc","ft","face_id",\
                "E","thickness","counter_point","Gf1","Gf2","Gc","lamda"])#"face_id","Gf1","Gf2","Gc","lamda"
        for contact_point_id in contact_points.keys():
            contact_point = contact_points[contact_point_id]
            contact_points_writer.writerow([contact_point["id"],contact_point["coordinate"][0]*scale_factor,contact_point["coordinate"][1]*scale_factor,contact_point["coordinate"][2]*scale_factor,\
                                            contact_point["normal"][0],contact_point["normal"][1],contact_point["normal"][2],\
                                                contact_point["tangent1"][0],contact_point["tangent1"][1],contact_point["tangent1"][2],\
                                                    contact_point["tangent2"][0],contact_point["tangent2"][1],contact_point["tangent2"][2],\
                                                        contact_point["candidate_id"],contact_point["antagonist_id"],contact_point["section_area"],\
                                                            contact_point["contact_type"],contact_point["cohesion"],contact_point["mu"],\
                                                                contact_point["fc"],contact_point["ft"],\
                                                                    contact_point["face_id"],contact_point["E"],contact_point["thickness"],\
                                                                        contact_point["counter_point"],contact_point["Gf1"],contact_point["Gf2"],\
                                                                        contact_point["Gc"],contact_point["lambda"]])
            
    

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
    if not stone_stone_contact:
        voxelize_pitch = 0
    else:
        voxelize_pitch = float(sys.argv[8])
    
    generate_rigid_block_model(material_json_path, mortar_ply_path, stones_dir, mortar_msh_path, output_dir, boundary_string)
    if not stone_stone_contact:
        #rename stone_stone_contact+"point_mortar.csv" to stone_stone_contact+mortar.csv
        points_combined = pd.read_csv(os.path.join(output_dir, "point_mortar.csv"))
        points_combined.to_csv(os.path.join(output_dir, "point.csv"))
    else:
        generate_ss_contact(material_json_path, stones_dir, \
            output_dir, voxelize_pitch = voxelize_pitch)
        # combine the points_ss.csv with points.csv
        points_ss = pd.read_csv(os.path.join(output_dir, "points_ss.csv"))
        points = pd.read_csv(os.path.join(output_dir, "point_mortar.csv"))
        points_combined = pd.concat([points,points_ss])
        points_combined.to_csv(os.path.join(output_dir, "point.csv"),index=False)
    #python /home/qiwang/Projects/28_RBSM_software/software/src/analysis/util_meshing.py /home/qiwang/Projects/28_RBSM_software/software/src/workspaces/run_SW3-E4-P/material.json /home/qiwang/Projects/28_RBSM_software/software/src/workspaces/run_SW3-E4-P/mortar.ply /home/qiwang/Projects/28_RBSM_software/software/src/workspaces/run_SW3-E4-P/temp /home/qiwang/Projects/28_RBSM_software/software/src/workspaces/run_SW3-E4-P/mortar_01.msh /home/qiwang/Projects/28_RBSM_software/software/src/workspaces/run_SW3-E4-P double_bending true
    #python /home/qiwang/Projects/28_RBSM_software/software/src/analysis/util_meshing.py /home/qiwang/Projects/28_RBSM_software/software/src/workspaces/run_SW3-E3/material.json /home/qiwang/Projects/28_RBSM_software/software/src/workspaces/run_SW3-E3/mortar_01.msh', '/home/qiwang/Projects/28_RBSM_software/software/src/workspaces/run_SW3-E3/temp', '/home/qiwang/Projects/28_RBSM_software/software/src/workspaces/run_SW3-E3/mortar_01.msh', '/home/qiwang/Projects/28_RBSM_software/software/src/workspaces/run_SW3-E3', 'double_bending', 'true'