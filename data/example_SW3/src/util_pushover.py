
import numpy as np
import math
from Kinematics import cal_gap_3d
import csv
import pandas as pd
def update_model_from_csv(model,pause_file_element,pause_file_point):
    # update center of elements
    displacement = pd.read_csv(pause_file_element)
    displacement = displacement[['id','cx','cy','cz']]
    for e in model.elems.values():
        e.center[0] = displacement[displacement['id']==e.id]['cx'].values[0]
        e.center[1] = displacement[displacement['id']==e.id]['cy'].values[0]
        e.center[2] = displacement[displacement['id']==e.id]['cz'].values[0]
    # update coor of contact points
    displacement = pd.read_csv(pause_file_point)
    displacement = displacement[['id','x','y','z','normal_force','tangent_force','tangent_force2']]
    for p in model.contps.values():
        p.coor[0] = displacement[displacement['id']==p.id]['x'].values[0]
        p.coor[1] = displacement[displacement['id']==p.id]['y'].values[0]
        p.coor[2] = displacement[displacement['id']==p.id]['z'].values[0]
        p.normal_force = displacement[displacement['id']==p.id]['normal_force'].values[0]
        p.tangent_force = displacement[displacement['id']==p.id]['tangent_force'].values[0]
        p.tangent_force2 = displacement[displacement['id']==p.id]['tangent_force2'].values[0]



def annotate_dead_elements(model, disp_threshold = None):
    """iterate through all elements and remove the ones with large displacement
    """
    for e in model.elems.values():
        if e.type == 'ground' or e.type =='beam':
            continue
        disp_e = np.linalg.norm(np.asarray(e.displacement))
        if disp_e>disp_threshold:
            e.type = 'dead'
    for p in model.contps.values():
        if model.elems[p.cand].type == 'dead' or model.elems[p.anta].type == 'dead':
            p.cont_type.cohesion = -1
def remove_dead_elements(model):
    #remove dead elements
    model.elems = {k: v for k, v in model.elems.items() if v.type != 'dead'}
    model.contps = {k: v for k, v in model.contps.items() if v.cont_type.cohesion != -1}

def group_points_per_element(model):
    element_to_points = dict()
    for p in model.contps.values():
        if p.cand not in element_to_points.keys():
            element_to_points[p.cand] = [p.id]
        else:
            element_to_points[p.cand].append(p.id)
    return element_to_points

def identify_dead_elements(model,element_to_points,ds_threshold = 1):
    #element_damage_state = dict()
    nb_dead_elements = 0
    for e in model.elems.values():
        if e.type == 'ground' or e.type =='beam':
            continue
        damage_states_of_all_points_on_element = []
        damage_states_of_all_points_on_element_mode2 = []
        for p_id in element_to_points[e.id]:
            if p_id in model.contps.keys():
                #damage_state_magnitude = np.sqrt(model.contps[p_id].Ds**2+model.contps[p_id].Dt**2+model.contps[p_id].Dc**2)
                damage_states_of_all_points_on_element.append(model.contps[p_id].Dt)
                damage_states_of_all_points_on_element_mode2.append(model.contps[p_id].Ds)
        #element_damage_state[e.id] = np.mean(damage_states_of_all_points_on_element)
        if np.mean(damage_states_of_all_points_on_element)>=ds_threshold or \
            np.mean(damage_states_of_all_points_on_element_mode2)>=ds_threshold:
            e.type = 'dead'
            nb_dead_elements += 1
    for p in model.contps.values():
        if model.elems[p.cand].type == 'dead' or model.elems[p.anta].type == 'dead':
            p.cont_type.cohesion = -1
    return nb_dead_elements

def remove_stone(model):
    new_contps = dict()
    for key_p,p in model.contps.items():
        if model.elems[p.cand].type.startswith('stone') or model.elems[p.anta].type.startswith('stone'):
            continue
        new_contps[key_p] = p
    model.contps = new_contps
    new_elems = dict()
    for key_e,e in model.elems.items():
        if e.type.startswith('stone'):
            continue
        new_elems[key_e] = e
    model.elems = new_elems

def remove_mortar(model,proportion =0.2):
    # set seed
    np.random.seed(0)
    new_elems = dict()
    deleted_elems = []
    for key_e,e in model.elems.items():
        if e.type.startswith('mortar') and np.random.rand()<proportion:
            deleted_elems.append(key_e)
            continue
        new_elems[key_e] = e
    print("Propotion of deleted mortar elements",len(deleted_elems)/len(model.elems))
    model.elems = new_elems
    new_contps = dict()
    for key_p,p in model.contps.items():
        if p.cand in deleted_elems or p.anta in deleted_elems:
            continue
        new_contps[key_p] = p
    model.contps = new_contps

def recalculate_elasticity(model):
    new_contps = dict()
    for key_p,p in model.contps.items():
        thickness_p = p.thickness/2#40
        E_p = p.cont_type.E#300
        kn_p = E_p/thickness_p#7.5
        thickness_counterpart = model.contps[p.counterPoint].thickness/2#40
        E_counterpart = model.contps[p.counterPoint].cont_type.E#300
        kn_counterpart = E_counterpart/thickness_counterpart#7.5
        kn_new = kn_p*kn_counterpart/(kn_p+kn_counterpart)#7.5*7.5/(7.5+7.5)#=3.75
        E_new = kn_new*(thickness_p+thickness_counterpart)#3.75*80=300
        thickness_new = thickness_p+thickness_counterpart#80
        #print("E_new",E_new)
        # print("kn_new",kn_new)
        # print("kn_counterpart",kn_counterpart)
        # print("kn_p",kn_p)
        # print("thickness_p",thickness_p)
        # print("thickness_counterpart",thickness_counterpart)
        #a = input("continue?")
        new_contps[key_p] = model.contps[key_p]
        new_contps[key_p].cont_type.E = E_new
        new_contps[key_p].thickness = thickness_new

        # renew the counter point
        counter_point_id = p.counterPoint
        new_contps[counter_point_id] = model.contps[counter_point_id]
        new_contps[counter_point_id].cont_type.E = E_new
        new_contps[counter_point_id].thickness = thickness_new
    return new_contps
# def recalculate_elasticity_simplified(model):
#     new_contps = dict()
#     for key_p,p in model.contps.items():
#         thickness_p = p.thickness#40
#         E_p = p.cont_type.E#300
#         kn_p = E_p/thickness_p#7.5
#         thickness_counterpart = model.contps[p.counterPoint].thickness#40
#         E_counterpart = model.contps[p.counterPoint].cont_type.E#300
#         kn_counterpart = E_counterpart/thickness_counterpart#7.5
#         kn_new = kn_p*kn_counterpart/(kn_p+kn_counterpart)#7.5*7.5/(7.5+7.5)#=3.75
#         E_new = kn_new*(thickness_p+thickness_counterpart)##3.75*(40+40)#=300
#         thickness_new = thickness_p+thickness_counterpart#80
#         #print("E_new",E_new)
#         # print("kn_new",kn_new)
#         # print("kn_counterpart",kn_counterpart)
#         # print("kn_p",kn_p)
#         # print("thickness_p",thickness_p)
#         # print("thickness_counterpart",thickness_counterpart)
#         #a = input("continue?")
#         new_contps[key_p] = model.contps[key_p]
#         new_contps[key_p].cont_type.E = E_new
#         new_contps[key_p].thickness = thickness_new

#         # renew the counter point
#         counter_point_id = p.counterPoint
#         new_contps[counter_point_id] = model.contps[counter_point_id]
#         new_contps[counter_point_id].cont_type.E = E_new
#         new_contps[counter_point_id].thickness = thickness_new
#     return new_contps

def recalculate_lamda(model):
    for p in model.contps.values():
        # vector_point_to_element_center = np.asarray(model.elems[p.cand].center)-np.asarray(p.coor)
        # t1 = np.asarray(p.tangent1)
        # t2 = np.asarray(p.tangent2)
        # t_tangent = t1+t2
        # proj_dist_to_tangent = np.dot(vector_point_to_element_center,t_tangent)/np.linalg.norm(t_tangent)
        # thickness_t = 2*abs(proj_dist_to_tangent)
        # # print("ratio between thickness_t and thickness",thickness_t/p.thickness)
        # # a = input("continue?")
        # lamda = (thickness_t/p.thickness)*(1+p.cont_type.lamda)-1
        # p.cont_type.lamda = lamda
        p.cont_type.lamda = 0.5*(1+p.cont_type.lamda)-1


    
def convert_model_unit(model,force_convert = 1e-3,length_convert = 1e3):
    for p in model.contps.values():
        p.coor = [p.coor[0]*length_convert,p.coor[1]*length_convert,p.coor[2]*length_convert]
        p.cont_type.cohesion = (p.cont_type.cohesion*force_convert)/(length_convert**2)
        p.cont_type.fc = (p.cont_type.fc*force_convert)/(length_convert**2)
        p.cont_type.ft = (p.cont_type.ft*force_convert)/(length_convert**2)
        p.cont_type.E = (p.cont_type.E*force_convert)/(length_convert**2)
        p.cont_type.Gc = p.cont_type.Gc*force_convert/length_convert
        p.cont_type.Gf1 = p.cont_type.Gf1*force_convert/length_convert
        p.cont_type.Gf2 = p.cont_type.Gf2*force_convert/length_convert
        p.cont_type.uc_elastic = p.cont_type.uc_elastic*length_convert
        p.cont_type.uc_ultimate = p.cont_type.uc_ultimate*length_convert
        p.cont_type.ut_elastic = p.cont_type.ut_elastic*length_convert
        p.cont_type.ut_ultimate = p.cont_type.ut_ultimate*length_convert
        p.cont_type.us_elastic = p.cont_type.us_elastic*length_convert
        p.cont_type.us_ultimate = p.cont_type.us_ultimate*length_convert
        p.section_h = p.section_h*length_convert*length_convert
        p.thickness = p.thickness*length_convert
    for e in model.elems.values():
        e.center = [e.center[0]*length_convert,e.center[1]*length_convert,e.center[2]*length_convert]
        e.dl = [e.dl[0]*force_convert,e.dl[1]*force_convert,e.dl[2]*force_convert,\
                e.dl[3]*force_convert*length_convert,e.dl[4]*force_convert*length_convert,\
                    e.dl[5]*force_convert*length_convert]
        e.ll = [e.ll[0]*force_convert,e.ll[1]*force_convert,e.ll[2]*force_convert,\
                e.ll[3]*force_convert*length_convert,e.ll[4]*force_convert*length_convert,\
                    e.ll[5]*force_convert*length_convert]
    return model


def compute_fracture_energy(model, GF2_equ = "1"):
    for p in model.contps.values():
        # if model.elems[p.cand].type.startswith('stone') or model.elems[p.anta].type.startswith('stone'):
        #     p.cont_type.Gc = 32*p.cont_type.fc*10/(10+p.cont_type.fc*10)
        # else:
        #     p.cont_type.Gc = 32*p.cont_type.fc/(10+p.cont_type.fc)
        p.cont_type.Gc = 32*p.cont_type.fc/(10+p.cont_type.fc)
        
        # mode I
        if p.cont_type.ft>0:
            p.cont_type.Gf1 = 0.07*np.log(1+0.17*p.cont_type.fc)
        else:
            p.cont_type.Gf1 = 0

        # mode II
        if p.cont_type.cohesion>0:
            if GF2_equ == "1":
                p.cont_type.Gf2 = 0.1*p.cont_type.cohesion
            elif GF2_equ == "2":
                p.cont_type.Gf2 = 10*p.cont_type.Gf1
            else:
                raise ValueError("GF2_equ must be 1 or 2")
        else:
            p.cont_type.Gf2 = 0
    return model

def compute_fracture_distance(model):
    for p in model.contps.values():
        kn = p.cont_type.E*p.section_h/p.thickness
        p.cont_type.uc_elastic = p.cont_type.fc*p.section_h/kn
        p.cont_type.uc_ultimate = p.cont_type.uc_elastic+(2*p.cont_type.Gc/p.cont_type.fc)
        
        # mode I
        if p.cont_type.ft>0:
            p.cont_type.ut_elastic = p.cont_type.ft*p.section_h/kn
            p.cont_type.ut_ultimate = p.cont_type.ut_elastic+(2*p.cont_type.Gf1/p.cont_type.ft)
        else:
            p.cont_type.ut_elastic = 0
            p.cont_type.ut_ultimate = 0

        # mode II
        if p.cont_type.cohesion>0:
            kt = kn/(2*(1+p.cont_type.lamda))
            p.cont_type.us_elastic = p.cont_type.cohesion*p.section_h/kt
            p.cont_type.us_ultimate = p.cont_type.us_elastic+(2*p.cont_type.Gf2/p.cont_type.cohesion)
        else:
            p.cont_type.us_elastic = 0
            p.cont_type.us_ultimate = 0
    return model

# def compute_fractue_energy(model):
#     for p in model.contps.values():
#         # if model.elems[p.cand].type.startswith('stone') or model.elems[p.anta].type.startswith('stone'):
#         #     p.cont_type.Gc = 32*p.cont_type.fc*10/(10+p.cont_type.fc*10)
#         # else:
#         #     p.cont_type.Gc = 32*p.cont_type.fc/(10+p.cont_type.fc)
#         p.cont_type.Gc = 32*p.cont_type.fc/(10+p.cont_type.fc)
#         kn = p.cont_type.E*p.section_h/p.thickness
#         #kn_interface = p.cont_type.E/p.thickness
#         p.cont_type.uc_elastic = p.cont_type.fc*p.section_h/kn
#         p.cont_type.uc_ultimate = p.cont_type.uc_elastic+(2*p.cont_type.Gc/p.cont_type.fc)
        
#         # mode I
#         if p.cont_type.ft>0:
#             p.cont_type.Gf1 = 0.07*np.log(1+0.17*p.cont_type.fc)
#             p.cont_type.ut_elastic = p.cont_type.ft*p.section_h/kn
#             p.cont_type.ut_ultimate = p.cont_type.ut_elastic+(2*p.cont_type.Gf1/p.cont_type.ft)
#         else:
#             p.cont_type.Gf1 = 0
#             p.cont_type.ut_elastic = 0
#             p.cont_type.ut_ultimate = 0

#         # mode II
#         if p.cont_type.cohesion>0:
#             p.cont_type.Gf2 = 0.1*p.cont_type.cohesion
#             #p.cont_type.Gf2 = 10*p.cont_type.Gf1
#             kt = kn/(2*(1+p.cont_type.lamda))
#             p.cont_type.us_elastic = p.cont_type.cohesion*p.section_h/kt
#             p.cont_type.us_ultimate = p.cont_type.us_elastic+(2*p.cont_type.Gf2/p.cont_type.cohesion)
#         else:
#             p.cont_type.Gf2 = 0
#             p.cont_type.us_elastic = 0
#             p.cont_type.us_ultimate = 0
#     return model

def cal_us_elastic_ultimate(p):
    p.cont_type.Gf2 = (0.13*(p.normal_force*1000)/p.section_h+0.058)/1000#kN/mm
    kn = p.cont_type.E*p.section_h/p.thickness
    kt = kn/(2*(1+p.cont_type.lamda))
    max_tangent_force = p.stored_cohesion+p.cont_type.mu*p.normal_force/p.section_h
    p.cont_type.us_elastic = max_tangent_force/kt
    p.cont_type.us_ultimate = p.cont_type.us_elastic+(2*p.cont_type.Gf2/max_tangent_force)

def write_intermediate_model(model,result_dir, iteration):
    contact_points_file = result_dir+f"/point_iteration{iteration}.csv"
    with open(contact_points_file, mode='w+') as contact_points_file:
        contact_points_writer = csv.writer(contact_points_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        contact_points_writer.writerow(["id","x","y","z","nx","ny","nz","t1x","t1y","t1z","t2x","t2y","t2z",\
            "candidate_id","antagonist_id","section_area","contact_type","cohesion","mu","fc","ft","face_id",\
                "E","thickness","normal_force","tangent_force","tangent_force2","displacementx","displacementy",\
                    "displacementz","Gc","Gf1","Gf2","uc_elastic","uc_ultimate","ut_elastic","ut_ultimate",\
                        "us_elastic","us_ultimate","gap0","gap1","gap2","Dc","Dt","Ds"])
        for contact_point_id in model.contps.keys():
            contact_point = model.contps[contact_point_id]
            contact_points_writer.writerow([contact_point.id,contact_point.coor[0],contact_point.coor[1],contact_point.coor[2],\
                                        contact_point.normal[0],contact_point.normal[1],contact_point.normal[2],\
                                            contact_point.tangent1[0],contact_point.tangent1[1],contact_point.tangent1[2],\
                                                contact_point.tangent2[0],contact_point.tangent2[1],contact_point.tangent2[2],\
                                                    contact_point.cand,contact_point.anta,contact_point.section_h,\
                                                        contact_point.cont_type.type,contact_point.cont_type.cohesion,contact_point.cont_type.mu,\
                                                            contact_point.cont_type.fc,contact_point.cont_type.ft,\
                                                                contact_point.faceID,contact_point.cont_type.E,contact_point.thickness,\
                                                                    contact_point.normal_force,contact_point.tangent_force,contact_point.tangent_force2,\
                                                                        contact_point.displacement[0],contact_point.displacement[1],contact_point.displacement[2],\
                                                                            contact_point.cont_type.Gc,contact_point.cont_type.Gf1,contact_point.cont_type.Gf2,\
                                                                                contact_point.cont_type.uc_elastic,contact_point.cont_type.uc_ultimate,\
                                                                                    contact_point.cont_type.ut_elastic,contact_point.cont_type.ut_ultimate,\
                                                                                        contact_point.cont_type.us_elastic,contact_point.cont_type.us_ultimate,\
                                                                                            contact_point.gap[0],contact_point.gap[1],contact_point.gap[2],\
                                                                                                contact_point.Dc,contact_point.Dt,contact_point.Ds])
    elements_file = result_dir+f"/element_iteration{iteration}.csv"
    with open(elements_file, mode='w+') as elements_file:
        elements_writer = csv.writer(elements_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        elements_writer.writerow(["id","type","cx","cy","cz","mass","shape","dl_fx","dl_fy","dl_fz","dl_mx","dl_my","dl_mz","ll_fx","ll_fy","ll_fz","ll_mx","ll_my","ll_mz",\
                                  "displacementx","displacementy","displacementz","rotationx","rotationy","rotationz"])
        for element_id in model.elems.keys():
            element = model.elems[element_id]
            elements_writer.writerow([element.id,element.type,element.center[0],element.center[1],element.center[2],\
                    element.mass,element.shape_file,element.dl[0],element.dl[1],element.dl[2],element.dl[3],\
                        element.dl[4],element.dl[5],element.ll[0],element.ll[1],element.ll[2],element.ll[3],\
                            element.ll[4],element.ll[5],element.displacement[0],element.displacement[1],element.displacement[2],\
                                element.displacement[3],element.displacement[4],element.displacement[5]])

def initialize_contact_force(contps):
    for p in contps.values():
        p.normal_force = 0
        p.tangent_force = 0
        p.tangent_force2 = 0
        p.gap = [0, 0,0]
        p.stored_cohesion = p.cont_type.cohesion
        p.stored_fc = p.cont_type.fc
        p.stored_ft = p.cont_type.ft

def cal_gap_3d_noclosing(contps,elems):
    for key, value in contps.items():
        value.gap[0] = 0
        value.gap[1] = 0
        value.gap[2] = max(value.gap[2],cal_ptp_dist_normal_project(
            value.coor, contps[value.counterPoint].coor, \
                contps[value.counterPoint].normal,force_positive=True))


def cal_ptp_dist_normal_project(point1, point2, normal2,force_positive = True):
    """Calculate the distance from point1 to point2, projected to the normal of point 2

    :param point1: contact point 1
    :type point1: ContPoint
    :param point2: contact point 2
    :type point2: ContPoint
    :param normal2: normal direction of the face contact point 2 belongs to
    :type normal2: list
    :return: projected distance
    :rtype: float
    """
    vector_2_to_1 = np.asarray(point1)-np.asarray(point2)
    reversed_normal2 = -1*np.asarray(normal2)
    gap = np.dot(vector_2_to_1, reversed_normal2)
    if gap >= 0:
        return gap
    else:
        if force_positive:
            return 0
        else:
            return gap


def cal_gap_3d_elastic(contps,elems,thickness_dict=None):
    cal_gap_3d(contps,force_positive=False)

    for p in contps.values():
        thickness = p.thickness
        E = p.cont_type.E
        lamda = p.cont_type.lamda

        kjn = E/thickness#approximation
        #kjn = 1e-1
        kn = kjn*p.section_h
        #kn = 1e8#!debug
        kt = kn/(2*(1+lamda))
        Ckt = 1/kt
        p.gap[0] = -Ckt*p.tangent_force
        p.gap[1] = -Ckt*p.tangent_force2
        #p.gap[2] = 1/kn*p.normal_force
        #print(p.gap)

def cal_ptp_dist_tangent_project(point1,point2,normal2):
    vector_2_to_1 = np.asarray(point1)-np.asarray(point2)
    reversed_normal2 = -1*np.asarray(normal2)
    gap = np.dot(vector_2_to_1, reversed_normal2)
    gap_tangent = np.sqrt(np.linalg.norm(vector_2_to_1)**2-gap**2)
    if gap_tangent<0:
        print("Warning: gap_tangent<0")
    return gap_tangent

def adjust_ft_c(contps):
    for p in contps.values():
        gap_normal = p.gap[2]
        # print("gap_normal",gap_normal)
        # print("p.cont_type.ut_elastic",p.cont_type.ut_elastic)
        # print("p.cont_type.ut_ultimate",p.cont_type.ut_ultimate)
        gap_tangent = cal_ptp_dist_tangent_project(p.coor, contps[p.counterPoint].coor, contps[p.counterPoint].normal)
        kn = p.cont_type.E*p.section_h/p.thickness
        kt = kn/(2*(1+p.cont_type.lamda))
        # gap_tangent_contribution_of_normal_force = max(p.normal_force*p.cont_type.mu/kt,0)
        # gap_tangent = gap_tangent-gap_tangent_contribution_of_normal_force
        gap_tangent = gap_tangent
        if gap_normal>p.cont_type.ut_ultimate:
            p.cont_type.ft = 0
            p.Dt = 1
        elif gap_normal<=p.cont_type.ut_ultimate and gap_normal>p.cont_type.ut_elastic:
            #interpolate
            p.Dt = max(p.Dt,\
                (gap_normal-p.cont_type.ut_elastic)/(p.cont_type.ut_ultimate-p.cont_type.ut_elastic))
            # print("p.Dt",p.Dt)
            # a = input("continue?")
            p.cont_type.ft = p.stored_ft*\
                (1-p.Dt)
            #print("p.cont_type.ft",p.cont_type.ft)
        elif gap_normal<=p.cont_type.ut_elastic and gap_normal>=-p.cont_type.uc_elastic:
            p.Dt = max(p.Dt,\
                       0)
            p.cont_type.ft = p.stored_ft
        elif gap_normal<-p.cont_type.uc_elastic and gap_normal>=-p.cont_type.uc_ultimate:
            #interpolate
            p.Dc = max(p.Dc,\
                    (gap_normal+p.cont_type.uc_elastic)/(-p.cont_type.uc_ultimate+p.cont_type.uc_elastic))
            p.cont_type.fc = p.stored_fc*\
                (1-p.Dc)
        else:
            p.Dc = 1
            p.cont_type.fc = 0
        
        p.Ds = max(p.Ds,p.Dt)

        # cal_us_elastic_ultimate(p)
        if gap_tangent>p.cont_type.us_ultimate:
            p.Ds = max(1,p.Ds)
            #p.cont_type.cohesion = p.cont_type.cohesion*(1-p.Ds)
        elif gap_tangent<=p.cont_type.us_ultimate and gap_tangent>p.cont_type.us_elastic:
            #interpolate
            p.Ds = max(p.Ds, (gap_tangent-p.cont_type.us_elastic)/(p.cont_type.us_ultimate-p.cont_type.us_elastic))
            # p.cont_type.cohesion = p.cont_type.cohesion*\
            #     (1-p.Ds)
        else:
            p.Ds = max(p.Ds,0)
            #p.cont_type.cohesion = p.cont_type.cohesion*(1-p.Ds)
        p.cont_type.cohesion = p.stored_cohesion*(1-p.Ds)

        # if p.Ds<1:
        #     p.cont_type.mu = 10


def _update_contp_force_3d(contps, forces):
    for i, value in enumerate(contps.values()):
        value.normal_force = forces[i*7+2]
        value.tangent_force = forces[i*7]
        value.tangent_force2 = forces[i*7+1]

def _update_contp_force_3d_la(contps,forces):
    for i, value in enumerate(contps.values()):
        value.normal_force = forces[i*4+2]
        value.tangent_force = forces[i*4]
        value.tangent_force2 = forces[i*4+1]
    
def control_elem_disp_3d(contps, elems,step_size,control_index, control_dof):
    factor_on_disp = step_size/elems[control_index].displacement[control_dof]

def _displace_model_3d(elems, contps,zero_displacement=True):
    # update vertices information because the next step could fail
    # # element vertices
    # for key, value in elems.items():
    #     vertices = np.array(value.vertices)
    #     center = np.asarray(value.center)
    #     vertices_res_center = vertices-center
    #     rot_angles = np.asarray(value.displacement[3:])
    #     rotated_vertices_res_center = rotate_3d(
    #         vertices_res_center, rot_angles, order='xyz')
    #     disp_center = np.asarray(value.displacement[:3])
    #     new_vertices = rotated_vertices_res_center+disp_center+center
    #     value.vertices = new_vertices.tolist()

    for k, value in contps.items():
        disp_center = np.asarray(elems[value.cand].displacement[:3])
        center = np.asarray(elems[value.cand].center)
        point_coord = np.asarray(value.coor)
        point_coord_res_center = point_coord-center
        rot_angles = np.asarray(elems[value.cand].displacement[3:])
        rotated_point_coord_res_center = rotate_3d(
            np.expand_dims(point_coord_res_center, axis=0), rot_angles)[0]
        new_point_coord = rotated_point_coord_res_center+disp_center+center
        value.coor = new_point_coord.tolist()

    for key, value in elems.items():
        value.center[0] = value.center[0]+value.displacement[0]
        value.center[1] = value.center[1]+value.displacement[1]
        value.center[2] = value.center[2]+value.displacement[2]
    if zero_displacement:
        for k, value in contps.items():
            value.displacement = [0, 0, 0]
        for key, value in elems.items():
            value.displacement = [0, 0, 0, 0, 0, 0]


def _update_elem_disp_3d(contps, elems, disps):
    if len(disps) != 6*len(elems.values()):
        raise Exception(
            'Displacement list length does not match number of elements')
    disp_index = 0
    for key, value in elems.items():
        value.displacement = [disps[disp_index*6],
                              disps[disp_index*6+1], disps[disp_index *
                                                           6+2], disps[disp_index*6+3],
                              disps[disp_index*6+4], disps[disp_index*6+5]]
        disp_index += 1

    for k, value in contps.items():
        disp_center = np.asarray(elems[value.cand].displacement[:3])
        center = np.asarray(elems[value.cand].center)
        point_coord = np.asarray(value.coor)
        point_coord_res_center = point_coord-center
        rot_angles = 1 * \
            np.asarray(elems[value.cand].displacement[3:])  # !important
        rotated_point_coord_res_center = rotate_3d(
            np.expand_dims(point_coord_res_center, axis=0), rot_angles)[0]
        new_point_coord = rotated_point_coord_res_center+disp_center+center
        value.displacement = (new_point_coord-point_coord).tolist()


def rotate_3d(coords, thetas, order='xyz'):
    """Rotate a point cloud.

    Rotate a point cloud using Euler angles with specified rotation sequence
    around origin. Extrinsic and intrinsic rotations are supported.
    Rotation angle is positive if the rotation is counter-clockwise.

    Parameters
    ----------
    coords : (N, 3) array_like
        Points to be rotated.
    thetas : (3,) array_like
        Euler's rotation angles (in radians).
    order : string, optional
        Axis sequence for Euler angles. Up to 3 characters belonging to the set
        {'x', 'y', 'z'} for intrinsic rotations, or {'X', 'Y', 'Z'} for
        extrinsic rotations [default: 'xyz'].

    Returns
    -------
    (N, 3) ndarray
        Rotated points.

    """
    # Rotation matrix around unit x axis
    #thetas[0] = 0
    cos_thetas = np.cos(thetas)
    sin_thetas = np.sin(thetas)
    R_x = np.array([[1, 0, 0],
                    [0, cos_thetas[0], -sin_thetas[0]],
                    [0, sin_thetas[0], cos_thetas[0]]
                    ])

    # Rotation matrix around unit y axis
    #thetas[1] = 0
    R_y = np.array([[cos_thetas[1], 0, sin_thetas[1]],
                    [0, 1, 0],
                    [-sin_thetas[1], 0, cos_thetas[1]]
                    ])

    # Rotation matrix around unit z axis
    #thetas[2] = 0
    R_z = np.array([[cos_thetas[2], -sin_thetas[2], 0],
                    [sin_thetas[2], cos_thetas[2], 0],
                    [0, 0, 1]
                    ])

    # Combined rotation matrix
    if order == 'xyz':
        R = np.dot(R_z, np.dot(R_y, R_x))
    elif order == 'yzx':
        R = np.dot(R_x, np.dot(R_z, R_y))

    # Rotate point cloud
    return np.dot(coords, R)


def _update_contp_force_2d(contps, forces):
    for i, value in enumerate(contps.values()):
        value.normal_force = forces[i*2+1]
        value.tangent_force = forces[i*2]


def _update_contp_crack_2d(contps, elems, contfs, sliding_points):
    nb_crack_points = 0
    # detect crack based on faces\
    # # Reinitialize crack state
    # for k, value in contps.items():
    #     value.cont_type.cohesion = value.stored_cohesion
    #     value.cont_type.ft = value.stored_ft
    #     value.crack_state = False
    # for key, contact_face in contfs.items():
    #     point_locations_np = np.zeros((4, 2))
    #     for i in range(len(contact_face.contps)):
    #         point_locations_np[i] = np.asarray(
    #             contps[contact_face.contps[i]].coor)
    #     face_center = np.mean(point_locations_np, axis=0)
    #     total_moment = 0
    #     tension_force = 0

    #     for i, contact_point_id in enumerate(contact_face.contps):
    #         if i % 2 != 0:
    #             coeff = -1
    #         else:
    #             coeff = 1
    #         force = coeff*contps[contact_point_id].normal_force * \
    #             np.asarray(contps[contact_point_id].normal)
    #         lever_ = -np.asarray(contps[contact_point_id].coor)+face_center
    #         moment = np.cross(force, lever_)  # internal moment direction
    #         total_moment += moment
    #         tension_force -= contps[contact_point_id].normal_force
    #     anker_point = np.asarray(contps[contact_face.contps[0]].coor)
    #     anker_point_element_center = np.asarray(
    #         elems[contps[contact_face.contps[0]].cand].center)
    #     anker_vector = anker_point-anker_point_element_center
    #     for i, contact_point_id in enumerate(contact_face.contps):
    #         contact_point_to_center = np.asarray(
    #             contps[contact_point_id].coor)-anker_point_element_center
    #         if np.array_equal(contact_point_to_center, anker_vector):
    #             continue
    #         elif np.cross(contact_point_to_center, anker_vector) == 0:
    #             continue
    #         else:
    #             sign = np.sign(
    #                 np.cross(contact_point_to_center, anker_vector))

    #         if np.sign(total_moment) == 0:  # same compression/tension on the whole section
    #             tensionpoint_id = [contact_point_id,
    #                                contps[contact_point_id].counterPoint]
    #             compressionpoint_id = [
    #                 p for p in contact_face.contps if p not in tensionpoint_id]
    #         elif sign == np.sign(total_moment):
    #             # tension
    #             tensionpoint_id = [contact_point_id,
    #                                contps[contact_point_id].counterPoint]
    #             compressionpoint_id = [
    #                 p for p in contact_face.contps if p not in tensionpoint_id]
    #         elif sign != np.sign(total_moment):
    #             # compression
    #             compressionpoint_id = [contact_point_id,
    #                                    contps[contact_point_id].counterPoint]
    #             tensionpoint_id = [
    #                 p for p in contact_face.contps if p not in compressionpoint_id]
    #         if (6*abs(total_moment)/(contact_face.height**2))+tension_force/contact_face.height >= contact_face.ft or (6*abs(total_moment)/(contact_face.height**2))+tension_force/contact_face.height <= -contact_face.fc:
    #             for p_id in tensionpoint_id:
    #                 contps[p_id
    #                        ].cont_type.cohesion = contps[p_id].c0
    #                 contps[p_id].cont_type.ft = 0
    #                 if contps[p_id].crack_state == False:
    #                     nb_crack_points += 1
    #                 contps[p_id].crack_state = True

    #         if (-6*abs(total_moment)/(contact_face.height**2))+tension_force/contact_face.height >= contact_face.ft or (-6*abs(total_moment)/(contact_face.height**2))+tension_force/contact_face.height <= -contact_face.fc:
    #             for p_id in compressionpoint_id:
    #                 contps[p_id
    #                        ].cont_type.cohesion = contps[p_id].c0
    #                 contps[p_id].cont_type.ft = 0
    #                 if contps[p_id].crack_state == False:
    #                     nb_crack_points += 1
    #                 contps[p_id].crack_state = True
    #         break
    for point_key in sliding_points:
        contps[point_key].cont_type.cohesion = contps[point_key].c0
        contps[point_key].cont_type.ft = 0
        if contps[point_key].crack_state == False:
            nb_crack_points += 1
        contps[point_key].crack_state = True
    # if (6*total_moment/(contact_face.height**2))+tension_force/contact_face.height > contact_face.ft or (6*total_moment/(contact_face.height**2))+tension_force/contact_face.height < -contact_face.fc:
    #     # the min_moment_edge is cracked by tension
    #     # set to default cohesion
    #     contps[min_moment_id].cont_type.cohesion = value.c0
    #     contps[min_moment_id].cont_type.ft = 0

    #     # set to default cohesion
    #     contps[contps[min_moment_id].counterPoint].cont_type.cohesion = value.c0
    #     contps[contps[min_moment_id].counterPoint].cont_type.ft = 0

    #     nb_crack_points += 2
    # if (-6*total_moment/(contact_face.height**2))+tension_force/contact_face.height > contact_face.ft or (-6*total_moment/(contact_face.height**2))+tension_force/contact_face.height < -contact_face.fc:
    #     # set to default cohesion
    #     contps[max_moment_id].cont_type.cohesion = value.c0
    #     contps[max_moment_id].cont_type.ft = 0

    #     # set to default cohesion
    #     contps[contps[max_moment_id].counterPoint].cont_type.cohesion = value.c0
    #     contps[contps[max_moment_id].counterPoint].cont_type.ft = 0
    # #     nb_crack_points += 2

    # for k, value in contps.items():
    #     value.cont_type.cohesion = value.stored_cohesion
    #     value.cont_type.ft = value.stored_ft
    #     value.crack_state = False
    #     if value.normal_force*value.cont_type.mu+value.cont_type.cohesion < abs(value.tangent_force) or value.normal_force < -0.5*value.cont_type.ft*value.section_h:
    #         value.cont_type.cohesion = value.c0  # set to default cohesion
    #         value.cont_type.ft = 0
    #         value.crack_state = True
    #         nb_crack_points += 1
    print(f"{nb_crack_points} points cracked")
    # # a point is cracked if there's displacement between it and its counterpoint
    # nb_crack_points = 0
    # for k, value in contps.items():
    #     value.cont_type.cohesion = value.stored_cohesion
    #     value.cont_type.ft = value.stored_ft
    #     if np.sum((np.array(value.displacement)-np.array(contps[value.counterPoint].displacement))**2) > _crack_tolerance:
    #         value.cont_type.cohesion = value.c0  # set to default cohesion
    #         value.cont_type.ft = 0
    #         nb_crack_points += 1
    # a point is cracked if there's displacement between it and its counterpoint


def _line1(sigma_c, sigma_t, h_j):
    a = (3*h_j ** 2*(sigma_c + sigma_t))/(8*(h_j*sigma_c + h_j*sigma_t))
    c = (3*h_j ** 2*sigma_t)/8
    return (-a, -c)


def _line2(sigma_c, sigma_t, h_j):
    a = (h_j ** 2*(sigma_c + sigma_t))/(8*(h_j*sigma_c + h_j*sigma_t))
    c = (3*h_j ** 2*(sigma_c - sigma_t))/16 - \
        (h_j ** 2*(sigma_c - 3*sigma_t))/8
    return (-a, -c)


def _line3(sigma_c, sigma_t, h_j):
    a = -(h_j ** 2*(sigma_c + sigma_t))/(8*(h_j*sigma_c + h_j*sigma_t))
    c = (h_j ** 2*(3*sigma_c - sigma_t))/8 - \
        (3*h_j ** 2*(sigma_c - sigma_t))/16
    return (-a, -c)


def _line4(sigma_c, sigma_t, h_j):
    a = -(3*h_j ** 2*(sigma_c + sigma_t))/(8*(h_j*sigma_c + h_j*sigma_t))
    c = (3*h_j ** 2*sigma_c)/8
    return (-a, -c)
