import logging
import re
import bs4
from typing import Any, Dict, List, Tuple, TextIO
from pipeline.grounding.validation_tools import extract_mapping_details
from pipeline.utils import (make_threeletter_code,
                            make_ori_mutant_threeletter,
                            make_ori_mutant,
                            split_mutant_and_chain,
                            split_res_and_chain)

# define search patterns for "residue_name_number" and "mutant"
# text patterns to catch bad tokenization and additional characters
pattern1 = r"[A-Z]{3}[0-9]+[A-Z]+"
pattern2 = r"[A-Z]{1}[0-9]+[A-Z]+"
# pattern for 3-letter point mutation
pattern3 = r"[A-Z]{3}[0-9]+[A-Z]{3}$"
# pattern for 1-letter point mutation
pattern4 = r"[A-Z]{1}[0-9]+[A-Z]{1}$"
# pattern for 1-letter "residue_name_number"
pattern5 = r"[A-Z]{1}[0-9]+$"
# pattern for 3-letter "residue_name_number"
pattern6 = r"[A-Z]{3}[0-9]+$"
# pattern for 3-letter "residue_name_number"
# with 4th letter representing chain ID
pattern7a = r"[A-Z]{4}[0-9]+$"
pattern7b = r"[A-Z]{3}[0-9]+[A-Z]{1}$"
# mutation with chain identifier
pattern8 = r"[A-Z]{4}[0-9]+[A-Z]{3}$"


def make_clash_list(residue: bs4.element.Tag) -> List[Dict[str, Any]]:
    """
    Function to extract a clash list from validation XML file

    Input
    :param residue: XML tag containing validation stats for a specific residue
    :type residue: bs4.element.Tag

    Output
    :return: clash_list; a list of clashes found for a residue
    :rtype: List[Dict[str, Any]]
    """
    clash_list = []
    try:
        clashes = residue.find_all("clash")
        for c in clashes:
            clash_id = c["cid"]
            atom = c["atom"]
            clashmag = c["clashmag"]
            dist = c["dist"]
            clash_dict = {"clash_id" : clash_id,
                            "atom" : atom,
                            "clashmag" : clashmag,
                            "dist" : dist}
            clash_list.append(clash_dict)
    except:
        clash_list = []

    return clash_list

def get_phi(residue: bs4.element.Tag) -> str:
    """
    Function to extract a phi angle from validation XML file

    Input
    :param residue: XML tag containing validation stats for a specific residue
    :type residue: bs4.element.Tag

    Output
    :return: phi; phi value found for a residue
    :rtype: str
    """
    try:
        phi = residue["phi"]
    except:
        phi = ""

    return phi

def get_psi(residue: bs4.element.Tag) -> str:
    """
    Function to extract a psi angle from validation XML file

    Input
    :param residue: XML tag containing validation stats for a specific residue
    :type residue: bs4.element.Tag

    Output
    :return: psi; psi value found for a residue
    :rtype: str
    """
    try:
        psi = residue["psi"]
    except:
        psi = ""

    return psi

def get_rama(residue: bs4.element.Tag) -> str:
    """
    Function to extract a Ramachandran score from validation XML file

    Input
    :param residue: XML tag containing validation stats for a specific residue
    :type residue: bs4.element.Tag

    Output
    :return: rama; rama value found for a residue
    :rtype: str
    """
    try:
        rama = residue["rama"]
    except:
        rama = ""

    return rama

def get_res_name(residue: bs4.element.Tag) -> str:
    """
    Function to extract a residue name from validation XML file

    Input
    :param residue: XML tag containing validation stats for a specific residue
    :type residue: bs4.element.Tag

    Output
    :return: res_name; resname value found for a residue
    :rtype: str
    """
    try:
        res_name = residue["resname"]
    except:
        res_name = ""

    return res_name

def get_res_num(residue: bs4.element.Tag) -> str:
    """
    Function to extract author-defined residue number from validation XML file

    Input
    :param residue: XML tag containing validation stats for a specific residue
    :type residue: bs4.element.Tag

    Output
    :return: res_num; resnum value found for a residue
    :rtype: str
    """
    try:
        res_num = residue["resnum"]
    except:
        res_num = ""

    return res_num

def get_res_full_name(residue: bs4.element.Tag) -> str:
    """
    Function to create a full residue name of name and number from validation
    XML file

    Input
    :param residue: XML tag containing validation stats for a specific residue
    :type residue: bs4.element.Tag

    Output
    :return: res_name_full; full residue name of resname and resnum
    :rtype: str
    """
    try:
        res_name_full = residue["resname"] + residue["resnum"]
    except:
        res_name_full = ""

    return res_name_full
        
def get_res_seq(residue: bs4.element.Tag) -> str:
    """
    Function to extract structure sequence number from validation
    XML file

    Input
    :param residue: XML tag containing validation stats for a specific residue
    :type residue: bs4.element.Tag

    Output
    :return: res_seq; structure sequence number for residue
    :rtype: str
    """
    try:
        res_seq = residue["seq"]
    except:
        res_seq = ""

    return res_seq

def get_res_chain(residue: bs4.element.Tag) -> str:
    """
    Function to extract chain from validation XML file

    Input
    :param residue: XML tag containing validation stats for a specific residue
    :type residue: bs4.element.Tag

    Output
    :return: chain; chain for residue
    :rtype: str
    """
    try:
        chain = residue["said"]
    except:
        chain = ""

    return chain

def get_rotamer(residue: bs4.element.Tag) -> str:
    """
    Function to extract a rotamer from validation XML file

    Input
    :param residue: XML tag containing validation stats for a specific residue
    :type residue: bs4.element.Tag

    Output
    :return: rotamer; rotamer for residue
    :rtype: str
    """
    try:
        rotamer = residue["rota"]
    except:
        rotamer = ""

    return rotamer

def get_alt_conf(residue: bs4.element.Tag) -> str:
    """
    Function to extract alternative conformations from validation XML file

    Input
    :param residue: XML tag containing validation stats for a specific residue
    :type residue: bs4.element.Tag

    Output
    :return: alt_conf; alternative conformations for residue
    :rtype: str
    """
    try:
        alt_conf = residue["altcode"]
    except:
        alt_conf = ""

    return alt_conf

def get_rscc(residue: bs4.element.Tag) -> str:
    """
    Function to extract RSCC from validation XML file

    Input
    :param residue: XML tag containing validation stats for a specific residue
    :type residue: bs4.element.Tag

    Output
    :return: rscc; real-space correlation coefficient, rscc, for residue
    :rtype: str
    """
    try:
        rscc = residue["rscc"]
    except:
        rscc = ""
    
    return rscc

def get_q_score(residue: bs4.element.Tag) -> str:
    """
    Function to extract Q_score from validation XML file

    Input
    :param residue: XML tag containing validation stats for a specific residue
    :type residue: bs4.element.Tag

    Output
    :return: q_score; Q_score for residue
    :rtype: str
    """
    try:
        q_score = residue["Q_score"]
    except:
        q_score = ""

    return q_score

def make_dist_outlier_list(residue: bs4.element.Tag) -> str:
    """
    Function to extract distance violations from validation XML file

    Input
    :param residue: XML tag containing validation stats for a specific residue
    :type residue: bs4.element.Tag

    Output
    :return: dist_list; list of distance violations for residue
    :rtype: List[Dict[str, Any]]
    """
    dist_list = []
    try:
        dist_viol = residue.find_all("distance_violation")
        for d in dist_viol:
            dist_id = d["rlist_id"]
            rest_id = d["rest_id"]
            dist_viol_value = d["dist_violation_value"]
            atom = d["atom"]
            dist_dict = {"dist_id" : dist_id,
                            "rest_id" : rest_id,
                            "dist_viol_value" : dist_viol_value,
                            "atom" : atom}
            dist_list.append(dist_dict)
    except:
        dist_list = []
    
    return dist_list

def make_angle_outlier_list(residue: bs4.element.Tag) -> str:
    """
    Function to extract angle violations from validation XML file

    Input
    :param residue: XML tag containing validation stats for a specific residue
    :type residue: bs4.element.Tag

    Output
    :return: angle_list; list of angle violations for residue
    :rtype: List[Dict[str, Any]]
    """
    angle_list = []
    try:
        angle_viol = residue.find_all("dihedralangle_violation")
        for d in angle_viol:
            angle_id = d["rlist_id"]
            rest_id = d["rest_id"]
            angle_viol_value = d["DihedralAngViolationValue"]
            atom = d["atom"]
            dist_dict = {"angle_id" : angle_id,
                            "rest_id" : rest_id,
                            "dist_viol_value" : angle_viol_value,
                            "atom" : atom}
            angle_list.append(dist_dict)
    except:
        angle_list = []
    
    return angle_list

def make_base_dict(residue: bs4.element.Tag) -> Dict[str, Any]:
    """
    Function to create a base dictionary for validation stats for a specific
    residue.
    
    Input
    :param residue: XML object containing validation stats for a specific
    residue
    :type residue: bs4.element.Tag

    Output
    :return: res_tag_dict; dictionary containing basic validation stats for
    selected residue
    :rtype: Dict[str, Any]
    """
    clash_list = make_clash_list(residue)
    res_name = get_res_full_name(residue)
    phi = get_phi(residue)
    psi = get_psi(residue)
    rama = get_rama(residue)
    res_name = get_res_name(residue)
    res_num = get_res_num(residue)
    res_seq = get_res_seq(residue)
    chain = get_res_chain(residue)
    rota = get_rotamer(residue)
    alt_conf = get_alt_conf(residue)

    res_tag_dict = {
                    "pdb_res_name" : res_name,
                    "pdb_res_number" : res_num,
                    "pdb_res_seq" : res_seq,
                    "pdb_res" : res_name,
                    "pdb_chain" : chain,
                    "ramachandran" : rama,
                    "rotamer" : rota,
                    "phi" : phi,
                    "psi" : psi,
                    "clashes" : clash_list,
                    "altconf" : alt_conf,
                    }

    return res_tag_dict

def get_stats_from_em(residues: List[bs4.element.Tag],
                      pdb_id: str,
                      wildtype_res: str
                      ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Function to extract validation statisitcs for a structure determined using
    cryoEM. An XML object for a structures validation XML is searched for a
    residue match with either the wildtype residue or the mutant residue.
    The residue itself is the entity that was annotated and extracted from
    the literature text. A PDB ID is required to be able to keep track of the
    returned dictionary containing the residue specific validation stats. If
    the entity type is 'residue_name_number' but upon closer analysis is
    actually of type 'mutant' then the entity type is updated.

    Input

    :param residues: XML object of a validation XML file to iterate over for
                     extraction of residue specific validation stats
    :type residues: List[bs4.element.Tag]

    :param pdb_id: PDB ID of the corresponding structure to the current
                   validation XML file
    :type pdb_id: str

    :param wildtype_res: the wildtype residue identified from the annotation
    :type wildtype_res: str

    :param mutant_res: the mutant residue identified from the annotation
    :type mutant_res: str

    :param res_name_num: default is 'False'; used to identify entities that
                         were predicted as 'mutant' but are actually
                         'residue_name_number' and annotation needs to be
                         updated
    :type res_name_num: bool


    Output

    :return: res_tag_list; list of dictionaries for the residue with the
             validation statistics for this residue for each chain in the
             validation XML file
    :rtype: List[Dict[str, Any]]

    :return: mini_dict; dictionary containing prediction metrics for the
             annotation; updated from the original if entity type was changes
    :rtype: Dict[str, Any]

    """
    res_tag_list = []

    for residue in residues:
        res_tag_dict = make_base_dict(residue)
        q_score = get_q_score(residue)
        res_tag_dict["pdb_id"] = pdb_id
        res_tag_dict["q_score"] = q_score
        res_tag_dict["wildtype_pdb_res"] = wildtype_res
        res_tag_list.append(res_tag_dict)

    return res_tag_list

def get_stats_from_nmr(residues: List[bs4.element.Tag],
                      pdb_id: str,
                      wildtype_res: str
                      ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Function to extract validation statisitcs for a structure determined using
    NMR. An XML object for a structures validation XML is searched for a
    residue match with either the wildtype residue or the mutant residue.
    The residue itself is the entity that was annotated and extracted from
    the literature text. A PDB ID is required to be able to keep track of the
    returned dictionary containing the residue specific validation stats. If
    the entity type is 'residue_name_number' but upon closer analysis is actually of type
    'mutant' then the entity type is updated.

    Input

    :param residues: XML object of a validation XML file to iterate over for
                     extraction of residue specific validation stats
    :type residues: List[bs4.element.Tag]

    :param pdb_id: PDB ID of the corresponding structure to the current
                   validation XML file
    :type pdb_id: str

    :param wildtype_res: the wildtype residue identified from the annotation
    :type wildtype_res: str

    :param mutant_res: the mutant residue identified from the annotation
    :type mutant_res: str

    :param res_name_num: default is 'False'; used to identify entities that
                         were predicted as 'mutant' but are actually
                         'residue_name_number' and annotation needs to be
                         updated
    :type res_name_num: bool


    Output

    :return: res_tag_list; list of dictionaries for the residue with the
             validation statistics for this residue for each chain in the
             validation XML file
    :rtype: List[Dict[str, Any]]

    :return: mini_dict; dictionary containing prediction metrics for the
             annotation; updated from the original if entity type was changes
    :rtype: Dict[str, Any]

    """
    res_tag_list = []

    for residue in residues:
        dist_list = make_dist_outlier_list(residue)
        angle_list = make_angle_outlier_list(residue)
        res_tag_dict = make_base_dict(residue)
        res_tag_dict["pdb_id"] = pdb_id
        res_tag_dict["distances"] = dist_list
        res_tag_dict["angles"] = angle_list
        res_tag_dict["wildtype_pdb_res"] = wildtype_res
        res_tag_list.append(res_tag_dict)

    return res_tag_list

def get_stats_from_xray(residues: List[bs4.element.Tag],
                      pdb_id: str,
                      wildtype_res: str,
                      ) -> List[Dict[str, Any]]:
    """
    Function to extract validation statisitcs for a structure determined using
    X-ray diffraction. An XML object for a structures validation XML is searched for a
    residue match with either the wildtype residue or the mutant residue.
    The residue itself is the entity that was annotated and extracted from
    the literature text. A PDB ID is required to be able to keep track of the
    returned dictionary containing the residue specific validation stats. If
    the entity type is 'residue_name_number' but upon closer analysis is actually of type
    'mutant' then the entity type is updated.

    Input

    :param residues: XML object of a validation XML file to iterate over for
                     extraction of residue specific validation stats
    :type residues: List[bs4.element.Tag]

    :param pdb_id: PDB ID of the corresponding structure to the current
                   validation XML file
    :type pdb_id: str

    :param wildtype_res: the wildtype residue identified from the annotation
    :type wildtype_res: str

    :param mutant_res: the mutant residue identified from the annotation
    :type mutant_res: str

    :param res_name_num: default is 'False'; used to identify entities that
                         were predicted as 'mutant' but are actually
                         'residue_name_number' and annotation needs to be
                         updated
    :type res_name_num: bool


    Output

    :return: res_tag_list; list of dictionaries for the residue with the
             validation statistics for this residue for each chain in the
             validation XML file
    :rtype: List[Dict[str, Any]]

    :return: mini_dict; dictionary containing prediction metrics for the
             annotation; updated from the original if entity type was changes
    :rtype: Dict[str, Any]

    """
    res_tag_list = []
    for residue in residues:
        res_tag_dict = make_base_dict(residue)
        rscc = get_rscc(residue)
        res_tag_dict["pdb_id"] = pdb_id
        res_tag_dict["rscc"] = rscc
        res_tag_dict["wildtype_pdb_res"] = wildtype_res
        res_tag_list.append(res_tag_dict)

    return res_tag_list

def get_res_match_items(pattern6: str, res: str) -> List[str]:
    """
    Function to check that a residue follows a 3-letter code with sequence
    position number standard.

    Input
    :param pattern6: regular expression pattern representing a protein residue
                     in 3-letter notation with sequence position
    :type pattern6: str

    :param res: residue to be validated and mapped
    :type res: str

    Output
    :return: res_items; a list of items with item '0' representing the
             amino acid name and item '1' being the sequence number
    :rtype: List[str]
    """
    assert re.match(pattern6, res)
    res_match = re.match(r"([a-z]+)([0-9]+)", res, re.I)
    if res_match:
        res_items = res_match.groups()

    return res_items

def find_residues_in_val_file(val_data: bs4.BeautifulSoup,
                              res_items: List[str]) -> List[bs4.element.Tag]:
    """
    Function to find a specific residue in a validation XML file.

    Input
    :param val_data: validation XML content
    :type val_data: bs4.BeautifulSoup

    :param res_items: a list of items with item '0' representing the
                      amino acid name and item '1' being the sequence number
    :type res_items: List[str]

    Output
    :return: res_residues; list of residues found for combination of residue
             name (amino acid name) and sequence number
    :rtype: List[bs4.element.Tag]
    """
    res_residues = val_data.find_all("ModelledSubgroup",
                                     attrs = {"resnum" : res_items[1],
                                              "resname" : res_items[0],
                                              "model" : "1"})
    
    return res_residues

# can't currently test that as I have no example where the prediction gets
# residue_name_number and mutant mixed up
# def check_anno_type():
#     mini_dict = {}
#     mini_dict["annotator"] = "post_processing"
#     mini_dict["score"] = "removed"
#     mini_dict["type"] = "mutant"
#     return mini_dict
    

def per_res_validation(soup: TextIO, text: str
                       ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Using an XML object of a validation XML file identify the lines that
    contain validation statistics for the residue found as entity in the
    text; collect those lines and specific metrics depending on experiment
    type used to determine the structure. Return a dictionary with an
    updated entity type if 'residue_name'number' turns out to be 'mutant'
    and also return a list of dictionaries containing selected metrics from
    the validation file for this residue

    Input

    :param soup: validation XML content
    :type soup: TextIO

    :param text: text span for entity
    :type text: str


    Output

    :return: res_tag_list; list of dictionaries for the residue with the
             validation statistics for this residue for each chain in the
             validation XML file
    :rtype: List[Dict[str, Any]

    :return: mini_dict; dictionary containing prediction metrics for the
             annotation; updated from the original if entity type was changes
    :rtype: Dict[str, Any]

    """
    val_data = bs4.BeautifulSoup(soup, "xml")
    entry = val_data.find("Entry")
    pdb_id = entry["pdbid"]
    percentbins = entry["percentilebins"]
    percentbins_split = percentbins.split(",")

    wildtype_res = ""
    mutant_res = ""
    res_tag_list = []

    if (text.isalnum() == False
        or re.match(pattern7a, text)
        or re.match(pattern7b, text)):
        try:
            text = split_res_and_chain(text)
        except:
            pass

    if re.match(pattern6, text):
        res = text

    elif re.match(pattern5, text):
        res = make_threeletter_code(text)

    # check if residue_name_number is actually of type mutant
    elif re.search(pattern1, text) or re.search(pattern2, text):
        if re.findall(r"[A-Z]{3}[0-9]+[A-Z]{3}", text):
            threeletter_mutant = re.findall(r"[A-Z]{3}[0-9]+[A-Z]{3}", text)
            text = threeletter_mutant[0]
            (wildtype_res,
             mutant_res) = make_ori_mutant(text)
            # print("HURRA HURRA HURRA HURRA HURRA HURRA ")
            # print("HURRA HURRA HURRA HURRA HURRA HURRA ")
            # mini_dict = check_anno_type()
            # print(mini_dict)

    if res != "":
        try:
            res_items = get_res_match_items(pattern6, res)
            res_residues = find_residues_in_val_file(val_data, res_items)
            if len(res_residues) > 0 and "em" in percentbins_split:
                res_tag_list = get_stats_from_em(res_residues,
                                                 pdb_id,
                                                 wildtype_res)
                return res_tag_list
            elif len(res_residues) > 0 and "nmr" in percentbins_split:
                res_tag_list = get_stats_from_nmr(res_residues,
                                                  pdb_id,
                                                  wildtype_res)
                return res_tag_list
            elif len(res_residues) > 0 and "xray" in percentbins_split:
                res_tag_list = get_stats_from_xray(res_residues,
                                                   pdb_id,
                                                   wildtype_res)
                return res_tag_list
        except:
            logging.error(
    f"""Didn't find residue {res} in validation file for PDB entry {pdb_id}""")
    
    if wildtype_res != "":
        try:
            wild_items = get_res_match_items(pattern6, wildtype_res)
            wild_residues = find_residues_in_val_file(val_data, wild_items)

            if len(wild_residues) > 0 and "em" in percentbins_split:
                res_tag_list = get_stats_from_em(wild_residues,
                                                 pdb_id,
                                                 wildtype_res)
                return res_tag_list
            elif len(wild_residues) > 0 and "nmr" in percentbins_split:
                res_tag_list = get_stats_from_nmr(wild_residues,
                                                  pdb_id,
                                                  wildtype_res)
                return res_tag_list
            elif len(wild_residues) > 0 and "xray" in percentbins_split:
                res_tag_list = get_stats_from_xray(wild_residues,
                                                   pdb_id,
                                                   wildtype_res)
                return res_tag_list
        except:
            logging.error(
    f"""Didn't find wildtype residue {wildtype_res} from mutation in validation file for PDB entry {pdb_id}""")
            
    if mutant_res != "":
        try:
            mutant_items = get_res_match_items(pattern6, mutant_res)
            mutant_residues = find_residues_in_val_file(val_data, mutant_items)

            if len(mutant_residues) > 0 and "em" in percentbins_split:
                res_tag_list = get_stats_from_em(wild_residues,
                                                 pdb_id,
                                                 wildtype_res)
                return res_tag_list
            elif len(mutant_residues) > 0 and "nmr" in percentbins_split:
                res_tag_list = get_stats_from_nmr(wild_residues,
                                                  pdb_id,
                                                  wildtype_res)
                return res_tag_list
            elif len(mutant_residues) > 0 and "xray" in percentbins_split:
                res_tag_list = get_stats_from_xray(wild_residues,
                                                   pdb_id,
                                                   wildtype_res)
                return res_tag_list
        except:
            logging.error(
    f"""Didn't find mutant residue {mutant_res} from mutation in validation file for PDB entry {pdb_id}""")


def mutant_validation(soup: bs4.BeautifulSoup, text: str
                      ) -> List[Dict[str, Any]]:
    """
    Using an XML object of a validation XML file identify the lines that
    contain validation statistics for the residue found as entity in the
    text; collect those lines and specific metrics depending on experiment
    type used to determine the structure. return a list of dictionaries
    containing selected metrics from the validation file for this residue

    Input

    :param soup: validation XML content
    :type soup: bs4.BeautifulSoup

    :param text: text span for entity
    :type text: str


    Output

    :return: res_tag_list; list of dictionaries for the residue with the
             validation statistics for this residue for each chain in the
             validation XML file
    :rtype: List[Dict[str, Any]]

    """
    val_data = bs4.BeautifulSoup(soup, "xml")
    entry = val_data.find("Entry")
    pdb_id = entry["pdbid"]
    percentbins = entry["percentilebins"]
    percentbins_split = percentbins.split(",")

    threeletter_mutant = ""
    m_wildtype_res = ""
    m_mutant_res = ""
    res_tag_list = []

    try:
        if (re.match(pattern1, text) or
            re.match(pattern2, text) or
            re.match(pattern8, text)):
            threeletter_mutant = re.findall(pattern3, text)
            oneletter_mutant = re.findall(pattern4, text)
        if threeletter_mutant != []:
            three_res_text = threeletter_mutant[0]
            (m_wildtype_res,
             m_mutant_res) = make_ori_mutant(three_res_text)
        if oneletter_mutant != []:
            one_res_text = oneletter_mutant[0]
            (m_wildtype_res,
             m_mutant_res) = make_ori_mutant_threeletter(one_res_text)
        if re.match(pattern8, text):
            (m_wildtype_res,
             m_mutant_res) = split_mutant_and_chain(text)
            
    except:
        res_tag_list = []
        return res_tag_list

    try:
        items_m_wildtype = get_res_match_items(pattern6, m_wildtype_res)
        wild_residues = find_residues_in_val_file(val_data, items_m_wildtype)

        if len(wild_residues) > 0 and "em" in percentbins_split:
            res_tag_list = get_stats_from_em(wild_residues,
                                             pdb_id,
                                             m_wildtype_res)
            return res_tag_list
        elif len(wild_residues) > 0 and "nmr" in percentbins_split:
            res_tag_list = get_stats_from_nmr(wild_residues,
                                              pdb_id,
                                              m_wildtype_res)
            return res_tag_list
        elif len(wild_residues) > 0 and "xray" in percentbins_split:
            res_tag_list = get_stats_from_xray(wild_residues,
                                               pdb_id,
                                               m_wildtype_res)
            return res_tag_list
    except:
        logging.error(
f"""Didn't find wildtype residue {m_wildtype_res} from mutation in validation file for PDB entry {pdb_id}""")

    try:
        items_m_mutant = get_res_match_items(pattern6, m_mutant_res)
        mutant_residues = find_residues_in_val_file(val_data, items_m_mutant)

        if len(mutant_residues) > 0 and "em" in percentbins_split:
            res_tag_list = get_stats_from_em(mutant_residues,
                                             pdb_id,
                                             m_wildtype_res)
            return res_tag_list
        elif len(mutant_residues) > 0 and "nmr" in percentbins_split:
            res_tag_list = get_stats_from_nmr(mutant_residues,
                                              pdb_id,
                                              m_wildtype_res)
            return res_tag_list
        elif len(mutant_residues) > 0 and "xray" in percentbins_split:
            res_tag_list = get_stats_from_xray(mutant_residues,
                                               pdb_id,
                                                m_wildtype_res)
            return res_tag_list
    except:
        logging.error(
f"""Didn't find mutant residue {m_mutant_res} from mutation in validation file for PDB entry {pdb_id}""")


def per_res_validation_mapping(map_data: TextIO, tag: Dict[str, Any]
                               ) -> Dict[str, Any]:
    """
    Using the content of the SIFTS mapping file to call a mapping function
    and check for all the found validation statistics for a particular
    residue from the text whether we can ground this residue by creating
    a link to a UniProt reference structure

    Input

    :param map_data: SIFTS XML content
    :type map_data: TextIO

    :param tag: dictionary containing validated residue
    :type tag: Dict[str, Any]


    Output

    :return: tag; dictionary containing validated and mapped residue for an
             annotation
    :rtype: Dict[str, Any]

    """
    pdb_res_name = tag["pdb_res_name"]
    pdb_seq = tag["pdb_res_seq"]

    (map_uni_name_three,
    map_uni_num,
    ref_uniprot_acc,
    ref_uniprot_acc_species) = extract_mapping_details(map_data,
                                                        pdb_res_name,
                                                        pdb_seq)
    if ref_uniprot_acc_species == "":
        uniprot_uri = "https://www.uniprot.org/uniprot/" + ref_uniprot_acc
    else:
        uniprot_uri = "https://www.uniprot.org/uniprot/" + ref_uniprot_acc_species
    uniprot_res = map_uni_name_three + str(map_uni_num)
    tag["uniprot_id"] = ref_uniprot_acc
    tag["uniprot_name"] = ref_uniprot_acc_species
    tag["uniprot_res"] = uniprot_res
    tag["uniprot_uri"] = uniprot_uri

    return tag