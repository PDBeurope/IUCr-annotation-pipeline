import bs4
from typing import Tuple, TextIO
from pipeline.utils import make_threeletter_code

def get_uniprot_crossref_details(residue: bs4.element.Tag
                                 ) -> Tuple[str, str, str, str]:
    """"Get details for mapped UniProt residue from cross-reference database.
    
    Input
    :params residue: XML object of the publication
    :type residue: bs4.element.Tag

    Output
    :return: map_uni_num; mapped UniProt residue number
    :rtype: str
    :return: ref_uniprot_acc; mapped UniProt accession
    :rtype: str
    :return: map_uni_name_three; mapped UniProt residue in 3-letter code
    :rtype: str
    """
    uniprot_crossref = residue.find("crossRefDb",
                                    {"dbSource" : "UniProt"})
    map_uni_num = uniprot_crossref["dbResNum"]
    map_uni_name = uniprot_crossref["dbResName"]
    ref_uniprot_acc = uniprot_crossref["dbAccessionId"]
    map_uni_name_three = make_threeletter_code(map_uni_name)
    
    return map_uni_num, ref_uniprot_acc, map_uni_name_three

def get_secondary_uniprot_acc(map_regions: bs4.element.Tag) -> str:
    """
    Getting secondary UniProt accession from SIFTS mapping XML file.

    Input
    :param map_regions: XML object containing mapped regions in SIFTS
                        mapping XML file
    :type map_regions: bs4.element.Tag

    Output
    :return: ref_unipprot_secondary_acc; secondary UniProt accession including
             species identifier
    :rtype: str
    """
    try:
        db_details = map_regions.find("dbDetail", {"dbSource" : "UniProt"})
        ref_uniprot_secondary_acc = str(db_details.string)
    except:
        ref_uniprot_secondary_acc = ""

    return ref_uniprot_secondary_acc
    
def extract_mapping_details(map_data: TextIO,
                            pdb_res_name: str,
                            pdb_res_num: str) -> Tuple[str, str, str, str]:
    """"Get the UniProt details for a specific residue from the mapping XML
    file.
    
    Input
    :params map_data: text content of a mapping XML file
    :type map_data: TextIO

    :params pdb_res_name: the name of the residue
    :type pdb_res_name: str

    :params pdb_res_num: the sequence number of the residue
    :type pdb_res_num: str

    Output
    :return: map_uni_name_three, map_uni_num, ref_uniprot_acc, uniprot_name;
             UniProt details for a specific residue
    :rtype: Tuple[str, str, str, str]
    """
    Bs_data = bs4.BeautifulSoup(map_data, "xml")

    residue = Bs_data.find("residue", {"dbSource" : "PDBe",
                                     "dbCoordSys" : "PDBe",
                                     "dbResNum" : pdb_res_num,
                                     "dbResName" : pdb_res_name})

    (map_uni_num,
    ref_uniprot_acc,
    map_uni_name_three) = get_uniprot_crossref_details(residue)

    segment = residue.find_parent("segment")
    map_regions  = segment.find("listMapRegion")
    ref_uniprot_acc_species = get_secondary_uniprot_acc(map_regions)

    return (map_uni_name_three,
            map_uni_num,
            ref_uniprot_acc,
            ref_uniprot_acc_species)