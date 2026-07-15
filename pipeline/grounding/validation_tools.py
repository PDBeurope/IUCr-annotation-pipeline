import bs4
import logging
from typing import Tuple
from pipeline.utils import make_threeletter_code

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)


# def get_uniprot_crossref_details(residue: bs4.element.Tag) -> Tuple[str, str, str, str]:
def get_uniprot_crossref_details(
    residue: bs4.element.Tag, said_id: str
) -> Tuple[str, str, str]:
    """ "Get details for mapped UniProt residue from cross-reference database.

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
    try:
        uniprot_crossref = residue.find("crossRefDb", {"dbSource": "UniProt"})
    except KeyError:
        logging.error(f"No Uniprot crossref available for residue with chain {said_id}")
    if not uniprot_crossref:
        map_uni_num = ""
        map_uni_name = ""
        ref_uniprot_acc = ""
    else:
        try:
            map_uni_num = uniprot_crossref["dbResNum"]
            map_uni_name = uniprot_crossref["dbResName"]
            ref_uniprot_acc = uniprot_crossref["dbAccessionId"]
        except KeyError:
            logging.error("Unable to get details from UniProt crossref line")
        if not any((map_uni_num, map_uni_name, ref_uniprot_acc)):
            map_uni_num = ""
            map_uni_name = ""
            ref_uniprot_acc = ""
        else:
            try:
                map_uni_name_three = make_threeletter_code(map_uni_name)
            except Exception:
                logging.error("Unable to extract UniProt cross-ref details")
                map_uni_num = ""
                ref_uniprot_acc = ""
                map_uni_name_three = ""

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
        db_details = map_regions.find("dbDetail", {"dbSource": "UniProt"})
        ref_uniprot_secondary_acc = str(db_details.text)
    except AttributeError:
        logging.error("Unable to get secondary UniProt accession")
        ref_uniprot_secondary_acc = ""

    return ref_uniprot_secondary_acc


# def extract_mapping_details(map_data: TextIO,
#                             pdb_res_name: str,
#                             pdb_res_num: str) -> Tuple[str, str, str, str]:
def extract_mapping_details(
    map_data: bs4.BeautifulSoup, pdb_res_name: str, pdb_res_num: str, said_id: str
) -> Tuple[str, str, str, str]:
    """ "Get the UniProt details for a specific residue from the mapping XML
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
    # Bs_data = bs4.BeautifulSoup(map_data, "xml")

    entity = map_data.find("entity", attrs={"type": "protein", "entityId": said_id})

    # residue = Bs_data.find("residue", {"dbSource" : "PDBe",
    #                                  "dbCoordSys" : "PDBe",
    #                                  "dbResNum" : pdb_res_num,
    #                                  "dbResName" : pdb_res_name})
    if entity:
        try:
            residue = entity.find(
                "residue",
                attrs={
                    "dbSource": "PDBe",
                    "dbCoordSys": "PDBe",
                    "dbResNum": pdb_res_num,
                    "dbResName": pdb_res_name,
                },
            )
        except Exception:
            logging.error("Unable to find residue in entity of SIFTS XML")

    # (map_uni_num,
    # ref_uniprot_acc,
    # map_uni_name_three) = get_uniprot_crossref_details(residue)

    # segment = residue.find_parent("segment")
    # map_regions  = segment.find("listMapRegion")
    # ref_uniprot_acc_species = get_secondary_uniprot_acc(map_regions)

    # return (map_uni_name_three,
    #         map_uni_num,
    #         ref_uniprot_acc,
    #         ref_uniprot_acc_species)
    if residue:
        try:
            map_uni_num, ref_uniprot_acc, map_uni_name_three = (
                get_uniprot_crossref_details(residue, said_id)
            )
        except KeyError:
            logging.error("Unable to get crossref details from SIFTS")
            map_uni_num = ""
            ref_uniprot_acc = ""
            map_uni_name_three = ""
        if not any((map_uni_num, ref_uniprot_acc, map_uni_name_three)):
            logging.error("No Uniprot crossref details found")
        try:
            segment = residue.find_parent("segment")
            map_regions = segment.find("listMapRegion")
            ref_uniprot_acc_species = get_secondary_uniprot_acc(map_regions)
        except Exception:
            ref_uniprot_acc_species = ""

    return (map_uni_name_three, map_uni_num, ref_uniprot_acc, ref_uniprot_acc_species)
