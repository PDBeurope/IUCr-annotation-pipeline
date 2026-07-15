import logging
import bs4
from typing import Any, Dict
from pipeline.grounding.grounding_tools import (
    per_res_validation,
    per_res_validation_mapping,
    mutant_validation,
)

# from pipeline.utils import (make_val_file_list,
#                             make_map_file_list,
from pipeline.utils import build_val_file_path, build_map_file_path, read_xml_gz_file

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)


# def validate_residue_name_number(ann: Dict[str, Any],
#                                  text: str,
#                                  val_file_list: List[str]) -> Dict[str, Any]:
def validate_residue_name_number(
    ann: Dict[str, Any], text: str, val_data: bs4.BeautifulSoup
) -> Dict[str, Any]:
    """
    Function to validate annotations of type 'residue_name_number' against
    corresponding residues in a PDBe validation XML file.

    Input
    :param ann: dictionary containing a single annotation
    :type ann: Dict[str, Any]

    :param text: the exact annotation found
    :type text: str

    :param val_file_list: list of file paths to validation XML files
    :type val_file_list: List[str]

    Output
    :return: ann; updated dictionary for a single annotation, now containing
             validation stats for residues found in the validation XML files
             matching the annotation text
    :rtype: Dict[str, Any]
    """
    # val_list = []
    # for vx in val_file_list:
    #     val_data = read_xml_file(vx)
    #     try:
    #         res_tag_list = per_res_validation(val_data, text)
    #         if res_tag_list:
    #             val_list.append((res_tag_list))
    #     except:
    #         continue
    # flat_res_val_list = [x for xs in val_list for x in xs]
    # ann["tags"] = flat_res_val_list

    # return ann
    if val_data:
        try:
            res_tag_list = per_res_validation(val_data, text)
        except ValueError:
            logging.error(
                f"Unable to create res_tag_list for residue_name_number {text}"
            )

    ann["tags"] = res_tag_list

    return ann


# def validate_mutant(ann: Dict[str, Any],
#                     text: str,
#                     val_file_list: List[str]) -> Dict[str, Any]:
def validate_mutant(
    ann: Dict[str, Any], text: str, val_data: bs4.BeautifulSoup
) -> Dict[str, Any]:
    """
    Function to validate annotations of type 'mutant' against
    corresponding residues in a PDBe validation XML file.

    Input
    :param ann: dictionary containing a single annotation
    :type ann: Dict[str, Any]

    :param text: the exact annotation found
    :type text: str

    :param val_file_list: list of file paths to validation XML files
    :type val_file_list: List[str]

    Output
    :return: ann; updated dictionary for a single annotation, now containing
             validation stats for residues found in the validation XML files
             matching the annotation text
    :rtype: Dict[str, Any]
    """
    # mutant_val_list = []
    # for vx in val_file_list:
    #     val_data = read_xml_file(vx)
    #     try:
    #         mutant_tag_list = mutant_validation(val_data, text)
    #         if mutant_tag_list:
    #             mutant_val_list.append(mutant_tag_list)
    #     except:
    #         continue
    # flat_mutant_val_list = [x for xs in mutant_val_list for x in xs]
    # ann["tags"] = flat_mutant_val_list

    # return ann
    if val_data:
        try:
            logging.info(f"Running mutant validation for {text}")
            mutant_tag_list = mutant_validation(val_data, text)
        except ValueError:
            logging.error(f"Unable to create res_tag_list for mutant {text}")

    ann["tags"] = mutant_tag_list

    return ann


# def map_anno_tags(ann: Dict[str, Any],
#                   map_file_list: List[str]) -> Dict[str, Any]:
def map_anno_tags(ann: Dict[str, Any], map_data: bs4.BeautifulSoup) -> Dict[str, Any]:
    """
    Function to map validated residues to their reference sequence in UniProt
    using SIFTS mapping XML files.

    Input
    :param ann: dictionary for a single annotation with validated residues from
                PDBe validation XML files
    :type ann: Dict[str, Any]

    :param map_file_list: list of file paths to SIFTS mapping XML files
    :type map_file_list: List[str]

    Output
    :return: ann; updated dictionary for a single annotation, now containing
             validation stats for residues found in the validation XML
             files matching the annotation text and mapping to reference
             sequence in UniProt from SIFTS mapping XML files
    :rtype: Dict[str, Any]
    """
    # updated_tags = []
    # for tag in ann["tags"]:
    #     tag_pdb = tag["pdb_id"]
    #     map_file_match = [i for i in map_file_list if tag_pdb.lower() in i]
    #     file_path = map_file_match[0]
    #     if file_path.endswith(".gz"):
    #         map_data = read_xml_gz_file(file_path)
    #     else:
    #         map_data = read_xml_file(file_path)
    #     try:
    #         mapped_tag = per_res_validation_mapping(map_data, tag)
    #         updated_tags.append(mapped_tag)
    #     except Exception as e:
    #         logging.error(f"Unable to get mapping to UniProt {e}")
    #         tag["uniprot_id"] = ""
    #         tag["uniprot_name"] = ""
    #         tag["uniprot_res"] = ""
    #         tag["uniprot_uri"] = ""
    #         updated_tags.append(tag)

    # return ann
    updated_tags = []

    for tag in ann["tags"]:
        tag_pdb = tag["pdb_id"]
        try:
            logging.info("Running SIFTS mapping for annotation tag")
            mapped_tag = per_res_validation_mapping(map_data, tag)
            updated_tags.append(mapped_tag)
            logging.info("Finished mapping annotation tag to SIFTS")
        except KeyError:
            logging.error(f"Mapping failed for {tag_pdb}.")
            tag["uniprot_id"] = ""
            tag["uniprot_name"] = ""
            tag["uniprot_res"] = ""
            tag["uniprot_uri"] = ""
            updated_tags.append(tag)

    ann["tags"] = updated_tags

    return ann


# def term_grounding_with_epmc_json(json: Dict[str, Any],
#                                   val_dir: str,
#                                   map_dir: str) -> Dict[str, Any]:
def term_grounding_with_epmc_json(
    json: Dict[str, Any],
    val_dir: str,
    map_dir: str,
    pdb_id: str = "",
) -> Dict[str, Any]:
    """
    Function to run residue grounding and validation using EuropePMC
    annotation JSON.

    Input
    :param json: EuropePMC-style annotation JSON
    :type json: Dict[str, Any]

    :param val_dir: path to directory containing validation XML files
    :type val_dir: str

    :param map_dir: path to directory containing gzipped SIFTS mapping XML
    files
    :type mad_dir: str

    Output
    :return: json; EuropePMC-style annotation JSON expanded for residue
             validation stats and SIFTS mapping to UniProt
    :rtype: Dict[str, Any]
    """
    # try:
    #     structures = json["linked_all_pdbs"]
    #     if structures:
    #         try:
    #             val_file_list = make_val_file_list(structures, val_dir)
    #         except:
    #             logging.error(f"No validation XML file found")
    #             pass
    #         try:
    #             map_file_list = make_map_file_list(structures, map_dir)
    #         except Exception as e:
    #             logging.error(f"No mapping XML file found. Error {e}")
    #             pass

    #         for ann in json["anns"]:
    #             text = ann["exact"].upper()
    #             if ann["type"] == "residue_name_number":
    #                 ann = validate_residue_name_number(ann,
    #                                                    text,
    #                                                    val_file_list)
    #                 ann = map_anno_tags(ann, map_file_list)

    #             if ann["type"] == "mutant":
    #                 ann = validate_mutant(ann,
    #                                       text,
    #                                       val_file_list)
    #                 ann = map_anno_tags(ann, map_file_list)

    #         return json
    # except:
    #     logging.error("No JSON file with EuropePMC annotations provided.")
    try:
        logging.info(f"Running residue grounding and validation for {pdb_id}...")
        if pdb_id:
            structures = pdb_id
        if structures:
            try:
                val_file_path = build_val_file_path(structures, val_dir)
                logging.info(f"Validation XML files found: {val_file_path}")
            except ValueError as e:
                logging.error(f"No validation XML file found. Error {e}")
            try:
                val_data = read_xml_gz_file(val_file_path)
            except ValueError:
                logging.error(f"Unable to read validation XML file for {structures}")
            try:
                val_soup = bs4.BeautifulSoup(val_data, "xml")
            except Exception:
                logging.error(f"Unable to parse validation XML file for {structures}")
            try:
                map_file_path = build_map_file_path(structures, map_dir)
                logging.info(f"SIFTS mapping file found: {map_file_path}")
            except ValueError as e:
                logging.error(f"No mapping XML file found. Error {e}")
            try:
                map_data = read_xml_gz_file(map_file_path)
            except ValueError:
                logging.error(f"Unable to read SIFTS mapping XML for {structures}")
            try:
                map_soup = bs4.BeautifulSoup(map_data, "xml")
            except Exception:
                logging.error(f"Unable to parse SIFTS mapping XML for {structures}")

            xml_dict = {val_file_path: val_soup, map_file_path: map_soup}

            for ann in json["anns"]:
                if ann["type"] != "residue_name_number" and ann["type"] != "mutant":
                    continue
                else:
                    text = ann["exact"].upper()
                    ai_score = float(ann["ai_score"]) >= 0.75
                    ai_score_threshold = 0.75
                    if (
                        ann["type"] == "residue_name_number"
                        and ai_score >= ai_score_threshold
                    ):
                        logging.info(
                            "Running validation for residue_name_number annotation."
                        )
                        ann = validate_residue_name_number(
                            ann, text, xml_dict[val_file_path]
                        )
                        logging.info(
                            "Finished validating residue_name_number annotation."
                        )
                        if ann["tags"]:
                            logging.info(
                                "Running SIFTS mapping for residue_name_number annotation."
                            )
                            ann = map_anno_tags(ann, xml_dict[map_file_path])
                            logging.info(
                                "Finished SIFTS mapping for residue_name_number annotation."
                            )
                    if ann["type"] == "mutant" and ai_score >= ai_score_threshold:
                        logging.info("Running validation for mutant annotation.")
                        ann = validate_mutant(ann, text, xml_dict[val_file_path])
                        logging.info(
                            "Finished getting validation stats for mutant anntation"
                        )
                        if ann["tags"]:
                            logging.info("Running SIFTS mapping for mutant annotation.")
                            ann = map_anno_tags(ann, xml_dict[map_file_path])
                            logging.info(
                                "Finished SIFTS mapping for mutant annotation."
                            )
    except ValueError as e:
        logging.error(f"No JSON file with EuropePMC annotations provided. Error {e}")
        json = {}

    return json
