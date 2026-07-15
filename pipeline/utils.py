import logging
import requests
import os
import gzip
import json
import re
import bs4
from requests.adapters import HTTPAdapter
from urllib3 import Retry
from nltk.tokenize import WordPunctTokenizer
from solrq import Q
from typing import Any, Dict, List, Tuple, TextIO


logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)


short_aa_dict = {
    "A": {
        "id": "Ala",
        "full_id": "Alanine",
    },
    "R": {
        "id": "Arg",
        "full_id": "Arginine",
    },
    "N": {
        "id": "Asn",
        "full_id": "Asparagine",
    },
    "D": {
        "id": "Asp",
        "full_id": "Aspartic acid (Aspartate)",
    },
    "C": {
        "id": "Cys",
        "full_id": "Cysteine",
    },
    "Q": {
        "id": "Gln",
        "full_id": "Glutamine",
    },
    "E": {
        "id": "Glu",
        "full_id": "Glutamic acid (Glutamate)",
    },
    "G": {
        "id": "Gly",
        "full_id": "Glycine",
    },
    "H": {
        "id": "His",
        "full_id": "Histidine",
    },
    "I": {
        "id": "Ile",
        "full_id": "Isoleucine",
    },
    "L": {
        "id": "Leu",
        "full_id": "Leucine",
    },
    "K": {
        "id": "Lys",
        "full_id": "Lysine",
    },
    "M": {
        "id": "Met",
        "full_id": "Methionine",
    },
    "F": {
        "id": "Phe",
        "full_id": "Phenylalanine",
    },
    "P": {
        "id": "Pro",
        "full_id": "Proline",
    },
    "O": {
        "id": "Pyl",
        "full_id": "Pyrrolysine",
    },
    "U": {"id": "Sec", "full_id": "Selenocysteine"},
    "S": {
        "id": "Ser",
        "full_id": "Serine",
    },
    "T": {
        "id": "Thr",
        "full_id": "Threonine",
    },
    "W": {
        "id": "Trp",
        "full_id": "Tryptophan",
    },
    "Y": {
        "id": "Tyr",
        "full_id": "Tyrosine",
    },
    "V": {
        "id": "Val",
        "full_id": "Valine",
    },
    "B": {
        "id": "Asx",
        "full_id": "Aspartic acid or Asparagine",
    },
    "Z": {
        "id": "Glx",
        "full_id": "Glutamine or Glutamic acid",
    },
    "X": {
        "id": "Xaa",
        "full_id": "Any amino acid",
    },
    "J": {"id": "Xle", "full_id": "Leucine or Isoleucine"},
    "None": {
        "id": "TERM",
        "full_id": "termination codon",
    },
}


def make_threeletter_code(one_letter_res: str) -> str:
    """
    Function to turn 1-letter amino acid residue into
    3-letter amino acid residue.

    Input

    :param one_letter_res: 1-letter amino acid residue
    :type one_letter_res: str


    Output

    :return: three_letter_res; 3-letter amino acid residue
    :rtype: str

    """
    try:
        chars = [x for x in one_letter_res]
        aminoacid: str = ""
        seq_num: str = ""
        for c in chars:
            if c.isalpha():
                aminoacid += c
            if c.isnumeric():
                seq_num += c
        threeletter = short_aa_dict[aminoacid]["id"].upper()
        three_letter_res = threeletter + seq_num
    except KeyError:
        three_letter_res = ""
        pass
    return three_letter_res


def make_ori_mutant_threeletter(one_letter_res: str) -> Tuple[str, str, str, str]:
    """
    Function to turn 1-letter amino acid into 3-letter amino acid
    and identify which part in a mutation is the wildtype residue
    and which one th emutation.

    Input

    :param one_letter_res: 1-letter amino acid mutation
    :type one_letter_res: str


    Output

    :return: full_ori_res, full_mutant_res, aminoacid, seq_num; wildtype
             3-letter amino acid residue with sequence position, mutant
             3-letter with sequence position, wildtype 3-letter amino acid
             name, wildtype amino acid sequenec number
    :rtype: Tuple[str, str, str, str]
    """
    aminoacid: str = ""
    seq_num: str = ""
    mutant_chars = [x for x in one_letter_res]
    ori_res_fix = str("".join(mutant_chars[:-1]))
    for c in ori_res_fix:
        if c.isalpha():
            aminoacid += c
        if c.isnumeric():
            seq_num += c
    aminoacid = short_aa_dict[aminoacid]["id"].upper()
    full_ori_res = aminoacid + seq_num
    mutant_res = mutant_chars[-1]
    mutant_threeletter = short_aa_dict[mutant_res]["id"].upper()
    full_mutant_res = mutant_threeletter + seq_num

    return full_ori_res, full_mutant_res


def make_ori_mutant(three_letter_res: str) -> Tuple[str, str, str, str]:
    """
    Function to turn 1-letter amino acid into 3-letter amino acid
    and identify which part in a mutation is the wildtype residue
    and which one th emutation.

    Input

    :param three_letter_res: 1-letter amino acid mutation
    :type three_letter_res: str


    Output

    :return: full_ori_res, full_mutant_res, aminoacid, seq_num; wildtype
             3-letter amino acid residue with sequence position, mutant
             3-letter with sequence position, wildtype 3-letter amino acid
             name, wildtype amino acid sequenec number
    :rtype: Tuple[str, str, str, str]
    """
    aminoacid: str = ""
    seq_num: str = ""
    mutant_chars = [x for x in three_letter_res]
    ori_res_fix = str("".join(mutant_chars[:-3]))
    for c in ori_res_fix:
        if c.isalpha():
            aminoacid += c
        if c.isnumeric():
            seq_num += c
    full_ori_res = aminoacid + seq_num
    mutant_res = str("".join(mutant_chars[-3:]))
    full_mutant_res = mutant_res + seq_num

    return full_ori_res, full_mutant_res


def split_res_and_chain(three_letter_res: str) -> str:
    """
    Break up a residue_name_number entity that may have additional characters
    to identify protein chains to just keep the residue name and the sequence
    number; uses three-letter code.

    Input

    :param three_letter_res: entity converted to 3-letter amino acid residue
                             and found to have additional characters
    :type three_letter_res: str


    Output

    :return: full_res; entity in 3-letter amino acid residue only with
             additional characters removed
    :rtype: str

    """
    aminoacid: str = ""
    seq_num: str = ""
    chars = [x for x in three_letter_res]
    for c in chars:
        if c.isalpha():
            aminoacid += c
        if c.isnumeric():
            seq_num += c
    aminoacid = aminoacid[0:3]
    full_res = aminoacid + seq_num

    return full_res


def split_mutant_and_chain(mutant_with_chain: str) -> Tuple[str, str]:
    """
    Break up a mutant entity that may have additional characters
    to identify protein chains to just keep the residue name and the sequence
    number; uses three-letter code.

    Input

    :param mutant_with_chain: 3-letter mutant with additional characters
    :type mutant_with_chain: str


    Output

    :return: tuple of wildtype and mutant with chain identifier removed
    :rtype: Tuple[str, str]

    """
    aminoacid: str = ""
    seq_num: str = ""
    chars = [x for x in mutant_with_chain]
    for c in chars:
        if c.isalpha():
            aminoacid += c
        if c.isnumeric():
            seq_num += c
    wildtype_aa = aminoacid[0:3]
    mutant_aa = aminoacid[-4:-1]
    wildtype = wildtype_aa + seq_num
    mutant = mutant_aa + seq_num

    return wildtype, mutant


def downloading_val_xml(pdb: str, val_dir: str) -> None:
    """
    Helper function to download validation XML files from PDBe.

    Input
    :param pdb: a PDB ID
    :type pdb: str

    :param val_dir: path to a dirctory writing validation XML files
    :type val_dir: str
    """
    base = "https://www.ebi.ac.uk/pdbe/entry-files/download/"
    full = base + pdb + "_validation.xml"
    result = requests.get(full)
    out = val_dir + pdb + "_validation.xml"
    with open(out, "w+") as f:
        f.write(result.text)


def downloading_map_xml(pdb: str, map_dir: str) -> None:
    """
    Helper function to download SIFTS mapping XML files from PDBe.

    Input
    :param pdb: a PDB ID
    :type pdb: str

    :param map_dir: path to a dirctory writing gzipped SIFTS mapping XML files
    :type map_dir: str
    """
    chunk_size = 128
    base = "https://www.ebi.ac.uk/pdbe/files/sifts/"
    full = base + pdb + ".xml.gz"
    result = requests.get(full, stream=True)
    out = map_dir + pdb + ".xml.gz"
    with open(out, "wb") as fd:
        for chunk in result.iter_content(chunk_size=chunk_size):
            fd.write(chunk)


def make_path_xml_content_dict(filepath_list, structures):
    dict = {}
    for structure in structures:
        for filepath in filepath_list:
            if filepath.endswith(".gz") and structure in filepath:
                try:
                    data = read_xml_gz_file(filepath)
                except ValueError:
                    logging.error(f"Unable to read XML file for {filepath}")
                try:
                    soup = bs4.BeautifulSoup(data, "xml")
                except Exception:
                    logging.error(f"Unable to parse XML for {filepath}")
            elif filepath.endswith(".xml") and structure in filepath:
                try:
                    data = read_xml_file(filepath)
                except ValueError:
                    logging.error(f"Unable to read XML file for {filepath}")
                try:
                    soup = bs4.BeautifulSoup(data, "xml")
                except Exception:
                    logging.error(f"Unable to parse XML for {filepath}")

            dict[structure] = soup

    return dict


def make_val_file_list(structures: List[str], val_dir: str) -> List[str]:
    """
    Create a list of directory paths to validation XML files.

    Input
    :param structures: list of PDB IDs
    :type structures: List[str]

    :param val_dir: base directory of validation XML location
    :type val_dir: str

    Output
    :return: val_file_list; list of directory paths to validation XML files
    :rtype: List[str]
    """
    val_file_list: List[str] = []
    for struc in structures:
        if os.path.isfile(val_dir + struc + "_validation.xml"):
            val_file_list.append(val_dir + struc + "_validation.xml")
        else:
            downloading_val_xml(struc, val_dir)
            val_full = val_dir + struc + "_validation.xml"
            val_file_list.append(val_full)
    return val_file_list


def make_map_file_list(structures: List[str], map_dir: str) -> List[str]:
    """
    Create a list of directory paths to SIFTS mapping XML files.

    Input
    :param structures: list of PDB IDs
    :type structures: List[str]

    :param map_dir: base directory of SIFTS mapping XML location
    :type map_dir: str

    Output
    :return: map_file_list; list of directory paths to SIFTS mapping XML files
    :rtype: List[str]
    """
    map_file_list: List[str] = []
    for struc in structures:
        if os.path.isfile(map_dir + struc + ".xml.gz"):
            map_file_list.append(map_dir + struc + ".xml.gz")
        else:
            downloading_map_xml(struc, map_dir)
            map_full = map_dir + struc + ".xml.gz"
            map_file_list.append(map_full)
    return map_file_list


def read_xml_file(filepath: str) -> TextIO:
    """
    Helper function to read in an XML file.

    Input
    :param filepath: path to XML file
    :type filepath: str

    Output
    :return: data; text content of XML file
    :rtype: TextIO
    """
    with open(filepath) as f:
        data = f.read()
    return data


def read_xml_gz_file(filepath: str) -> TextIO:
    """
    Helper function to read in a gzipped XML file.

    Input
    :param filepath: path to gzipped XML file
    :type filepath: str

    Output
    :return: data; text content of XML file
    :rtype: TextIO
    """
    with gzip.open(filepath, "rb") as f:
        data = f.read()
    return data


def write_output_json(ml_json: Dict[str, Any], outfile: str) -> None:
    """
    Helper function to write final JSON to disk.

    Input
    :param ml_json: dictionary with JSON content to be written out
    :type ml_json: Dict[str, Any]

    :param outfile: path to writing destination
    :type outfile: str
    """
    with open(outfile, "w", encoding="utf-8") as f:
        json.dump(ml_json, f, indent=4, ensure_ascii=False)


def expand_to_epmc_annotation(
    each_sentence: str, m: Dict[str, Any], annotator: str
) -> Dict[str, Any]:
    """
    Function to expand a single annotation to follow EuropePMC-style JSON.

    Input
    :param each_sentence: the sentence carrying the annotation
    :type each_sentence: str

    :param m: dictionary for a single sentence to extract from
    :type m: Dict[str, Any]

    :param annotator: the name of the annotator/algorithm
    :type annotator: str

    Output
    :return: minidict; sentence dictionary in EuropePMC-style
    :rtype: Dict[str, Any]
    """
    minidict: Dict[str, Any] = {}
    minidict["sentence"] = each_sentence
    sent_id = m["id"]
    minidict["sent_id"] = sent_id
    minidict["section"] = m["section"]
    a_start = m["char_start"]
    a_end = m["char_end"]
    try:
        tokenizer = WordPunctTokenizer()
        sent_front = each_sentence[:a_start]
        split_text = tokenizer.tokenize(sent_front)
        w_position = len(split_text) + 1
        epmc_w_pos = str(sent_id) + "." + str(w_position)
        minidict["position"] = epmc_w_pos
    except Exception:
        minidict["position"] = ""
    try:
        if a_start == 0 or a_start <= 30:
            sent_front = each_sentence[:a_start]
            minidict["prefix"] = sent_front
        else:
            a_start_new = a_start - 30
            sent_front = each_sentence[a_start_new:a_start]
            minidict["prefix"] = sent_front
    except Exception:
        minidict["prefix"] = ""
    try:
        anno_length = a_end - a_start
        sent_back = each_sentence[a_end:]
        if len(sent_back) == anno_length:
            sent_back = each_sentence[a_end:]
            minidict["postfix"] = sent_back
        else:
            a_end_new = a_end + 30
            sent_back = each_sentence[a_end:a_end_new]
            minidict["postfix"] = sent_back
    except Exception:
        minidict["postfix"] = ""

    minidict["exact"] = m["exact"]
    minidict["type"] = m["type"]
    minidict["ai_score"] = str(m["ai_score"])
    minidict["annotator"] = annotator
    minidict["char_start"] = m["char_start"]
    minidict["char_end"] = m["char_end"]
    minidict["tags"] = {}

    return minidict


def make_request(search_dict: Dict[str, str]) -> Dict[str, Any]:
    """
    Make a get request to the PDBe API.

    Input
    :param search_dict: the terms used to search
    :type search_dict: dict

    Output
    :return dict: response JSON
    :rtype: Dict[str, Any]
    """
    search_url = "https://www.ebi.ac.uk/pdbe/search/pdb/select?"
    response = requests.post(search_url, data=search_dict)

    if response.status_code == 200:
        return response.json()
    else:
        logging.error(f"[No data retrieved - {response.status_code}] {response.text}")

    return {}


def format_search_terms(
    search_terms: str, filter_terms: str | None, number_of_rows: int = 100
) -> Dict[str, str]:
    """
    Format search and filter terms for PDBe API request.

    Input
    :param search_terms: the terms used to search
    :type search_terms: str

    :param filter_terms: terms to filter the search results for
    type filter_terms: str | None

    Output
    :return ret: dictionary with formated search and filter terms
    :rtype: Dict[str, str]
    """
    ret = {"q": str(search_terms)}
    if filter_terms:
        fl = "{}".format(",".join(filter_terms))
        ret["fl"] = fl
    ret["rows"] = number_of_rows

    return ret


def run_search(
    search_terms: str, filter_terms: str | None, number_of_rows: int = 100
) -> Dict[str, str]:
    """
    Execute functions for a PDBe API request.

    Input
    :param search_terms: the terms used to search
    :type search_terms: str

    :param filter_terms: terms to filter the search results for
    type filter_terms: str | None

    Output
    :return results: dictionary with the JSON response
    :return dict: Dict[str, str]
    """
    search_term = format_search_terms(search_terms, filter_terms, number_of_rows)
    response = make_request(search_term)
    results = response.get("response", {}).get("docs", [])
    return results


def get_linked_structures(src: str, paperid: str) -> List[str]:
    """
    Make API call to Europe PMC to get linked structures.

    Input
    :param src: indexed publication resource (PMC or MED)
    :type src: str

    :param paperid: unique ID for publication; either PMCID or PMID; needs
                    to match the src
    :type paperid: str

    Output
    :return structures: list of PDB IDs linked to the publication
    :rtype: List[str]
    """
    structures: List[str] = []
    url = "https://www.ebi.ac.uk/europepmc/webservices/rest/" + src + "/"
    full_url = url + paperid + "/datalinks?category=Protein%20Structures&format=json"

    try:
        result = requests_retry_session().get(full_url, timeout=5).json()
    except Exception:
        logging.error(f"Unable to get linked structures from EuropePMC for {paperid}.")
    try:
        result_list = result["dataLinkList"]
        cat = result_list["Category"]
        for c in cat:
            if c["Name"] == "Protein Structures":
                sec = c["Section"]
                for s in sec:
                    r_list = s["Linklist"]
                    link_list = r_list["Link"]
                    for link in link_list:
                        target = link["Target"]
                        ident = target["Identifier"]
                        entry = ident["ID"]
                        structures.append(entry)
    except Exception:
        structures = []
    return structures


def pmid_pmcid_from_title(title: str, doi: str) -> Tuple[str, str]:
    """
    Make API call to Europe PMC to get PubMed and PubMed Central IDs using a
    publication title.

    Input
    :param title: article title
    :type title: str

    :param doi: article DOI
    :type doi: str

    Output
    :return: Tuple of PubMed ID and PubMed Central IDfor article
    :rtype: Tuple[str, str]
    """
    pmid: str = ""
    pmcid: str = ""
    url = "https://www.ebi.ac.uk/europepmc/webservices/rest/search?query="
    full_url = url + title + "&resultType=core&cursorMark=*&pageSize=25&format=json"
    try:
        # result = requests.get(full_url).json()
        result = requests_retry_session().get(full_url, timeout=5).json()
    except Exception:
        logging.error(
            f"Unable to retrieve PMID and PMCID for title from EuropePMC for {title}."
        )
    if "resultList" not in result:
        return "", ""
    result_list = result["resultList"]
    r_list = result_list["result"]
    for r in r_list:
        if r["doi"].upper() == doi or r["doi"] == doi:
            pmid = r_list[0]["pmid"]
            pmcid = r_list[0]["pmcid"]
            return pmid, pmcid
        else:
            continue


def get_article_license(soup: bs4.BeautifulSoup) -> str:
    """
    Get article license from XML soup.

    Input
    :param soup: article as XML soup
    :type soup: bs4.BeautifulSoup

    Output
    :return publisher_license: publisher license type
    :rtype: str
    """
    publisher_license: str = ""
    article_permissions = soup.find_all("license")
    if article_permissions:
        for article_license in article_permissions:
            if article_license.has_attr("license-type"):
                publisher_license = article_license.get("license-type")
                return publisher_license


def get_publisher_id_and_doi(soup: bs4.BeautifulSoup) -> Tuple[str, str]:
    """
    Get different article IDs from XML soup: pmid, pmcid, doi, publisher_id.

    Input
    :param soup: article as XML soup
    :type soup: bs4.BeautifulSoup

    :param title: article title
    :type title: str

    Output
    :return: Tuple of article PubMed ID, PubMed Central ID, DOI and
                  publisher ID
    :rtype: Tuple[str, str, str, str]
    """
    doi: str = ""
    publisher_id: str = ""
    article_id_list: List[bs4.Tag]
    article_id_list = soup.find_all("article-id")
    if article_id_list:
        try:
            publisher_details = soup.find("article-id", {"pub-id-type": "publisher-id"})
            publisher_id = str(publisher_details.string)
        except Exception:
            publisher_id = ""
        try:
            doi_details = soup.find("article-id", {"pub-id-type": "doi"})
            doi = str(doi_details.string)
        except Exception:
            doi = ""

        return publisher_id, doi


def get_pdb_id_from_supplemental(
    soup: bs4.BeautifulSoup,
) -> Tuple[List[str], List[str]]:
    """
    Get linked PDB structures from article supplemental material.

    Input
    :param soup: article as XML soup
    :type soup: BeautifulSoup

    Output
    :return: Tuple of list of all linked PDB IDs and list of all
                      linked primary PDB IDs
    :rtype: Tuple[List[str], List[str]]
    """
    all_pdbs: List[str] = []
    primary_pdbs: List[str] = []
    supple = soup.find_all("supplementary-material")
    for sup in supple:
        s_para = sup.find_all("p")
        # look for the links:
        # <ext-link ext-link-type="uri"
        # xlink:href="https://doi.org/10.2210/pdb7xrz/pdb">BRIL–SRP2070Fab
        # complex, 7xrz</ext-link>
        # <ext-link ext-link-type="pdb" xlink:href="1t0a">
        for s in s_para:
            plnk = s.find_all("ext-link")
            for link in plnk:
                lt = ""
                lh = ""
                pc = ""
                if link.has_attr("ext-link-type"):
                    lt = link.get("ext-link-type")
                if link.has_attr("xlink:href"):
                    lh = link.get("xlink:href")
                if lt == "pdb":
                    pc = lh
                else:
                    ms = re.search(r"/pdb([^/]+)/pdb", lh)
                    if ms:
                        pc = ms.group(1)
                if pc != "":
                    primary_pdbs.append(pc)
                    all_pdbs.append(pc)
    return all_pdbs, primary_pdbs


def get_article_title(soup: bs4.BeautifulSoup) -> str:
    """
    Get article title from soup.

    Input
    :param soup: article as XML soup
    :type soup: bs4.BeautifulSoup

    Output
    :return title: article title
    :rtype: str
    """
    title: str = ""
    title = soup.find_all("article-title")
    title = title[0].text.strip()
    return title


def get_all_linked_structures(
    soup: bs4.BeautifulSoup, pmid: str, pmcid: str
) -> Tuple[List[str], List[str], List[str]]:
    """
    Get all linked structures for article.

    Input
    :param soup: article as XML soup
    :type soup: bs4.BeautifulSoup

    :param pmid: PubMed ID for article
    :type pmid: str

    :param pmcid: PubMed Central ID for article
    :type pmcid: str

    Output
    :return: Tuple of all linked PDB IDs, primary PDB IDs and additional
             PDB IDs for article
    :rtype: Tuple[List[str], List[str], List[str]]
    """
    all_pdbs: List[str] = []
    primary_pdbs: List[str] = []
    secondary_pdbs: List[str] = []

    all_pdbs, primary_pdbs = get_pdb_id_from_supplemental(soup)

    secondary_pdbs = []
    if pmid != "" or pmcid != "":
        if pmcid != "":
            src = "PMC"
            entries = get_linked_structures(src, pmcid)
            for entry in entries:
                all_pdbs.append(entry)
                if entry in primary_pdbs and all_pdbs:
                    continue
                else:
                    secondary_pdbs.append(entry)
        elif pmid != "":
            src = "MED"
            entries = get_linked_structures(src, pmid)
            for entry in entries:
                all_pdbs.append(entry)
                if entry in primary_pdbs and all_pdbs:
                    continue
                else:
                    secondary_pdbs.append(entry)
    all_pdbs = list(set(all_pdbs))
    return all_pdbs, primary_pdbs, secondary_pdbs


def make_organism_dict_for_pdbs(all_pdbs: List[str]) -> Dict[str, Dict[str, Any]]:
    """
    Get species and taxonomy identifier for all linked structures for article.

    Input
    :param all_pdbs: list of all linked PDB IDs
    :type all_pdbs: List[str]


    Output
    :return pdb_prot_species_dict: nested dictionry with PDB ID as key and
                                   dictionary of species details as value
    :rtype: Dict[str, Dict[str, Any]]
    """
    pdb_prot_species_dict: Dict[str, List[Dict[str, Any]]] = {}
    for pdb in all_pdbs:
        org_science: str = ""
        org_syn: str = ""
        tax_id: str = ""
        uni_access: str = ""
        search_terms = Q(pdb_id=pdb.lower())
        filter_terms = [
            "organism_scientific_name",
            "organism_synonyms",
            "tax_id",
            "uniprot_accession",
        ]
        results = run_search(search_terms, filter_terms, number_of_rows=100)
        if len(results) == 0:
            continue
        species_list = []
        for r in results:
            mini: Dict[str, Any] = {}
            try:
                org_science = r["organism_scientific_name"][0]
                mini["organism_scientific_name"] = org_science
            except KeyError:
                mini["organism_scientific_name"] = ""
            try:
                org_syn = r["organism_synonyms"]
                mini["organism_synonyms"] = org_syn
            except KeyError:
                mini["organism_synonyms"] = []
            try:
                tax_id = r["tax_id"]
                mini["tax_id"] = tax_id
            except KeyError:
                mini["tax_id"] = []
            try:
                uni_access = r["uniprot_accession"][0]
                mini["uniprot_accession"] = uni_access
            except KeyError:
                mini["uniprot_accession"] = ""
            species_list.append(mini)
        pdb_prot_species_dict[pdb] = species_list
    return pdb_prot_species_dict


def make_output_json(
    source: str,
    pmcid: str,
    pmid: str,
    doi: str,
    publisher_id: str,
    publisher_license: str,
    pmc_access: str,
    permitting_license: str,
    title: str,
    primary_pdbs: List[str],
    secondary_pdbs: List[str],
    all_pdbs: List[str],
    pdb_prot_species_dict: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Make the metadata part of the final JSON.

    Input
    :param source: indexed publication resource (PMC, MED or '')
    :type source: str

    :param pmcid: PubMed Central ID for article
    :type pmcid: str

    :param pmid: PubMed ID for article
    :type pmid: str

    :param doi: artcile DOI
    :type doi: str

    :param publisher_id: article publisher ID
    :type publisher_id: str

    :param publisher_license: article publisher license type
    :type publisher_license: str

    :param title: article title
    :type title: str

    :param primary_pdbs: list of all linked primary PDB IDs
    :type primary_pdbs: List[str]

    :param secondary_pdbs: list of all additional linked PDB IDs
    :type secondary_pdbs: List[str]

    param all_pdbs: list of all liked PDB IDs
    :type all_pdbs: List[str]


    Output
    :return json_generated: dictionry with article metadata
    :rtype: Dict[str, Any]
    """
    json_generated: Dict[str, Any] = {}
    json_generated["src"] = source
    json_generated["id"] = pmcid
    json_generated["pmcid"] = pmcid
    json_generated["pmid"] = pmid
    json_generated["doi"] = doi
    json_generated["publisher_id"] = publisher_id
    json_generated["publisher_license"] = publisher_license
    json_generated["pmc_access"] = pmc_access
    json_generated["permitting_license"] = permitting_license
    json_generated["title"] = title
    json_generated["linked_primary_pdbs"] = primary_pdbs
    json_generated["linked_secondary_pdbs"] = secondary_pdbs
    json_generated["linked_all_pdbs"] = all_pdbs
    json_generated["species_all_pdbs"] = pdb_prot_species_dict
    json_generated["provider"] = "IUCr"
    return json_generated


def get_pmc_access(pmcid: str) -> Tuple[str, str]:
    """
    API call to get PMC access and license information for given PubMed Central ID.

    Input
    :param pmcid: unique PubMed Central ID for article
    :type pmcid: str


    Output
    :return pmc_access, permitting_license: Tuple of PMC access and whether the license is permissive or not
    :rtype: Tuple[str, str]
    """
    full_url = (
        "https://www.ebi.ac.uk/europepmc/webservices/rest/article/"
        + "PMC/"
        + pmcid
        + "?resultType=core&format=json"
    )
    try:
        response = requests_retry_session().get(full_url, timeout=5).json()
        details = response["result"]
    except Exception:
        logging.error(
            f"Unable to retrieve article information from EuropePMC for {pmcid}."
        )
    try:
        pmc_access = details["isOpenAccess"]
    except Exception:
        pmc_access = ""

    return pmc_access


def requests_retry_session(
    retries: int = 3,
    backoff_factor: float = 0.3,
    status_forcelist: Tuple[int, int, int] = (500, 502, 504),
    session: requests.Session | None = None,
):
    """
    Applying retry logic to requests seesion handeling API errors and timeouts

    Input
    :param retries: number of retries to attempt
    :type retries: int

    :param backoff_factor: determines the time between retries
    :type backoff_factor: float

    :param status_forcelist: API error codes triggering retries
    :type status_forcelist: Tuple[int, int,int]

    :param session: requests session for making an API call
    :type session: requests.Session


    Output
    :return session: requests session with retry logic applied
    :rtype: requests.Session
    """
    session = session or requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def map_permitted_license(license: str) -> str:
    """
    Check the license type and return whether it is permissive or not

    Input
    :param license: license type
    :type license: str


    Output
    :return permitted_license: Boelean indicating whether the license is permissive or not
    :rtype: str
    """
    if license in [
        "CC-BY",
        "cc by",
        "cc-by",
        "CC BY",
        "CC0",
        "cc0",
        "CC-BY-SA",
        "cc-by-sa",
        "CC BY SA",
        "cc by sa",
    ]:
        permitted_license = "Y"
    else:
        permitted_license = "N"
    return permitted_license
