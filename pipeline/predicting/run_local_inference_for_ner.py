from bs4 import BeautifulSoup
from collections import defaultdict
import argparse
import logging
import pathlib
from typing import Any, Dict, List
from optimum.pipelines import pipeline
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForTokenClassification
from pipeline.predicting.inference_tools import (
    merge_with_same_spans,
    get_all_paragraphs,
    get_para_sentences_as_dict,
    get_abstract_sentences_as_dict)
from pipeline.grounding.grounding import (
    term_grounding_with_epmc_json)
from pipeline.utils import (write_output_json,
                            expand_to_epmc_annotation,
                            get_article_license,
                            get_article_ids,
                            get_all_linked_structures,
                            get_article_title,
                            make_organism_dict_for_pdbs,
                            make_output_json)



def extract_sentence_level_details(soup: BeautifulSoup) -> Dict[int, Any]:
    """
    Turning paragraph sections into a sentence dictionary.

    Find the relevant text blocks in the XML file, split them into sentences,
    and add them to a dictionary along with a section label and a unique ID.

    Input
    :param soup: XML object of the publication
    :type soup: BeautifulSoup

    Output
    :return: sentence_dict; dictionary with unique sentence IDs as keys and a
    dictionary with sentence details as value
    :rtype: Dict[int, Any]
    """
    # create an empty dictionary to return
    sentence_dict: Dict[int, Any] = {}
    sentence_counter: int = 1

    # find all the sections and the abstract in the document
    all_sections = soup.find_all("sec")
    abs_full = soup.find_all("abstract")

    # get the sentences in the abstract
    sentence_dict, sentence_counter = get_abstract_sentences_as_dict(abs_full,
                                                                     sentence_dict,
                                                                     sentence_counter)

    # pair the paragraphs with the sections
    paras_with_section = get_all_paragraphs(all_sections)

    # get the sentences for the different sections
    sentence_dict = get_para_sentences_as_dict(paras_with_section,
                                               sentence_dict,
                                               sentence_counter)

    return sentence_dict

def get_ml_tagged_sentences(sentence_dict: Dict[int, Any],
                            model_path_quantised: str) -> Dict[int, Any]:
    """
    Iterate over all the sentences in the dictionary and try them with the
    NER model to get residue-level protein structure annotations.

    Input
    :param sentence_dict: dictionary containing the sentences
    :type sentence_dict: Dict[int, Any]

    :param model_path_quantised: path to quamtised model location
    :type model_path_quantised: str

    Output
    :return: sentence_dict; updated dictionary for sentences to have
    annotations
    :rtype: Dict[int, Any]
    """
    # initiate the model for prediction
    model_quantized = ORTModelForTokenClassification.from_pretrained(
                        model_path_quantised,
                        file_name = "model_quantized.onnx")
    # initiate the tokenizer
    tokenizer_quantized = AutoTokenizer.from_pretrained(model_path_quantised,
                                                        model_max_length = 512,
                                                        batch_size = 4,
                                                        truncation = True)
    # create the prediction pipeline
    ner_quantized = pipeline("token-classification",
                             model = model_quantized,
                             tokenizer = tokenizer_quantized,
                             aggregation_strategy = "first")
    
    # create a sentence list to iterate over during prediction
    all_sentences = []

    for e in sentence_dict:
        sent = sentence_dict[e]["sentence"]
        all_sentences.append(sent)

    assert len(sentence_dict) == len(all_sentences)
    
    # set number of sentences used for prediction and make a new dictionary
    # that will return the annotations
    no_sentences = 8
    ML_annotations = defaultdict(list)

    # split the list of sentences into smaller batches to iterate over
    for i in range((len(all_sentences) + no_sentences - 1) // no_sentences ):
        batch_sentences = all_sentences[i * no_sentences:(i + 1) * no_sentences]

        # make the actual prediction for a batch of sentences

        pred = ner_quantized(batch_sentences)

        count = 0
        for all_ent in pred:
            my_sentence = batch_sentences[count]
            if all_ent:
                x_list_=[]
                # iterate over all predicted annotations for the batch and
                # make a separate entry for each sentence in the batch and
                # pair them with the respective list of annotations in a
                # dictionary
                for ent in all_ent:
                    x_list_.append([ent["start"], ent["end"], ent["entity_group"],
                                    my_sentence[ent["start"]:ent["end"]], ent["score"]])
                ML_annotations[my_sentence].extend(merge_with_same_spans(x_list_))
            count = count + 1

    # iterate over the sentences with their respective annotations in the newly
    # created dictionary; find matching sentences in the input dictionary and
    # extend the latter by adding the annotations; return the extended intput
    # dictionary for sentences
    for s in ML_annotations.keys():
        annos = ML_annotations[s]
        for e in sentence_dict:
            if s == sentence_dict[e]["sentence"]:
                sentence_dict[e]["annotations"] = annos

    return sentence_dict


# this will get matches
def get_sentences_matches_tags(sentences_tags: Dict[str, Any]
                               ) -> Dict[str, List[Dict[str, Any]]]:
    """
    Re-arrange the sentence dictionary so that the annotations for a sentence
    (used as a key) are collected in a list of dictionaries and linked the
    sentence. The annotations previously as tuples are now converted into
    dictionaries.

    Input

    :param sentence_dict: dictionary of all the sentences
    :type sentence_dict: dict

    Output

    :return: matches; dictionary with one entry per annotation using the
    sentence as key
    :rtype: Dict[str, List[Dict[str, Any]]]
    """
    # make an empty dictionary to attach sentences with annnotations to
    matches = defaultdict(list)
    # iterate over the input dictionary and find the entries that have the 
    # key "annotations"
    for i in sentences_tags:
        entry = sentences_tags[i]
        sentence = entry["sentence"]

        if "annotations" in list(entry.keys()):
            # for each annotation found in the list of the "annotation"
            # key make a new entry to the empty dictionry; when making
            # a new entry use the sentence as the key and add a dictionary
            # with the annotation details as the value; the finally returned
            # dictionary has repeat mention of the same key as many
            # annotations have been found in the sentence
            for a in entry["annotations"]:
                mini_dict = {}
                mini_dict["exact"] = a[3]
                mini_dict["type"] = a[2]
                mini_dict["ai_score"] = a[4]
                mini_dict["char_start"] = a[0]
                mini_dict["char_end"] = a[1]
                mini_dict["id"] = entry["id"]
                mini_dict["section"] = entry["section"]
                matches[sentence].append(mini_dict)

    return matches

def generate_final_json(soup: BeautifulSoup,
                        match: Dict[str, List[Dict[str, Any]]],
                        annotator: str) -> Dict[str, Any]:
    """
    Run this function to generate a final dictionary for the current
    publication 

    Input

    :param soup: XML of current publication
    :type soup: BeautifulSoup

    :param match_ml: dictionary of all annotations found for the current 
                     publication on a per-sentence basis
    :type match_ml: Dict[str, List[Dict[str, Any]]]

    :param annotator: name of annotator used for prediction to be added
                      to the annotations
    :type annotator: str


    Output

    :return: json_generated; a dictionary for the current publication in
             EuropePMC standard containing all the sentences with their
             respective annotations on a per-sentence basis; also contains
             additional metadata for downstream purposes
    :rtype: dict
    """
    # make a new, empty JSON for later return;
    # cerate some variables and lists
    json_generated: Dict[str, Any] = {}
    pmcid = ""
    pmid = ""
    doi = ""
    publisher_id = ""
    publisher_license = ""
    title = ""
    all_pdbs = []
    primary_pdbs = []
    secondary_pdbs = []

    # get the title for the paper
    try:
        title = get_article_title(soup)
    except:
        logging.error(f"No title found")

    # try to find original license
    try:
        publisher_license = get_article_license(soup)
    except Exception as e:
        logging.error(f"""No license-type
                        {e}""")
        
    # try to find a form of a unique article identifier
    try:
        pmid, pmcid, doi, publisher_id = get_article_ids(soup, title)
    except Exception as e:
        logging.error(f"""No article-id tag
                        {e}""")
    # check for supplementary material to identify the PDB entry codes for which
    # the paper is the primary citation; add those IDs to a list;
    # add the primary IDs to the JSON
    try:
        all_pdbs, primary_pdbs, secondary_pdbs = get_all_linked_structures(soup, pmid, pmcid)
    except:
        logging.error(f"Couldn't find primary PDB entries for publication")

    try:
        pdb_prot_species_dict = make_organism_dict_for_pdbs(all_pdbs)
    except:
        pdb_prot_species_dict = {}

    #  first check whether both pmc and pmids are available 
    #  if available; make sure PubMed Central ID starts with "PMC"
    if len(pmcid)>0 and len(pmid)>0:
        # check if pmcid starts with PMC
        if pmcid.startswith("PMC"):
            pmcid = pmcid
        else:
            pmcid = "PMC" + pmcid
    # add the additional metadata of various unique identifiers,
    # different lists of PDB entries and identified species for
    # all PDB entries in the text; check first for PubMed
    # Central ID, then PubMed ID and if neither than write a
    # generic output; last option when when no PMC or PubMed ID
    # coulod be found
    if pmcid:
        json_generated = make_output_json("PMC",
                     pmcid,
                     pmid,
                     doi,
                     publisher_id,
                     publisher_license,
                     title,
                     primary_pdbs,
                     secondary_pdbs,
                     all_pdbs,
                     pdb_prot_species_dict)

    elif pmid:
        json_generated = make_output_json("MED",
                     pmcid,
                     pmid,
                     doi,
                     publisher_id,
                     publisher_license,
                     title,
                     primary_pdbs,
                     secondary_pdbs,
                     all_pdbs,
                     pdb_prot_species_dict)

    else:
        json_generated = make_output_json("",
                     pmcid,
                     pmid,
                     doi,
                     publisher_id,
                     publisher_license,
                     title,
                     primary_pdbs,
                     secondary_pdbs,
                     all_pdbs,
                     pdb_prot_species_dict)
        
    # turn all the details into a proper JSON; add a key "anns"
    # to the future output JSON; key has a list as value in which
    # each element is a dictionary for an annotation
    try:
        interested_sentences = generate_interested_sentences_in_json_format(match,
                                                                            annotator)
        if interested_sentences:
            json_generated["anns"] = interested_sentences
        else:
            json_generated["anns"] = None
    
    except Exception as e:
        logging.error(f"""error in pub date or generate_interested_sentences_in_json_format
                        {e}""")

    return json_generated

# generate dictionary for matches and co-occurances, section and other scores
def generate_interested_sentences_in_json_format(match: Dict[str, List[Dict[str, Any]]],
                                                 annotator: str
                                                 ) -> List[Dict[str, Any]]:
    """
    Run this function to unpick the per-sentence dictionary with the annotations
    of the current publication. Turn each entry into a EuropePMC style annotation.

    Input

    :param match: dictionary of all annotations found for the current 
                     publication on a per-sentence basis
    :type match: Dict[str, Any]

    :param annotator: name of annotator used for prediction to be added
                      to the annotations
    :type annotator: str


    Output

    :return: interested_sentences; list for all annotations found for
             current publication on a per-sentence basis in EuropePMC standard
    :rtype: List[Dict]
    """
    # make an empty list to add all the annotation dictionaries to
    interested_sentences = []
    # iterate over all the entries in the input dictionary; each entry is
    # an annotation with the sentence it was found in as key
    for each_sentence in match.keys():
        all_matches = match[each_sentence]
        if all_matches:
            for m in all_matches:
                # create an empty dictionary for the current annotation in the
                # current sentence to create EuropePMC annotation
                minidict = expand_to_epmc_annotation(each_sentence, m, annotator)
                # list to be returned
                interested_sentences.append(minidict)
    # return all the annotations found for the current paper as a list
    # with each element in the list being a dictionary in EuropePMC
    # standard for a single annotation
    return interested_sentences


def process_each_file_in_job_per_article(file_path: str,
                                         model: str,
                                         annotator: str,
                                         val_dir: str,
                                         map_dir: str,
                                         out: str,
                                         out_name: str = "") -> None:
    """
    Main worker function to use a quantised model from ONNX framework to make
    predictions for structure specific named entities. Will run the following
    steps:
    1) open and read the JATS XML file
    2) find text blocks and split them into sentences with unique identifiers
       and keep track of sections
    3) make batches of 8 sentences and predict on them
    4) identify and collect sentences with annotations and add additional
       details
    5) turn the annotations into a EuropePMC standard JSON
    6) send the JSON through a term grounding step for annotations of type
       'residue_name_number', 'mutant' and 'protein'
    Finally, write a EuropePMC standard JSON file to disk containing grounded
    annotations

    Input

    :param file_path: path to input publications in XML
    :type file_path: str

    :param model: path to quantised, predictive model; model is expected to be
                  named 'model_quantized.onnx'
    :type model: str

    :param val_dir: path to directory containing validation XML files
    :type val_dir: str

    :param map_dir: path to directory containing SIFTS mapping XML files
    :type map_dir: str

    :param out: path to output directory writing JSONL object with annotations
                to
    :type out: str

    :param out_name: user-specified output file name; will be expanded to '.json'
    :type out_name: str

    :param annotator: name of annotator to be added to the predicted
                      annotations to be able to identify which algorithm and
                      version provided the annotations, e.g. 'autoannotator_v2.1_quant'
    :type annotator: str

    Output

    :return: JSON file for a publication containing all sentences with annotations
             and those of entity type 'residue_name_number', 'mutant' and 'protein'
             will also have been grounded and linked to a reference source

    :rtype: Dict[str, Any]
    """
    # check that the output directory exists
    pathlib.Path(out).mkdir(parents = True,
                            exist_ok = True)
    # try to open the input XML file
    try:
        logging.info(f"Opening input JATS XML file")
        with open(file_path) as x:
            data = x.read()
        logging.info(f"Reading XML content")
        # create the XML object for processing
        xml_soup = BeautifulSoup(data, "xml")
        # extract the sentence level details from the XML object
        # and return the isolated sentences with labels for section and
        # a unique sentence ID in a discionary
        logging.info(f"Splitting XML content into sentences with unique identifiers")
        all_sentences_dict = extract_sentence_level_details(xml_soup)
        # run the sentences in the dictionary through the prediction algorithn 
        # and only extract the sentences that have had predictions returned 
        # for them
        logging.info(f"Predicting on sentences batches to create annotations")
        mltag_sentences = get_ml_tagged_sentences(all_sentences_dict, model)
        # expand the tagged sentences so that there is a dictionary entry for
        # each annotation that had been created; ignore sentences that do not
        # have any predictions
        logging.info(f"Identify sentences with annotations and add additional \n"
                     f"details to annotations")
        ml_match = get_sentences_matches_tags(mltag_sentences)
        # use all the annotations to create an output JSON file following the
        # EuropePMC annotation standard
        try:
            logging.info(f"Creating final JSON for output")
            ml_json = generate_final_json(xml_soup,
                                          ml_match,
                                          annotator)
        except Exception as e:
            logging.error(f"""Exception in generating final json : {file_path}
                            {e}""")
        try:
            ml_json = term_grounding_with_epmc_json(ml_json,
                                                    val_dir,
                                                    map_dir)
        except Exception as e:
            logging.error(f"""Exception in generating final json after grounding: {file_path}
                            {e}""")

        # write the JSON file depending on what type of identifier has been
        # found for the file or use a custom name
        # use custom name
        if out_name != "":
            outfile = out + out_name + ".json"
            write_output_json(ml_json, outfile)
        # use PubMed Central ID
        elif ml_json["pmcid"] != "":
            pmcid = ml_json["pmcid"]
            outfile = out + "epmc_style_annos_" + pmcid + ".json"
            write_output_json(ml_json, outfile)
        # use PubMed ID
        elif ml_json["pmid"] != "":
            pmid = ml_json["pmid"]
            outfile = out + "epmc_style_annos_" + pmid + ".json"
            write_output_json(ml_json, outfile)
        # use publisher ID
        elif ml_json["publisher_id"] != "":
            publisher_id = ml_json["publisher_id"]
            outfile = out + "epmc_style_annos_" + publisher_id + ".json"
            write_output_json(ml_json, outfile)
        # if no ID available write a generic file
        else:
            # Writing to JSON
            outfile = out + "epmc_style_annos.json"
            write_output_json(ml_json, outfile)
    except:
            logging.error("error processing, so skipping file")


def main():
    logging.basicConfig(level = logging.INFO)

    parser = argparse.ArgumentParser(
                        description = """This script will process JATS XML files
                        from IUCr journals to extract annotations from residue-level
                        named entity recognistion. The 20 different entity types that
                        are annotated are: 'chemical', 'complex_assembly', 'evidence',
                        'experimental_method', 'gene', 'mutant', 'oligomeric_state',
                        'protein', 'protein_state', 'protein_type', 'ptm', 'residue_name',
                        'residue_name_number', 'residue_number', 'residue_range',
                        'site', 'species', 'structure_element', 'taxonomy_domain',
                        'bond_interaction'""")
    parser.add_argument(
                        "--jats-xml",
                        help = "JATS XML file for which to predict annotations",
                        dest = "jats_xml"
                        )
    parser.add_argument(
                        "--model-dir",
                        help = "directory containing core files of trained model",
                        dest = "model_dir"
                        )
    parser.add_argument(
                        "--model-name",
                        help = "string giving the algorithm's name, e.g. autoannotator_v2.1_quant",
                        dest = "model_name"
                        )
    parser.add_argument(
                        "--val-dir",
                        help = "directory containing validation XML files",
                        dest = "val_dir",
                        metavar = "PATH"
                        )
    parser.add_argument(
                        "--map-dir",
                        help = "directory containing mapping XML files",
                        dest = "map_dir",
                        metavar = "PATH"
                        )
    parser.add_argument(
                        "--output-dir",
                        help = "directory to write the JSON file with EuropePMC style annotations",
                        dest = "output_dir"
                        )
    parser.add_argument(
                        "--out-name",
                        help = "user-defined output file name expanded to .json",
                        default = "",
                        dest = "out_name"
    )
    
    args = parser.parse_args()
                      
    process_each_file_in_job_per_article(args.jats_xml,
                                         args.model_dir,
                                         args.model_name,
                                         args.val_dir,
                                         args.map_dir,
                                         args.output_dir,
                                         args.out_name)
    
    logging.info(args.jats_xml + " : NER finished!")


if __name__ == "__main__":
    main()