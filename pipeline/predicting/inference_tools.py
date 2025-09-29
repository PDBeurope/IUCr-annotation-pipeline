import bs4
import en_core_sci_sm
from collections import OrderedDict
from typing import Any, Dict, List, Tuple

section_list = ["introduction",
                "materials|methods",
                "methods",
                "results",
                "discussion|interpretation",
                "conclusions",
                "display-objects"]

def split_into_sentences(paragraph: str) -> Dict[int, List[Any]]:
    """
    Split a paragraph into individual sentences using SciSpacy.
    
    Input

    :param paragraph: text to be split into sentences
    :type paragraph: str

    Output

    :return: sentences_to_return; a dictionary of individual sentences
    :rtype: Dict[int, List[Any]]
    """
    # using normal spacy
    # loading spacy and english scientific vocab
    nlp = en_core_sci_sm.load()
    # passing the raw text for a paragraph
    doc = nlp(paragraph)
    
    # making an empty list to add individual sentences from paragraph to
    sentences_to_return = OrderedDict()

    # iterate over the individual sentences in the paragraph as split by SciSpacy
    for i, sent_ in enumerate(doc.sents):

        # replace "new line" break with a white space
        sent = str(sent_).replace("\n", " ")
        
        # get the length of the current sentence to determine the position of the
        # stop character
        sent_length = len(sent)
        
        # assemble the sentence for appending
        sentence = [sent, 0, sent_length]
        # appending sentence to list for returning
        sentences_to_return[i] = sentence
        
    return sentences_to_return

def get_all_paragraphs(all_sections: List[bs4.element.Tag]) -> Dict[str, List[str]]:
    """
    Get all paragraphs with section titles.
    
    Input

    :param all_sections: list of sections containing paragraphs and titles
    :type all_sections: List[bs4.element.Tag]

    Output

    :return: para_dict; a dictionary with section titles as keys and list of paragraphs
    as values 
    :rtype: Dict[str, List[str]]
    """
    para_dict: Dict[str, List[str]] = {}
    # split the remaining text into sentences and attach to the dictionary
    for section in all_sections:
        if section.has_attr("sec-type") and section.get("sec-type") in section_list:
            parent_sec_type = section.get("sec-type")
            paragraphs = section.find_all("p")
            mini_list = []
            for p in paragraphs:
                plain = p.text
                mini_list.append(plain)
            para_dict[parent_sec_type] = mini_list

    return para_dict

def get_para_sentences_as_dict(paras_with_section: Dict[str, List[str]],
                               sentence_dict: Dict[int, Any],
                               sentence_counter: int) -> Dict[int, Any]:
    """
    Get individual sentences from all paragraphs with section titles and create unique
    sentence IDs.
    
    Input

    :param paras_with_section: dictionary of paragraphs and section titles
    :type paras_with_section: Dict[str, List[str]]

    :param sentence_dict: dictionary to append individual sentences to
    :type sentence_dict: Dict[int, Any]

    :param sentence_counter: sentence counter to create unique sentence IDs
    :type sentence_counter: int

    Output

    :return: sentence_dict; updated sentence_dict
    :rtype: Dict[int, Any]
    """
    sentence_dict: Dict[int, Any]
    section_keys = list(paras_with_section.keys())
    for k in section_keys:
        para_list = paras_with_section[k]
        for p in para_list:
            text_snippets = split_into_sentences(p)
            for ts in list(text_snippets.keys()):
                sentence = text_snippets[ts]
                snp_dict = {"sentence" : sentence[0],
                            "id" : sentence_counter,
                            "sent_start" : sentence[1],
                            "sent_end" : sentence[2],
                            "section" : k}
                sentence_dict[sentence_counter] = snp_dict
                sentence_counter = sentence_counter + 1
    return sentence_dict

def get_abstract_sentences_as_dict(abs_full: List[bs4.element.Tag],
                                   sentence_dict: Dict[int, Any],
                                   sentence_counter: int
                                   ) -> Tuple[Dict[int, Any], int]:
    """
    Get individual sentences from abstract and create unique
    sentence IDs.
    
    Input

    :param abs_full: pargraphs in abstract
    :type abs_full: List[bs4.element.Tag]

    :param sentence_dict: dictionary to append individual sentences to
    :type sentence_dict: Dict[int, Any]

    :param sentence_counter: sentence counter to create unique sentence IDs
    :type sentence_counter: int

    Output

    :return: sentence_dict; updated sentence_dict
    :rtype: Dict[int, Any]

    :return sentence_counter: sentence counter to create unique sentence IDs
    :rtype sentence_counter: int
    """
    sentence_counter: int
    sentence_dict: Dict[int, Any]
    for a in abs_full:

        paragraphs = a.find_all("p")

        for p in paragraphs:
            plain = p.text

            text_snippet = split_into_sentences(plain)

            for ts in list(text_snippet.keys()):
                sentence = text_snippet[ts]
                snp_dict = {"sentence" : sentence[0],
                            "id" : sentence_counter,
                            "sent_start" : sentence[1],
                            "sent_end" : sentence[2],
                            "section" : "abstract"}
                sentence_dict[sentence_counter] = snp_dict
                sentence_counter = sentence_counter + 1
    return sentence_dict, sentence_counter

def merge_with_same_spans(x_list: List[Tuple[Any]]) -> List[Tuple[Any]]:
    """
    Postprocessing to merge annotations.

    A little helper function to ensure that annotations with concecutive spans
    are merged into a single annotation.

    Input
    :param x_list: nested list of sentences
    :type x_list: List[Tuple[Any]]

    Output
    :return: merged_list; list of tuples with merged annotations
    :rtype: List[Tuple[Any]]
    """
    merged_list:List[List[str]] = []
    for sublist in x_list:
        if (merged_list
            and merged_list[-1][1] == sublist[0]
            and merged_list[-1][2] == sublist[2]):
            merged_list[-1][1] = sublist[1]
            merged_list[-1][3] += sublist[3]
        else:
            merged_list.append(sublist)

    return merged_list