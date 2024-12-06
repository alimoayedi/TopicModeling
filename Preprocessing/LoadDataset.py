import re
import os


class LoadReutersDataset:

    def __init__(self, data_path):
        self.data_path = data_path
        self.topics_dic = {}
        self.places_dic = {}
        self.peoples_dic = {}
        self.orgs_dic = {}
        self.exchanges_dic = {}
        self.companies_dic = {}
        self.bodies_dic = {}

    def load(self, place=False, people=False, orgs=False, exchange=False, companies=False):
        for filename in os.listdir(self.data_path):
            if filename.startswith('reut2-'):
                file_path = os.path.join(self.data_path, filename)
                print('*** opening file: ', file_path)
                with open(file_path, 'r', encoding='latin-1') as file:
                    content = file.read()
                    # Use regex to extract the text content between <BODY> and </BODY> tags
                    # for each document we have a pair in form(type) of tuple
                    match_documents = re.findall(r'<REUTERS([^>]*)>(.*?)</REUTERS>', content, re.DOTALL)
                    for id_part, doc_part in match_documents:  # each entry is in form of tuple (pair)
                        doc_id = re.search(r'NEWID="([^"]+)"', id_part).group(1)
                        match_topic = re.search(r'<TOPICS[^>]*>(.*?)</TOPICS>', doc_part, re.DOTALL)
                        match_place = re.search(r'<PLACES[^>]*>(.*?)</PLACES>', doc_part, re.DOTALL)
                        match_people = re.search(r'<PEOPLE[^>]*>(.*?)</PEOPLE>', doc_part, re.DOTALL)
                        match_orgs = re.search(r'<ORGS[^>]*>(.*?)</ORGS>', doc_part, re.DOTALL)
                        match_exchange = re.search(r'<EXCHANGES[^>]*>(.*?)</EXCHANGES>', doc_part, re.DOTALL)
                        match_companies = re.search(r'<COMPANIES[^>]*>(.*?)</COMPANIES>', doc_part, re.DOTALL)
                        match_body = re.search(r'<BODY[^>]*>(.*?)</BODY>', doc_part, re.DOTALL)

                        if match_body:
                            doc_topics = re.findall(r'<D[^>]*>(.*?)</D>', match_topic.group(1), re.DOTALL)
                            doc_places = re.findall(r'<D[^>]*>(.*?)</D>', match_place.group(1), re.DOTALL)
                            doc_peoples = re.findall(r'<D[^>]*>(.*?)</D>', match_people.group(1), re.DOTALL)
                            doc_orgs = re.findall(r'<D[^>]*>(.*?)</D>', match_orgs.group(1), re.DOTALL)
                            doc_exchanges = re.findall(r'<D[^>]*>(.*?)</D>', match_exchange.group(1), re.DOTALL)
                            doc_companies = re.findall(r'<D[^>]*>(.*?)</D>', match_companies.group(1), re.DOTALL)
                            doc_bodies = re.sub(r'<[^>]+>', '', match_body.group(1))

                            self.topics_dic[doc_id] = doc_topics
                            self.places_dic[doc_id] = doc_places
                            self.peoples_dic[doc_id] = doc_peoples
                            self.orgs_dic[doc_id] = doc_orgs
                            self.exchanges_dic[doc_id] = doc_exchanges
                            self.companies_dic[doc_id] = doc_companies
                            self.bodies_dic[doc_id] = doc_bodies

        return self.bodies_dic, self.topics_dic, self.places_dic, self.peoples_dic, self.orgs_dic, self.exchanges_dic, \
            self.companies_dic

