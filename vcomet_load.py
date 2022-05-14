from asyncio import events
from database.milvus import MilvusAPI
from database.arangodb import DatabaseConnector
import torch
#import cv2


from vlm .clip_api import CLIP_API


class VCOMET_LOAD:
    def __init__(self):
        self.milvus_events = MilvusAPI(
            'milvus', 'vcomet_vit_l14_embedded_event', 'nebula_visualcomet', 768)
        self.milvus_places = MilvusAPI(
            'milvus', 'vcomet_vit_l14_embedded_place', 'nebula_visualcomet', 768)
        self.milvus_actions = MilvusAPI(
            'milvus', 'vcomet_vit_l14_embedded_actions', 'nebula_visualcomet', 768)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dbc = DatabaseConnector()
        self.vc_db = dbc.connect_db("nebula_visualcomet")
        self.vlmodel = CLIP_API(vlm_name='vit')

    def drop_load(self):
        self.milvus_events.drop_database()
        self.milvus_places.drop_database()
        self.milvus_actions.drop_database() 

    def load_vit_vcomet_place(self):
        query = 'FOR doc IN vcomet_kg RETURN DISTINCT doc.place'
        cursor = self.vc_db.aql.execute(query)
        places = []
        for vc in cursor:
            if len(vc.split()) < 9:  
                vector = self.vlmodel.clip_encode_text(vc)
                #print(vector)
                #print(len(vector))
                meta = {
                            'filename': 'none',
                            'movie_id':'none',
                            'nebula_movie_id': 'none',
                            'stage': 'none',
                            'frame_number': 'none',
                            'sentence': vc,
                            'vector': vector
                        }
                places.append(meta)
        return(places)
                #self.milvus_places.insert_vectors([vector], [meta])
                #print(meta)
                #input()
    
    def load_vit_vcomet_events(self):
        query = 'FOR doc IN vcomet_kg RETURN DISTINCT doc.event'
        cursor = self.vc_db.aql.execute(query)
        events = []
        for vc in cursor:
            if len(vc.split()) < 9:  
                vector = self.vlmodel.clip_encode_text(vc)
                #print(vector)
                #print(len(vector))
                meta = {
                            'filename': 'none',
                            'movie_id':'none',
                            'nebula_movie_id': 'none',
                            'stage': 'none',
                            'frame_number': 'none',
                            'sentence': vc,
                            'vector': vector
                        }
                events.append(meta)
                #self.milvus_events.insert_vectors([vector], [meta])
                #print(meta)
                #input()
        return(events)

    def load_vit_vcomet_actions(self):    
        query = 'FOR doc IN vcomet_kg RETURN DISTINCT doc'
        print(query)
        actions = []
        cursor = self.vc_db.aql.execute(query)    
        for doc in cursor:
            if 'intent' in doc:
                for intent in doc['intent']:
                    actions.append(intent)
            if 'before' in doc:
                for before in doc['before']:
                    actions.append(before)
            if 'after' in doc:
                for after in doc['after']:
                    actions.append(after)
        actions = list(dict.fromkeys(actions))
        actions_ = []
        for vc in actions:
            if len(vc.split()) < 9: 
                #print(vc) 
                vector = self.vlmodel.encode_text(vc, class_name='clip_vit')
                print(len(vector.tolist()[0]))
                meta = {
                            'filename': 'none',
                            'movie_id':'none',
                            'nebula_movie_id': 'none',
                            'stage': 'none',
                            'frame_number': 'none',
                            'sentence': vc,
                            'vector': vector
                        }
                #self.milvus_actions.insert_vectors([vector.tolist()[0]], [meta])
                actions_.append(meta)
        return (actions_)
           
def main():
    kg = VCOMET_LOAD()
    #kg.load_vit_vcomet_place()
    kg.load_vit_vcomet_actions()
   
    
if __name__ == "__main__":
    main()