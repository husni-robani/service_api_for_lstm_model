# the way to know path of the project
from pathlib import Path
# print(Path.cwd())

import torch
import sys
# append another directory
sys.path.append('/Users/bani/dev/project_TA/service_api_for_lstm_model/model')
sys.path.append('/Users/bani/dev/project_TA/service_api_for_lstm_model/utils')
# custom dependencies
import model as module_model
import preprocessing

class Predict():
    def __init__(self):
        self.model = module_model.LSTMClassifier()
        self.model.load_state_dict(torch.load('./model/pretrained_model/lstm_model.pt'))
        self.study_program_encoder = {
            "Akuntansi": 0,
            "Manajemen": 1,
            "Teknik Informatika": 2,
            "Bahasa Inggris": 3,
            "DKV": 4
        }
    def make_predict(self, text):
        # preprocess the text to numerical representation
        preprocess = preprocessing.TextPreprocessing()
        text = preprocess.preprocess_text(text)

        # predict with model
        self.model.eval()
        with torch.inference_mode():
            output = self.model(text)
            predicted_class = torch.argmax(output, dim=1).item()
        
        # decode study program code
        for study_program, code in self.study_program_encoder.items():
            if predicted_class == code:
                result = study_program
        
        return result, predicted_class, output
